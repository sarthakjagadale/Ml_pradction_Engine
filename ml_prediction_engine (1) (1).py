"""
Portfolio ML Prediction Engine — v3.0
======================================
Fixes vs v2:
No data leakage  — scaler.fit() only on TRAIN split, then transform test
Time-series safe — lag/MA computed BEFORE split, sequences built correctly
Weekends excluded — pd.bdate_range() throughout
Multi-asset       — Stocks, Mutual Funds, Gold, Silver + crypto-adjacent
Clean ml_predictions JSON — backend-ready schema

Assets modelled (calibrated GBM + real sector vols):
  Stocks       : AAPL, MSFT, RELIANCE.NS, TCS.NS, INFY.NS, HDFC.NS
  Mutual Funds : MIPAX, VFIAX, HDFC_TOP100, SBI_BLUECHIP
  Commodities  : Gold (GC=F), Silver (SI=F)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import uuid
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── Asset Catalogue ─────────────────────────────────────────────────────────
ASSETS = {
    # id                name                    type           mu     sigma  price_base
    "AAPL":           ("Apple Inc.",            "Stock",       0.28,  0.25,  189.00),
    "MSFT":           ("Microsoft Corp.",       "Stock",       0.25,  0.22,  415.00),
    "RELIANCE.NS":    ("Reliance Industries",   "Stock",       0.18,  0.24,  2920.00),
    "TCS.NS":         ("Tata Consultancy Svcs", "Stock",       0.16,  0.20,  3780.00),
    "INFY.NS":        ("Infosys Ltd.",          "Stock",       0.14,  0.22,  1820.00),
    "HDFCBANK.NS":        ("HDFC Bank",             "Stock",       0.15,  0.21,  1620.00),
    #"MIPAX":          ("Muni Intermediate ETF", "MutualFund",  0.07,  0.06,  14.50),
    #"VFIAX":          ("Vanguard 500 Index",    "MutualFund",  0.12,  0.16,  450.00),
    #"HDFC_TOP100":    ("HDFC Top 100 Fund",     "MutualFund",  0.14,  0.15,  890.00),
    #"SBI_BLUECHIP":   ("SBI Bluechip Fund",     "MutualFund",  0.13,  0.14,  72.00),
    "GC=F":           ("Gold Futures",          "Commodity",   0.12,  0.14,  3320.00),
    "SI=F":           ("Silver Futures",        "Commodity",   0.10,  0.22,  32.50),
}

MODEL_VERSION = "v3.0"
HISTORY_DAYS  = 756          # ~3 years of trading days
PREDICT_DAYS  = 30
SEQ_LEN       = 20
TEST_FRAC     = 0.15


# ─── 1. Synthetic Market Data (GBM) ──────────────────────────────────────────
def generate_prices(mu: float, sigma: float, s0: float, n_days: int,
                    seed: int = 0) -> np.ndarray:
    """Geometric Brownian Motion on business days."""
    rng = np.random.RandomState(seed)
    dt  = 1 / 252
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.randn(n_days)
    return s0 * np.cumprod(np.exp(returns))


def build_dataframe(asset_id: str) -> pd.DataFrame:

    print(f"     Downloading real data for {asset_id}...")

    df = yf.download(
        asset_id,
        period="3y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No market data found for {asset_id}")

    df = df.reset_index()

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean column names
    df.columns = [
        str(col).lower().replace(" ", "_")
        for col in df.columns
    ]

    # Keep required columns
    df = df[[
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]]

    # Remove missing values
    df = df.dropna().reset_index(drop=True)

    return df


# ─── 2. Feature Engineering (leak-safe) ──────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Returns
    d["daily_return"] = d["close"].pct_change()
    # Safe log return
    safe_close = d["close"].replace(0, np.nan)

    ratio = safe_close / safe_close.shift(1)

    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    ratio = ratio.clip(lower=1e-9)

    d["log_return"] = np.log(ratio)

    d["log_return"] = d["log_return"].fillna(0)

    # Moving averages (computed on full history — no leakage; these are backward-looking)
    for w in [5, 10, 20]:
        d[f"ma_{w}"] = d["close"].rolling(w).mean()
    for w in [12, 26]:
        d[f"ema_{w}"] = d["close"].ewm(span=w, adjust=False).mean()

    # Volatility
    for w in [5, 10, 20]:
        d[f"vol_{w}"] = d["daily_return"].rolling(w).std()

    # Lag features
    for lag in range(1, 6):
        d[f"lag_{lag}"] = d["close"].shift(lag)

    # Range & volume change
    d["price_range"] = d["high"] - d["low"]
    # Safe volume change
    safe_volume = d["volume"].replace(0, np.nan)

    d["vol_change"] = safe_volume.pct_change()
    d["vol_change"] = d["vol_change"].fillna(0)

    # RSI-14
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # Bollinger
    bb_mid        = d["close"].rolling(20).mean()
    bb_std        = d["close"].rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (bb_mid + 1e-9)

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]     = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()

    # Remove infinities
    d = d.replace([np.inf, -np.inf], np.nan)

    # Clip extremely large values
    numeric_cols = d.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:

        d[col] = d[col].clip(
            lower=-1e6,
            upper=1e6
        )

    # Remove NaNs
    d = d.dropna().reset_index(drop=True)

    return d


# ─── 3. Sequence Builder & Train/Test Split ───────────────────────────────────
FEATURE_COLS = [
    "close", "open", "high", "low",
    "daily_return", #"log_return",
    "ma_5", "ma_10", "ma_20", "ema_12", "ema_26",
    "vol_5", "vol_10", "vol_20",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "price_range", #"vol_change",
    "rsi_14", "bb_width", "macd", "macd_sig",
]


def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len: i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ─── 4. Model — Ridge on flattened sequences ──────────────────────────────────
def train_model(df_feat: pd.DataFrame):
    raw_feat   = df_feat[FEATURE_COLS].values
    raw_target = df_feat["close"].values
    # Convert to float64
    raw_feat = raw_feat.astype(np.float64)
    raw_target = raw_target.astype(np.float64)

    # Replace invalid values
    raw_feat = np.nan_to_num(
        raw_feat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    raw_target = np.nan_to_num(
        raw_target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Clip extreme values
    raw_feat = np.clip(
        raw_feat,
        -1e6,
        1e6
    )

    raw_target = np.clip(
        raw_target,
        -1e6,
        1e6
    )
    
    # Final safety cleaning before scaling

    raw_feat = pd.DataFrame(raw_feat)

    raw_feat = raw_feat.replace(
        [np.inf, -np.inf],
        np.nan
    )

    raw_feat = raw_feat.fillna(0)

    raw_feat = raw_feat.clip(
        lower=-1e6,
        upper=1e6
    )

    raw_feat = raw_feat.values

    # Clean target
    raw_target = pd.Series(raw_target)

    raw_target = raw_target.replace(
        [np.inf, -np.inf],
        np.nan
    )

    raw_target = raw_target.fillna(0)

    raw_target = raw_target.clip(
        lower=-1e6,
        upper=1e6
    )

    raw_target = raw_target.values

    n        = len(raw_feat)
    tr_end   = int(n * (1 - TEST_FRAC))   # ← split index on RAW rows

    # ✅ Fit scalers ONLY on train rows
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_feat              = np.zeros_like(raw_feat, dtype=float)
    scaled_feat[:tr_end]     = feat_scaler.fit_transform(raw_feat[:tr_end])
    scaled_feat[tr_end:]     = feat_scaler.transform(raw_feat[tr_end:])

    scaled_target            = np.zeros((n, 1), dtype=float)
    scaled_target[:tr_end]   = target_scaler.fit_transform(raw_target[:tr_end].reshape(-1,1))
    scaled_target[tr_end:]   = target_scaler.transform(raw_target[tr_end:].reshape(-1,1))
    scaled_target            = scaled_target.flatten()

    # Build sequences AFTER scaling
    X_all, y_all = build_sequences(scaled_feat, scaled_target, SEQ_LEN)

    # Re-derive split index in sequence space
    seq_tr_end = int(len(X_all) * (1 - TEST_FRAC))
    X_tr, X_te = X_all[:seq_tr_end], X_all[seq_tr_end:]
    y_tr, y_te = y_all[:seq_tr_end], y_all[seq_tr_end:]

    # Flatten for Ridge (lightweight, no TF needed)
    X_tr_f = X_tr.reshape(len(X_tr), -1)
    X_te_f = X_te.reshape(len(X_te), -1)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_tr_f, y_tr)

    y_pred_sc = model.predict(X_te_f)
    y_pred = target_scaler.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()
    y_true = target_scaler.inverse_transform(y_te.reshape(-1,1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return model, feat_scaler, target_scaler, mape, mae, rmse, y_true, y_pred


def recompute_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute all indicators after adding a new predicted candle.
    """
    
    df = history_df.copy()

    # Returns
    df["daily_return"] = df["close"].pct_change()
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()

    # EMA
    for w in [12, 26]:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # Volatility
    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["daily_return"].rolling(w).std()

    # Lag features
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Range & volume
    df["price_range"] = df["high"] - df["low"]
    df["vol_change"]  = df["volume"].pct_change()

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()

    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # Bollinger
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"]) /
        (bb_mid + 1e-9)
    )

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df

# ─── 5. 30-Day Forecast ───────────────────────────────────────────────────────
def forecast(model, df_feat, feat_scaler, target_scaler):

    history = df_feat.copy()
    future_prices = []

    for _ in range(PREDICT_DAYS):
        
        

        # Take latest feature rows
        latest_features = history[FEATURE_COLS].tail(SEQ_LEN)

        # Scale
        scaled_features = feat_scaler.transform(latest_features)

        # Flatten sequence
        X_in = scaled_features.flatten().reshape(1, -1)

        # Predict scaled price
        pred_scaled = model.predict(X_in)[0]

        # Convert back to real price
        pred_price = target_scaler.inverse_transform(
            [[pred_scaled]]
        )[0, 0]

        future_prices.append(float(pred_price))

        # Create new synthetic candle
        last_row = history.iloc[-1].copy()

        new_date = last_row["date"] + timedelta(days=1)

        new_row = {
            "date": new_date,
            "open": pred_price,
            "high": pred_price * 1.01,
            "low": pred_price * 0.99,
            "close": pred_price,
            "volume": last_row["volume"]
        }

        # Append new row
        history = pd.concat([
            history,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        # Recompute ALL indicators dynamically
        history = recompute_features(history)

        # Remove NaNs
        history = history.dropna().reset_index(drop=True)

    return future_prices

# ─── 6. Risk / Confidence Scoring ─────────────────────────────────────────────
def risk_score(df_feat: pd.DataFrame, asset_type: str) -> dict:
    recent_vol = float(df_feat["vol_20"].iloc[-1])
    ann_vol    = recent_vol * np.sqrt(252)

    hist_vol     = df_feat["vol_20"].dropna()
    vol_min, vol_max = float(hist_vol.min()), float(hist_vol.max())
    vol_score    = float(np.clip((recent_vol - vol_min) / (vol_max - vol_min + 1e-9), 0, 1))

    # Asset-type adjusted thresholds
    if asset_type == "MutualFund":
        thresholds = (0.08, 0.15)
    elif asset_type == "Commodity":
        thresholds = (0.12, 0.22)
    else:
        thresholds = (0.15, 0.30)

    if ann_vol < thresholds[0]:   risk_level = "Low"
    elif ann_vol < thresholds[1]: risk_level = "Medium"
    else:                         risk_level = "High"

    return {
        "risk_level":       risk_level,
        "volatility_score": round(vol_score, 4),
        "annual_vol_pct":   round(ann_vol * 100, 2),
    }


def confidence(mape: float, vol_score: float) -> float:
    acc  = float(np.clip(1 - mape / 100, 0, 1))
    stab = float(1 - vol_score)
    return round(float(np.clip(0.6 * acc + 0.4 * stab, 0, 1)), 4)


def loss_probability(df_feat: pd.DataFrame, future_prices: list[float]) -> float:
    """Probability that price 30d out is BELOW current price (from MC simulation)."""
    current = float(df_feat["close"].iloc[-1])
    hist_returns = df_feat["daily_return"].dropna().values
    mu    = float(np.mean(hist_returns))
    sigma = float(np.std(hist_returns))

    rng   = np.random.RandomState(99)
    simulated_final = []
    for _ in range(2000):
        r = rng.normal(mu, sigma, PREDICT_DAYS)
        simulated_final.append(current * np.prod(1 + r))

    return round(float(np.mean(np.array(simulated_final) < current)), 4)

def feature_importance(model):

    booster = model.get_booster()

    score_dict = booster.get_score(
        importance_type="gain"
    )

    importance_data = []

    for key, value in score_dict.items():

        feature_index = int(key.replace("f", ""))

        real_feature = FEATURE_COLS[
            feature_index % len(FEATURE_COLS)
        ]

        importance_data.append({
            "feature": real_feature,
            "importance": round(float(value), 4)
        })

    importance_data = sorted(
        importance_data,
        key=lambda x: x["importance"],
        reverse=True
    )

    return importance_data[:5]
# ─── 7. Build ml_predictions Record ──────────────────────────────────────────
def run_asset(asset_id: str) -> dict:
    name, atype, mu, sigma, s0 = ASSETS[asset_id]
    print(f"  ▶ {asset_id:<18} {name}")

    df_raw  = build_dataframe(asset_id)
    df_feat = engineer_features(df_raw)
    
    if len(df_feat) < 100:
        raise ValueError(
            f"Not enough cleaned feature rows for {asset_id}"
        )

    model, f_sc, t_sc, mape, mae, rmse, y_true, y_pred = train_model(df_feat)

    future_prices = forecast(model, df_feat, f_sc, t_sc)
    
    top_features = feature_importance(model)

    current_price = float(df_feat["close"].iloc[-1])
    pred_30d      = future_prices[-1]
    exp_return    = round((pred_30d - current_price) / current_price * 100, 4)

    rinfo         = risk_score(df_feat, atype)
    conf          = confidence(mape, rinfo["volatility_score"])
    loss_prob     = loss_probability(df_feat, future_prices)

    # risk_score → 0-100 int
    raw_risk_num  = round(rinfo["volatility_score"] * 100, 1)

    # Technical signals
    latest_rsi    = float(df_feat["rsi_14"].iloc[-1])
    macd_val      = float(df_feat["macd"].iloc[-1])
    macd_sig_val  = float(df_feat["macd_sig"].iloc[-1])
    rsi_signal    = "Overbought" if latest_rsi > 70 else ("Oversold" if latest_rsi < 30 else "Neutral")
    macd_signal   = "Bullish" if macd_val > macd_sig_val else "Bearish"

    trend = ("Uptrend" if exp_return > 1.0 else "Downtrend" if exp_return < -1.0 else "Stable")

    if (
        exp_return >= 10
        and conf >= 0.85
        and loss_prob <= 0.30
        and raw_risk_num <= 40
    ):
        recommendation = "Strong Buy"

    elif (
        exp_return >= 5
        and conf >= 0.75
        and loss_prob <= 0.40
        and raw_risk_num <= 60
    ):
        recommendation = "Buy"

    elif (
        exp_return >= 2
        and conf >= 0.60
        and loss_prob <= 0.50
    ):
        recommendation = "Hold"

    elif (
        exp_return < 0
        and loss_prob >= 0.60
    ):
        recommendation = "Strong Sell"

    else:
        recommendation = "Sell"

    last_date     = df_feat["date"].iloc[-1]
    future_dates  = pd.bdate_range(start=last_date + timedelta(days=1), periods=PREDICT_DAYS)
    hist_series   = [
        {"date": str(d.date()), "price": round(p, 4)}
        for d, p in zip(df_feat["date"].iloc[-90:], df_feat["close"].iloc[-90:])
    ]
    forecast_series = [
        {"date": str(d.date()), "price": round(p, 4)}
        for d, p in zip(future_dates, future_prices)
    ]

    return {
        # ── ml_predictions table columns ──────────────────────────
        "asset_id":          asset_id,
        "expected_return":   exp_return,         # % over 30 days
        "risk_score":        raw_risk_num,        # 0–100
        "loss_probability":  loss_prob,           # 0–1
        "confidence_score":  conf,                # 0–1
        "generated_at":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version":     MODEL_VERSION,

        # ── extended metadata (for risk/liquidation engines) ───────
        "asset_name":        name,
        "asset_type":        atype,
        "current_price":     round(current_price, 4),
        "predicted_price_30d": round(pred_30d, 4),
        "trend":             trend,
        "risk_level":        rinfo["risk_level"],
        "annual_vol_pct":    rinfo["annual_vol_pct"],
        "recommendation":    recommendation,
        "top_drivers": top_features,
        "technical": {
            "rsi_14":      round(latest_rsi, 2),
            "rsi_signal":  rsi_signal,
            "macd_signal": macd_signal,
        },
        "model_metrics": {
            "mape": round(mape, 4),
            "mae":  round(mae, 4),
            "rmse": round(rmse, 4),
        },
        "chart_data": {
            "historical": hist_series,
            "forecast":   forecast_series,
        },
    }


# ─── 8. Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Portfolio ML Prediction Engine — v3.0")
    print("=" * 60)

    records = []
    for asset_id in ASSETS:
        try:
            rec = run_asset(asset_id)
            records.append(rec)
        except Exception as e:
            print(f"  ✗ {asset_id}: {e}")

    # ── ml_predictions table (clean, backend-ready) ──
    ml_predictions = [
        {
            "asset_id":         r["asset_id"],
            "expected_return":  r["expected_return"],
            "risk_score":       r["risk_score"],
            "loss_probability": r["loss_probability"],
            "confidence_score": r["confidence_score"],
            "generated_at":     r["generated_at"],
            "model_version":    r["model_version"],
        }
        for r in records
    ]

    output = {
        "schema_version": "3.0",
        "generated_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version":  MODEL_VERSION,
        "total_assets":   len(records),
        "ml_predictions": ml_predictions,    # ← direct DB insert
        "full_insights":  records,            # ← rich data for dashboard/engines
    }

    with open("ml_predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅  Done!")
    print(f"   Assets processed : {len(records)}")
    print(f"   Output           : ml_predictions.json\n")
    print("─" * 60)
    print(f"  {'ASSET':<18} {'RET%':>7} {'RISK':>6} {'LOSS_P':>8} {'CONF':>7}  REC")
    print("─" * 60)
    for r in records:
        print(f"  {r['asset_id']:<18} {r['expected_return']:>+6.2f}%  "
              f"{r['risk_score']:>5.1f}  {r['loss_probability']:>7.3f}  "
              f"{r['confidence_score']:>6.3f}  {r['recommendation']}")
    print("─" * 60)
    return output


if __name__ == "__main__":
    result = main()
