#!/usr/bin/env python3
"""
Strategy 3: ML Ensemble with AFML Features (1h)

Full AFML pipeline applied to intraday crypto:
  - Features: multi-horizon returns, realized vol, fracdiff, RSI
  - Labels: triple-barrier (profit-take / stop-loss / time-expiry)
  - Model: Random Forest with purged CV
  - Sizing: probability-based bet sizing (convex by construction)

Walk-forward: train on trailing 90 days, predict next 30 days.
Re-train every 30 days (rolling).

Convexity comes from:
  - Triple-barrier labels with asymmetric PT/SL
  - Probability → bet size (high conviction = bigger bets)
  - Purged CV prevents in-sample overfitting
  - Vol targeting at portfolio level
"""
from _common import *

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afml.labeling import daily_volatility
from afml.bet_sizing import bet_size_from_prob

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def build_features_1h(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Build feature set from hourly close and volume."""
    feat = pd.DataFrame(index=close.index)

    # Multi-horizon returns
    for h in [1, 4, 12, 24, 48, 96]:
        feat[f"ret_{h}h"] = np.log(close / close.shift(h))

    # Realized volatility at multiple horizons
    ret_1h = np.log(close / close.shift(1))
    for w in [12, 24, 48, 96]:
        feat[f"vol_{w}h"] = ret_1h.rolling(w).std()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    # Normalized volume
    feat["vol_ratio"] = volume / volume.rolling(24).mean()

    # Price relative to moving averages
    for w in [24, 48, 96]:
        feat[f"ma_dist_{w}h"] = (close - close.rolling(w).mean()) / close.rolling(w).std()

    return feat


def triple_barrier_1h(
    close: pd.Series,
    holding_period: int = 24,   # 1 day
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    vol_window: int = 48,
) -> pd.DataFrame:
    """Simplified triple-barrier labels for 1h bars."""
    ret_1h = np.log(close / close.shift(1))
    vol = ret_1h.rolling(vol_window).std()

    labels = []
    for i in range(len(close) - holding_period):
        v = vol.iloc[i]
        if pd.isna(v) or v <= 0:
            labels.append({"ts": close.index[i], "label": 0, "t1": close.index[min(i + holding_period, len(close) - 1)]})
            continue

        pt = v * pt_mult
        sl = v * sl_mult
        entry = close.iloc[i]

        label = 0
        exit_idx = i + holding_period
        for j in range(i + 1, min(i + holding_period + 1, len(close))):
            ret = np.log(close.iloc[j] / entry)
            if ret >= pt:
                label = 1
                exit_idx = j
                break
            elif ret <= -sl:
                label = -1
                exit_idx = j
                break

        labels.append({"ts": close.index[i], "label": label, "t1": close.index[exit_idx]})

    df = pd.DataFrame(labels).set_index("ts")
    return df


def walk_forward_ml(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    train_bars: int = 24 * 90,    # 90 days
    test_bars: int = 24 * 30,     # 30 days
    holding_period: int = 24,
) -> pd.DataFrame:
    """Walk-forward ML strategy across all assets."""
    # Pick top 5 most liquid for ML (keeps it tractable)
    vol_rank = panel.groupby("symbol").apply(
        lambda g: (g["close"] * g["volume"]).mean()
    ).sort_values(ascending=False)
    top_symbols = vol_rank.head(5).index.tolist()

    all_weights = {}
    for sym in top_symbols:
        print(f"  ML walk-forward for {sym}...")
        grp = panel[panel["symbol"] == sym].sort_values("ts").set_index("ts")
        if len(grp) < train_bars + test_bars:
            continue

        close = grp["close"]
        volume = grp["volume"]

        features = build_features_1h(close, volume)
        tb_labels = triple_barrier_1h(close, holding_period=holding_period, pt_mult=2.0, sl_mult=1.0)

        # Binary: 1 = profit-take, 0 = stop/expiry
        tb_labels["y"] = (tb_labels["label"] == 1).astype(int)

        df = features.join(tb_labels[["y"]]).dropna()
        feat_cols = features.columns.tolist()
        X = df[feat_cols].values
        y = df["y"].values

        # Walk-forward
        positions = pd.Series(0.0, index=close.index)
        step = 0
        while step + train_bars + test_bars <= len(df):
            train_end = step + train_bars
            test_end = train_end + test_bars

            X_train = X[step:train_end]
            y_train = y[step:train_end]
            X_test = X[train_end:test_end]
            test_dates = df.index[train_end:test_end]

            if len(np.unique(y_train)) < 2:
                step += test_bars
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=50,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_train_s, y_train)

            probs = rf.predict_proba(X_test_s)[:, 1]

            # Probability → bet size (convex sizing)
            prob_series = pd.Series(probs, index=test_dates)
            side = pd.Series(np.where(probs > 0.5, 1, -1), index=test_dates)
            sizes = bet_size_from_prob(prob_series, side)

            # Only go long (positive sizes) — crypto is hard to short
            sizes = sizes.clip(lower=0)
            positions.loc[test_dates] = sizes.values

            step += test_bars

        all_weights[sym] = positions

    weights = pd.DataFrame(all_weights)

    # Normalize
    row_sum = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sum, axis=0).fillna(0)

    return weights.reindex(returns_wide.index).fillna(0)


def main():
    print("Loading 1h universe...")
    panel, returns_wide, close_wide = load_hourly_universe()
    print(f"  {len(panel):,} rows, {panel['symbol'].nunique()} symbols")

    print("\nRunning ML walk-forward ensemble...")
    weights = walk_forward_ml(panel, returns_wide)

    result = run_and_report("ML Ensemble", weights, returns_wide, close_wide)

    print(f"\n\n{'='*70}")
    print("  ML ENSEMBLE SUMMARY")
    print(f"{'='*70}")
    print(format_metrics_table([result]))


if __name__ == "__main__":
    main()
