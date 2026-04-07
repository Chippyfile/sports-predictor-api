#!/usr/bin/env python3
"""
nba_ou_retrain.py — Train NBA Over/Under total prediction model
================================================================
Uses the same 55 features as v27 margin model, but targets total points.
Architecture: CatBoost → simple and effective for totals.

Usage:
    python3 nba_ou_retrain.py              # Train + evaluate
    python3 nba_ou_retrain.py --upload     # Train + upload to Supabase as 'nba_ou'
"""
import sys, os, time, warnings, pickle, io
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: CatBoost not available, falling back to GBM")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Reuse v27 feature builder
from nba_build_features_v27 import load_training_data, build_features

# v27 validated feature set
V27_FEATURES = [
    "elo_diff", "win_pct_diff", "net_rating_diff", "ppg_diff", "opp_ppg_diff",
    "fgpct_diff", "threepct_diff", "ftpct_diff", "orb_pct_diff", "fta_rate_diff",
    "ato_ratio_diff", "steals_diff", "blocks_diff", "to_margin_diff",
    "opp_fgpct_diff", "opp_threepct_diff", "form_diff", "rest_diff",
    "tempo_avg", "market_spread", "spread_vs_market", "has_market",
    "margin_trend_diff", "margin_var_diff", "margin_accel_diff",
    "margin_skew_diff", "momentum_halflife_diff", "streak_diff",
    "win_aging_diff", "pyth_residual_diff", "pyth_luck_diff",
    "games_14d_diff", "season_pct", "common_opp_margin_diff",
    "opp_quality_form_diff", "ats_rolling_diff",
    "lineup_value_diff", "scoring_entropy_diff", "ceiling_diff",
    "efg_diff", "opp_efg_diff", "ts_diff",
    "roll_dreb_diff", "roll_paint_pts_diff", "roll_max_run_diff",
    "roll_fast_break_diff", "roll_ft_trip_rate_diff",
    "espn_pregame_wp", "crowd_pct", "ref_home_whistle", "ref_foul_rate",
    "away_travel", "altitude_factor",
    "is_early_season", "reverse_line_movement",
]

# Additional features useful for O/U that are available in the feature builder
# tempo_avg is already included — it's the #1 predictor for totals
OU_EXTRA_FEATURES = [
    "market_total",      # Vegas O/U line (strongest signal)
    "steals_to_diff",    # turnover quality affects pace/scoring
]


def main():
    upload = "--upload" in sys.argv

    print("=" * 70)
    print("  NBA O/U MODEL — Total Points Prediction")
    print("  Using v27 features + market_total")
    print("=" * 70)

    # ── Load and build features ──
    df = load_training_data("nba_training_data.parquet")
    X_all, all_feature_names = build_features(df)

    # Target: total points
    df["target_total"] = df["actual_home_score"] + df["actual_away_score"]
    y_total = df["target_total"].values

    # Feature set: v27 features + O/U extras
    ou_features = V27_FEATURES + [f for f in OU_EXTRA_FEATURES if f in all_feature_names]
    available = [f for f in ou_features if f in X_all.columns]
    missing = [f for f in ou_features if f not in X_all.columns]
    if missing:
        print(f"\n  Missing features (will skip): {missing}")

    X = X_all[available].values
    feature_names = available
    print(f"\n  Features: {len(feature_names)}")
    print(f"  Games: {len(X)}")
    print(f"  Target mean: {y_total.mean():.1f}, std: {y_total.std():.1f}")

    # ── Walk-forward validation ──
    print(f"\n  Walk-forward validation (30-fold)...")
    n_folds = 30
    fold_size = len(X) // (n_folds + 3)
    min_train = fold_size * 3

    all_preds = np.full(len(X), np.nan)
    all_true = y_total.copy()
    t0 = time.time()

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, len(X))
        if train_end >= len(X):
            break

        X_tr, y_tr = X[:train_end], y_total[:train_end]
        X_te = X[train_end:test_end]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if HAS_CAT:
            model = CatBoostRegressor(
                depth=4, iterations=600, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0
            )
        else:
            model = GradientBoostingRegressor(
                max_depth=4, n_estimators=300, learning_rate=0.05, random_state=42
            )

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
        all_preds[train_end:test_end] = preds

        if (fold + 1) % 10 == 0:
            print(f"    Fold {fold+1}/{n_folds} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Walk-forward complete in {elapsed:.0f}s")

    # ── Evaluate ──
    valid = ~np.isnan(all_preds)
    preds_v = all_preds[valid]
    true_v = all_true[valid]
    mae = np.mean(np.abs(preds_v - true_v))
    print(f"\n  Walk-forward MAE: {mae:.3f}")
    print(f"  Mean predicted: {preds_v.mean():.1f}, Mean actual: {true_v.mean():.1f}")
    print(f"  Bias: {(preds_v.mean() - true_v.mean()):.2f}")

    # ── O/U accuracy at edge thresholds ──
    # Need market total for this
    mkt_total_idx = feature_names.index("market_total") if "market_total" in feature_names else None
    if mkt_total_idx is not None:
        mkt_totals = X[valid, mkt_total_idx]
        has_mkt = mkt_totals > 0

        if has_mkt.sum() > 100:
            p_mkt = preds_v[has_mkt]
            t_mkt = true_v[has_mkt]
            m_mkt = mkt_totals[has_mkt]

            model_over = p_mkt > m_mkt
            actual_over = t_mkt > m_mkt
            # Exclude pushes
            not_push = t_mkt != m_mkt
            ou_correct = (model_over == actual_over) & not_push
            ou_total = not_push

            print(f"\n  === O/U ACCURACY BY EDGE ===")
            print(f"  {'Threshold':>10s}  {'Games':>6s}  {'Acc':>6s}  {'ROI':>8s}")
            print(f"  {'-'*38}")

            for thresh in [0, 2, 4, 5, 6, 7, 8, 10, 12]:
                edge = np.abs(p_mkt - m_mkt)
                mask = (edge >= thresh) & ou_total
                if mask.sum() >= 10:
                    acc = ou_correct[mask].sum() / mask.sum() * 100
                    roi = (acc / 100 * 1.909 - 1) * 100  # -110 odds
                    print(f"  {thresh:>8d}+  {mask.sum():>6d}  {acc:>5.1f}%  {roi:>+7.1f}%")

    # ── Train production model ──
    print(f"\n  Training production model on all {len(X)} games...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if HAS_CAT:
        prod_model = CatBoostRegressor(
            depth=4, iterations=600, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=42, verbose=0
        )
    else:
        prod_model = GradientBoostingRegressor(
            max_depth=4, n_estimators=300, learning_rate=0.05, random_state=42
        )

    prod_model.fit(X_s, y_total)
    in_sample_mae = np.mean(np.abs(prod_model.predict(X_s) - y_total))
    print(f"  In-sample MAE: {in_sample_mae:.3f}")

    # Compute bias correction (mean residual from walk-forward)
    bias = float(np.mean(true_v - preds_v))
    print(f"  Bias correction: {bias:+.2f}")

    # ── Save ──
    bundle = {
        "reg": prod_model,
        "scaler": scaler,
        "ou_feature_cols": feature_names,
        "bias_correction": bias,
        "cv_mae": round(mae, 3),
        "n_train": len(X),
        "model_type": f"CatBoost_d4_600_OU",
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    local_path = "nba_ou_model.pkl"
    with open(local_path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    sz = os.path.getsize(local_path)
    print(f"\n  Saved: {local_path} ({sz // 1024} KB)")

    # ── Upload ──
    if upload:
        print("\n  Uploading to Supabase as 'nba_ou'...")
        try:
            from db import sb_upsert_model
            sb_upsert_model("nba_ou", bundle)
            print("  ✅ Uploaded to Supabase model_store as 'nba_ou'")
        except ImportError:
            # Manual upload
            import requests, base64, json
            try:
                from config import SUPABASE_URL, SUPABASE_KEY
            except ImportError:
                SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
                SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

            buf = io.BytesIO()
            pickle.dump(bundle, buf, protocol=4)
            b64 = base64.b64encode(buf.getvalue()).decode()

            meta = {
                "mae_cv": bundle["cv_mae"],
                "n_train": bundle["n_train"],
                "model_type": bundle["model_type"],
                "size_bytes": len(buf.getvalue()),
                "trained_at": bundle["trained_at"],
                "bias_correction": bundle["bias_correction"],
            }

            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/model_store",
                headers=headers,
                json={"name": "nba_ou", "data": b64, "metadata": meta},
            )
            if resp.status_code < 300:
                print(f"  ✅ Uploaded to Supabase ({len(buf.getvalue()) // 1024} KB)")
            else:
                print(f"  ❌ Upload failed: {resp.status_code} {resp.text[:200]}")

    print(f"\n  {'='*60}")
    print(f"  O/U MODEL COMPLETE")
    print(f"  Features: {len(feature_names)}")
    print(f"  Walk-forward MAE: {mae:.3f}")
    print(f"  Bias correction: {bias:+.2f}")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
