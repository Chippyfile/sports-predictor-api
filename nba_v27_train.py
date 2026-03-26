#!/usr/bin/env python3
"""
nba_v27_train.py — Train NBA v27 model (Lasso, 38 features)

Architecture: Lasso alpha=0.1 regressor + isotonic-calibrated LogisticRegression
Feature selection: L1 (alpha=0.2) from 102 v27 candidates → 38 survivors

Usage:
    python3 nba_v27_train.py                # Train + save locally
    python3 nba_v27_train.py --deploy       # Train + save to Supabase model_store
    python3 nba_v27_train.py --validate     # Train + full walk-forward validation
"""
import sys, os, json, time, argparse, warnings, copy
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, brier_score_loss

from nba_build_features_v27 import load_training_data, build_features

# ── Configuration ──
LASSO_ALPHA_SELECT = 0.2     # For feature selection (38 features)
LASSO_ALPHA_TRAIN = 0.1      # For final model (slightly less sparse for better MAE)
N_CV_FOLDS = 10
MODEL_NAME = "nba"            # Matches existing save_model("nba", ...) convention

# The 38 features selected by Lasso alpha=0.2 (from sweep)
V27_FEATURE_SET = [
    "lineup_value_diff", "win_pct_diff", "scoring_hhi_diff", "espn_pregame_wp",
    "ceiling_diff", "matchup_efg", "ml_implied_spread", "sharp_spread_signal",
    "efg_diff", "opp_suppression_diff", "net_rtg_diff", "steals_to_diff",
    "threepct_diff", "b2b_diff", "ftpct_diff", "ou_gap",
    "roll_dreb_diff", "ts_regression_diff", "roll_paint_pts_diff", "ref_home_whistle",
    "opp_ppg_diff", "roll_max_run_avg", "away_is_public_team", "away_after_loss",
    "games_last_14_diff", "h2h_total_games", "three_pt_regression_diff", "games_diff",
    "ref_foul_proxy", "roll_fast_break_diff", "crowd_pct", "matchup_to",
    "overround", "roll_ft_trip_rate_diff", "home_after_loss", "rest_diff",
    "spread_juice_imbalance", "vig_uncertainty",
]


def walk_forward_oof(X, y, model_fn, n_splits=10):
    """Walk-forward OOF predictions for unbiased evaluation."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    oof = np.full(n, np.nan)

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_start >= n:
            break
        model = model_fn()
        model.fit(X[:train_end], y[:train_end])
        oof[val_start:val_end] = model.predict(X[val_start:val_end])

    return oof


def train_v27(deploy=False, validate=False):
    """Train NBA v27 model."""
    print("=" * 70)
    print("  NBA v27 TRAINING — Lasso (38 features)")
    print("=" * 70)

    # ── Load data ──
    t0 = time.time()
    df = load_training_data("nba_training_data.parquet")

    # ── Build all 102 features ──
    print("\nBuilding v27 features...")
    X_full, all_feature_names = build_features(df)
    print(f"  Built {len(all_feature_names)} features for {len(df)} games")

    # ── Select 38 features ──
    # Verify all expected features exist
    missing = [f for f in V27_FEATURE_SET if f not in all_feature_names]
    if missing:
        print(f"\n  WARNING: {len(missing)} features not found in v27 builder: {missing}")
        print(f"  Using intersection...")
        feature_cols = [f for f in V27_FEATURE_SET if f in all_feature_names]
    else:
        feature_cols = V27_FEATURE_SET.copy()

    X = X_full[feature_cols].values
    y_margin = df["target_margin"].values
    y_win = (y_margin > 0).astype(int)
    market_spread = df["market_spread_home"].values

    print(f"  Selected {len(feature_cols)} features")
    print(f"  Target: home margin (range {y_margin.min():.0f} to {y_margin.max():.0f})")

    # ── Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train regressor (Lasso) ──
    print(f"\nTraining Lasso (alpha={LASSO_ALPHA_TRAIN})...")
    reg = Lasso(alpha=LASSO_ALPHA_TRAIN, max_iter=10000, random_state=42)
    reg.fit(X_scaled, y_margin)

    # Active features after training
    active_mask = np.abs(reg.coef_) > 1e-8
    n_active = active_mask.sum()
    print(f"  Active features: {n_active}/{len(feature_cols)}")
    for i, (name, coef) in enumerate(sorted(
            zip(feature_cols, reg.coef_), key=lambda x: abs(x[1]), reverse=True)):
        if abs(coef) > 1e-8:
            print(f"    {i+1:>3}. {name:<35} coef={coef:>+.4f}")

    # ── Walk-forward MAE ──
    print(f"\n{N_CV_FOLDS}-fold walk-forward MAE...")
    oof_preds = walk_forward_oof(
        X_scaled, y_margin,
        lambda: Lasso(alpha=LASSO_ALPHA_TRAIN, max_iter=10000, random_state=42),
        n_splits=N_CV_FOLDS
    )
    valid_mask = ~np.isnan(oof_preds)
    cv_mae = mean_absolute_error(y_margin[valid_mask], oof_preds[valid_mask])
    print(f"  Walk-forward MAE: {cv_mae:.3f}")

    # ── Bias correction ──
    residuals = y_margin[valid_mask] - oof_preds[valid_mask]
    bias_correction = float(np.mean(residuals))
    print(f"  Bias correction: {bias_correction:+.4f}")

    # ── Train classifier + isotonic calibration ──
    print(f"\nTraining classifier (LogisticRegression + isotonic)...")
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), cv=min(5, N_CV_FOLDS)
    )
    clf.fit(X_scaled, y_win)

    # OOF win probabilities for isotonic
    clf_oof = cross_val_predict(
        LogisticRegression(max_iter=1000), X_scaled, y_win, cv=5, method="predict_proba"
    )[:, 1]

    # Isotonic calibration
    isotonic = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    isotonic.fit(clf_oof, y_win)

    # Brier score
    calibrated_probs = isotonic.predict(clf_oof)
    brier = brier_score_loss(y_win, calibrated_probs)
    print(f"  Brier score (isotonic): {brier:.4f}")
    print(f"  Win accuracy: {(np.round(calibrated_probs) == y_win).mean()*100:.1f}%")

    # ── ATS evaluation on OOF ──
    model_spread_oof = -oof_preds[valid_mask]
    edge = model_spread_oof - market_spread[valid_mask]
    actual_margin = y_margin[valid_mask]
    for threshold in [0, 4, 7, 10]:
        mask = np.abs(edge) >= threshold
        n_picks = mask.sum()
        if n_picks < 20:
            continue
        model_home_covers = edge < -threshold
        actual_home_covers = actual_margin + market_spread[valid_mask] > 0
        correct = (model_home_covers == actual_home_covers)[mask]
        pct = correct.mean() * 100
        roi = (pct / 100 * 1.91 - 1) * 100
        print(f"  ATS {threshold}+: {pct:.1f}% ({n_picks} picks, ROI {roi:+.1f}%)")

    # ── Build bundle ──
    bundle = {
        "scaler": scaler,
        "reg": reg,
        "clf": clf,
        "isotonic": isotonic,
        "feature_cols": feature_cols,
        "n_train": len(df),
        "mae_cv": round(cv_mae, 3),
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "Lasso_v27",
        "model_version": "v27",
        "lasso_alpha": LASSO_ALPHA_TRAIN,
        "n_features": len(feature_cols),
        "n_active_features": int(n_active),
        "bias_correction": bias_correction,
        "brier_score": round(brier, 4),
    }

    # ── Save ──
    if deploy:
        from db import save_model
        save_model(MODEL_NAME, bundle)
        print(f"\n  Model saved to Supabase as '{MODEL_NAME}'")
    else:
        import joblib
        os.makedirs("models", exist_ok=True)
        path = "models/nba_v27.pkl"
        joblib.dump(bundle, path)
        print(f"\n  Model saved locally to {path}")
        print(f"  Size: {os.path.getsize(path)/1024:.0f} KB")

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    # ── Validation (optional) ──
    if validate:
        print(f"\n{'='*70}")
        print(f"  FULL VALIDATION — {N_CV_FOLDS}-fold walk-forward")
        print(f"{'='*70}")

        # More detailed ATS analysis
        print(f"\n  ATS by edge bucket:")
        for lo, hi in [(0, 2), (2, 4), (4, 7), (7, 10), (10, 999)]:
            mask = (np.abs(edge) >= lo) & (np.abs(edge) < hi)
            n_picks = mask.sum()
            if n_picks < 10:
                continue
            model_home_covers = edge < -lo  # simplification for bucket
            actual_home_covers = actual_margin + market_spread[valid_mask] > 0
            correct = (model_home_covers == actual_home_covers)[mask]
            pct = correct.mean() * 100
            print(f"    Edge [{lo},{hi}): {pct:.1f}% ({n_picks} picks)")

        # Calibration by predicted margin bucket
        print(f"\n  Calibration by predicted margin:")
        for lo, hi in [(-999, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 999)]:
            mask = (oof_preds[valid_mask] >= lo) & (oof_preds[valid_mask] < hi)
            n = mask.sum()
            if n < 20:
                continue
            pred_avg = oof_preds[valid_mask][mask].mean()
            actual_avg = y_margin[valid_mask][mask].mean()
            print(f"    [{lo:>4},{hi:>4}): pred={pred_avg:>+6.1f}, actual={actual_avg:>+6.1f}, "
                  f"diff={actual_avg-pred_avg:>+5.1f} ({n} games)")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"  NBA v27 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Model:    Lasso (alpha={LASSO_ALPHA_TRAIN})")
    print(f"  Features: {len(feature_cols)} selected, {n_active} active after L1")
    print(f"  Games:    {len(df)}")
    print(f"  MAE:      {cv_mae:.3f}")
    print(f"  Brier:    {brier:.4f}")
    print(f"  Bias:     {bias_correction:+.4f}")

    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA v27 Training")
    parser.add_argument("--deploy", action="store_true", help="Save to Supabase")
    parser.add_argument("--validate", action="store_true", help="Full validation")
    args = parser.parse_args()

    train_v27(deploy=args.deploy, validate=args.validate)
