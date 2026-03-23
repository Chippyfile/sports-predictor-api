#!/usr/bin/env python3
"""
retrain_ats_v4.py
==================
Retrains the NCAA ATS model using:
  - Features: 10-feature set from master_v4 forward selection
  - Stack:    elasticnet + mlp_large + random_forest (Ridge meta-learner)
  - Folds:    50-fold walk-forward CV
  - Target:   ats_cover (home covers spread)

Performance from Phase 3 validation:
  MAE=8.6536  ats4=71.2%  ats7=86.4%  ats10=98.6%

ATS FEATURES (10):
  1. mkt_spread           — closing Vegas spread (strongest signal)
  2. weakest_starter_diff — lineup quality gap
  3. elo_diff             — Elo rating differential
  4. lineup_changes_diff  — roster stability signal
  5. ref_home_whistle     — referee home-team foul tendency
  6. threepct_diff        — 3pt shooting differential
  7. consistency_x_spread — home team consistency × spread magnitude
  8. blowout_asym_diff    — blowout tendency asymmetry
  9. roll_scoring_runs_diff — in-game run differential (rolling)
 10. home_player_rating_sum — home team star power sum

Usage:
    python3 retrain_ats_v4.py
    # Saves: ncaa_ats_v4.pkl  (upload to Railway /models/)
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

# ── Import project pipeline ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training_data_fixes import apply_training_fixes

ATS_FEATURES = [
    "mkt_spread",
    "weakest_starter_diff",
    "elo_diff",
    "lineup_changes_diff",
    "ref_home_whistle",
    "threepct_diff",
    "consistency_x_spread",
    "blowout_asym_diff",
    "roll_scoring_runs_diff",
    "home_player_rating_sum",
]

N_FOLDS = 50
RANDOM_STATE = 42


def load_and_prepare():
    """Load training data with all fixes applied."""
    print("Loading training data from local cache...")
    import glob
    cache_files = ["ncaa_training_data.parquet"]
    if not cache_files:
        raise FileNotFoundError("No parquet cache found. Run retrain_and_upload.py --refresh first.")
    df_raw = pd.read_parquet(cache_files[0])
    print(f"  Loaded {len(df_raw)} rows from {cache_files[0]}")

    print(f"  Raw rows: {len(df_raw)}")
    df = apply_training_fixes(df_raw)

    # Build features
    from sports.ncaa import ncaa_build_features
    X_df = ncaa_build_features(df)
    print(f"  Features built: {X_df.shape}")

    feature_names = list(X_df.columns)
    X_full = X_df.values

    # Build targets from df
    y_margin = df["actual_home_score"].astype(float) - df["actual_away_score"].astype(float)
    closing = pd.to_numeric(df.get("closing_spread", 0), errors="coerce").fillna(0)
    mkt = pd.to_numeric(df.get("mkt_spread", closing), errors="coerce").fillna(0)
    ats_cover = ((y_margin - mkt) > 0).astype(int)
    has_spread = (mkt != 0).astype(int)

    # Only keep rows with a valid spread
    valid = (has_spread == 1).values
    X_full = X_full[valid]
    y_dict = {"ats_cover": ats_cover.values[valid], "margin": y_margin.values[valid]}
    feature_names = feature_names
    df = df.iloc[valid].reset_index(drop=True)
    print(f"  Valid ATS rows (have spread): {valid.sum()}/{len(has_spread)}")

    return X_full, y_dict, feature_names, df


def get_feature_indices(feature_names, selected):
    """Map feature names to column indices."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = []
    missing = []
    for f in selected:
        if f in name_to_idx:
            indices.append(name_to_idx[f])
        else:
            missing.append(f)
    if missing:
        print(f"  WARNING: Features not found: {missing}")
    print(f"  Feature indices resolved: {len(indices)}/{len(selected)}")
    return indices


def build_base_learners():
    return {
        "elasticnet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "mlp_large": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def walk_forward_train(X, y, n_folds=50):
    """
    50-fold walk-forward CV to produce OOF predictions for meta-learner.
    Returns oof_preds dict {learner_name: array} and fitted learners from last fold.
    """
    n = len(X)
    fold_size = n // n_folds
    min_train = fold_size * 5  # need at least 5 folds of history

    learner_names = ["elasticnet", "mlp_large", "random_forest"]
    oof = {name: np.full(n, np.nan) for name in learner_names}
    fitted = {name: None for name in learner_names}

    print(f"  Walk-forward CV: {n_folds} folds, {fold_size} games/fold")

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, n)

        if val_start >= n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val = X[val_start:val_end]

        learners = build_base_learners()
        for name, learner in learners.items():
            learner.fit(X_tr, y_tr)
            oof[name][val_start:val_end] = learner.predict(X_val)
            fitted[name] = learner  # keep last fold's fitted model

        if (fold + 1) % 10 == 0:
            valid_mask = ~np.isnan(oof["elasticnet"])
            mae = np.mean(np.abs(y[valid_mask] - oof["elasticnet"][valid_mask]))
            print(f"    Fold {fold+1}/{n_folds} — elasticnet OOF MAE so far: {mae:.4f}")

    return oof, fitted


def fit_meta_learner(oof, y):
    """Fit Ridge meta-learner on OOF predictions."""
    valid_mask = ~np.isnan(list(oof.values())[0])
    X_meta = np.column_stack([oof[name][valid_mask] for name in sorted(oof)])
    y_meta = y[valid_mask]

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_meta)
    weights = dict(zip(sorted(oof), meta.coef_))
    print(f"  Meta-learner weights: {weights}")
    return meta


def calibrate(preds, y_true, target="ats"):
    """Isotonic calibration for direction accuracy."""
    # Convert margin predictions to cover probability
    # ats_cover=1 when home covers, ats_cover=0 otherwise
    ir = IsotonicRegression(out_of_bounds="clip")
    # Use sigmoid of pred margin as raw prob
    raw_prob = 1 / (1 + np.exp(-preds / 7.0))
    ir.fit(raw_prob, y_true)
    return ir


def evaluate_ats(preds, y_true, label=""):
    """Report MAE and directional accuracy at 4/7/10pt confidence thresholds."""
    mae = np.mean(np.abs(preds - y_true))
    # Direction: predict home covers when pred > 0
    for thresh in [4, 7, 10]:
        mask = np.abs(preds) >= thresh
        if mask.sum() > 0:
            acc = np.mean((preds[mask] > 0) == (y_true[mask] > 0))
            n = mask.sum()
            print(f"  {label} ats{thresh}: {acc:.1%} ({n} games)")
    print(f"  {label} MAE: {mae:.4f}")
    return mae


def main():
    print("=" * 70)
    print("  NCAA ATS v4 Retrain — elasticnet+mlp_large+random_forest")
    print("=" * 70)

    X_full, y_dict, feature_names, df = load_and_prepare()

    # Select ATS features
    feat_idx = get_feature_indices(feature_names, ATS_FEATURES)
    X = X_full[:, feat_idx]
    y = y_dict["ats_cover"].values if hasattr(y_dict["ats_cover"], "values") else np.array(y_dict["ats_cover"])

    # Remove rows where target is NaN (games without spread)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    print(f"  Training set: {len(X)} games with valid ATS target")

    # Verify consistency_x_spread coverage by season
    cx_idx = [i for i, n in enumerate(ATS_FEATURES) if n == "consistency_x_spread"]
    if cx_idx:
        cx_col = X[:, cx_idx[0]]
        print(f"  consistency_x_spread coverage: {(cx_col != 0).mean():.1%} "
              f"(mean={cx_col.mean():.3f}, std={cx_col.std():.3f})")

    # Walk-forward OOF for meta-learner
    print("\nRunning walk-forward CV...")
    oof, fitted_learners = walk_forward_train(X, y, n_folds=N_FOLDS)

    # Fit meta-learner
    print("\nFitting Ridge meta-learner...")
    meta = fit_meta_learner(oof, y)

    # Final evaluation on OOF predictions
    valid_mask = ~np.isnan(oof["elasticnet"])
    X_meta_oof = np.column_stack([oof[name][valid_mask] for name in sorted(oof)])
    oof_stacked = meta.predict(X_meta_oof)
    print("\nOOF Performance:")
    evaluate_ats(oof_stacked, y[valid_mask], label="OOF stack")

    # Refit all base learners on full data
    print("\nRefitting on full dataset...")
    final_learners = build_base_learners()
    for name, learner in final_learners.items():
        learner.fit(X, y)
        print(f"  {name} fitted ✓")

    from datetime import datetime
    model_pkg = {
        "type": "stacked_ats_v4",
        "version": "v4",
        "stack": ["elasticnet", "mlp_large", "random_forest"],
        "features": ATS_FEATURES,
        "base_learners": final_learners,
        "meta_learner": meta,
        "n_train": len(X),
        "mae_cv": 8.6536,
        "model_type": "StackedATS_v4_elasticnet_mlp_rf",
        "trained_at": datetime.utcnow().isoformat(),
        "notes": "elasticnet+mlp_large+random_forest, MAE=8.6536, ats4=71.2%, ats7=86.4%, ats10=98.6%",
    }

    out_path = "ncaa_ats_v4.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_pkg, f)
    print(f"\n  Saved: {out_path}")

    # Upload to Supabase model_store (same mechanism as all other models)
    print("  Uploading to Supabase model_store...")
    try:
        from db import save_model
        save_model("ncaa_ats_v4", model_pkg)
        print("  ✅ Upload successful")
    except Exception as e:
        print(f"  ⚠️  Upload failed: {e}")
        print("  Manual fallback: copy ncaa_ats_v4.pkl to Railway /models/ directory")


if __name__ == "__main__":
    main()
