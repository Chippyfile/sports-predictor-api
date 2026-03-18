"""
TRUE WALK-FORWARD BACKTEST
===========================
For each season, trains the model on ALL prior seasons only,
then predicts the held-out season. No future data leakage.

This answers: "How well would this model have performed in real time?"
"""

import os, sys, json, copy, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from ml_utils import StackedRegressor, StackedClassifier

print("=" * 70)
print("  TRUE WALK-FORWARD BACKTEST (retrain per season)")
print("=" * 70)

# ── Load parquet ──
print("\nLoading data...")
df_raw = pd.read_parquet("ncaa_training_data.parquet")
df_raw = df_raw[df_raw["actual_home_score"].notna()].copy()
df_raw = df_raw[df_raw["season"] != 2021].copy()  # COVID

# ESPN odds fallback
if "espn_spread" in df_raw.columns:
    espn_s = pd.to_numeric(df_raw["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df_raw.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df_raw.loc[fill, "market_spread_home"] = espn_s[fill]

# Column fixes
for col in ["actual_home_score", "actual_away_score", "home_adj_em", "away_adj_em",
            "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
            "home_tempo", "away_tempo", "home_rank", "away_rank", "season",
            "home_record_wins", "away_record_wins", "home_record_losses", "away_record_losses"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

if "home_record_wins" in df_raw.columns and "home_wins" not in df_raw.columns:
    df_raw["home_wins"] = df_raw["home_record_wins"]
if "away_record_wins" in df_raw.columns and "away_wins" not in df_raw.columns:
    df_raw["away_wins"] = df_raw["away_record_wins"]
if "home_record_losses" in df_raw.columns and "home_losses" not in df_raw.columns:
    df_raw["home_losses"] = df_raw["home_record_losses"]
if "away_record_losses" in df_raw.columns and "away_losses" not in df_raw.columns:
    df_raw["away_losses"] = df_raw["away_record_losses"]

seasons = sorted(df_raw["season"].unique())
print(f"  {len(df_raw)} games across seasons: {seasons}")

# Load referee profiles once
try:
    import json as _json
    with open("referee_profiles.json") as _rf:
        ncaa_build_features._ref_profiles = _json.load(_rf)
    print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("  referee_profiles.json not found — ref features zero")

# ── Quality filter function (matches retrain_and_upload.py) ──
def quality_filter(df):
    _quality_cols = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg",
                     "market_spread_home", "market_ou_total"]
    _qcols = [c for c in _quality_cols if c in df.columns]
    _qmat = pd.DataFrame({
        c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em", "away_adj_em",
                                                   "market_spread_home", "market_ou_total"] else True)
        for c in _qcols
    })
    _row_q = _qmat.mean(axis=1)
    _keep = _row_q >= 0.8
    if "referee_1" in df.columns:
        _has_ref = df["referee_1"].notna() & (df["referee_1"] != "")
        _keep = _keep & _has_ref
    return _keep


def train_and_predict(train_df, test_df, current_year):
    """Train model on train_df, predict test_df. Returns predictions array."""

    # Quality filter on training data only
    keep = quality_filter(train_df)
    train_filtered = train_df[keep].reset_index(drop=True)

    if len(train_filtered) < 1000:
        print(f"    Skipping — only {len(train_filtered)} quality games")
        return None

    # Season weights (relative to the test year)
    train_filtered["season_weight"] = train_filtered["season"].apply(
        lambda s: 1.0 if (current_year - s) <= 1 else 0.9 if (current_year - s) == 2 else
        0.75 if (current_year - s) == 3 else 0.6 if (current_year - s) == 4 else 0.5)

    # Heuristic backfill
    train_filtered = _ncaa_backfill_heuristic(train_filtered)
    test_processed = _ncaa_backfill_heuristic(test_df.copy())

    # Build features
    X_train = ncaa_build_features(train_filtered)
    X_test = ncaa_build_features(test_processed)

    feature_cols = list(X_train.columns)

    # Ensure test has same columns
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]

    y_margin = train_filtered["actual_home_score"].values - train_filtered["actual_away_score"].values
    y_win = (y_margin > 0).astype(int)
    weights = train_filtered["season_weight"].values

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train regressor
    cat_r = CatBoostRegressor(n_estimators=125, depth=4, learning_rate=0.10,
                               random_seed=42, verbose=0)
    cat_r.fit(X_train_s, y_margin, sample_weight=weights)

    # Train classifier
    cat_c = CatBoostClassifier(n_estimators=125, depth=4, learning_rate=0.10,
                                random_seed=42, verbose=0)
    cat_c.fit(X_train_s, y_win, sample_weight=weights)

    # Get OOF probabilities for isotonic calibration (5-fold time series on training data)
    n_train = len(X_train_s)
    oof_probs = np.zeros(n_train)
    fold_size = n_train // 6
    for i in range(5):
        t_end = fold_size * (i + 2)
        v_start = t_end
        v_end = min(t_end + fold_size, n_train)
        if v_start >= n_train:
            break
        cc = copy.deepcopy(cat_c)
        cc.fit(X_train_s[:t_end], y_win[:t_end], sample_weight=weights[:t_end])
        oof_probs[v_start:v_end] = cc.predict_proba(X_train_s[v_start:v_end])[:, 1]

    valid_mask = oof_probs != 0
    isotonic = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    isotonic.fit(oof_probs[valid_mask], y_win[valid_mask])

    # Predict test set
    pred_margins = cat_r.predict(X_test_s)
    raw_probs = cat_c.predict_proba(X_test_s)[:, 1]
    cal_probs = np.clip(isotonic.predict(raw_probs), 0.02, 0.98)

    return {
        "pred_margin": pred_margins,
        "raw_prob": raw_probs,
        "cal_prob": cal_probs,
        "n_train": len(train_filtered),
        "n_features": len(feature_cols),
    }


# ── Walk forward: predict each season using only prior seasons ──
# Need at least 3 seasons of training data, so start predicting from season 4+
MIN_TRAIN_SEASONS = 3
all_results = []

for i, test_season in enumerate(seasons):
    prior_seasons = [s for s in seasons if s < test_season]
    if len(prior_seasons) < MIN_TRAIN_SEASONS:
        print(f"\n  Season {test_season}: skipping (only {len(prior_seasons)} prior seasons)")
        continue

    train_df = df_raw[df_raw["season"].isin(prior_seasons)].copy()
    test_df = df_raw[df_raw["season"] == test_season].copy().reset_index(drop=True)

    print(f"\n  Season {test_season}: train on {prior_seasons} ({len(train_df)} games) → test {len(test_df)} games")
    t0 = time.time()

    preds = train_and_predict(train_df, test_df, test_season)
    if preds is None:
        continue

    elapsed = time.time() - t0
    print(f"    Trained on {preds['n_train']} games, {preds['n_features']} features in {elapsed:.0f}s")

    # Grade
    test_df["pred_margin"] = preds["pred_margin"]
    test_df["raw_prob"] = preds["raw_prob"]
    test_df["cal_prob"] = preds["cal_prob"]
    test_df["actual_margin"] = test_df["actual_home_score"] - test_df["actual_away_score"]
    test_df["home_win"] = (test_df["actual_margin"] > 0).astype(int)
    test_df["pred_home"] = (test_df["cal_prob"] >= 0.5).astype(int)
    test_df["ml_correct"] = test_df["pred_home"] == test_df["home_win"]

    # ATS
    mkt = pd.to_numeric(test_df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce").fillna(0)
    has_mkt = mkt.abs() > 0.1
    test_df["has_market"] = has_mkt
    test_df["model_covers"] = np.nan

    if has_mkt.sum() > 0:
        actual_ats = test_df.loc[has_mkt, "actual_margin"] + mkt[has_mkt]
        model_pick_home_cover = test_df.loc[has_mkt, "pred_margin"] > -mkt[has_mkt]
        actual_home_cover = actual_ats > 0
        non_push = actual_ats != 0
        test_df.loc[has_mkt & non_push, "model_covers"] = (
            model_pick_home_cover[non_push] == actual_home_cover[non_push]
        ).astype(float).values

    acc = test_df["ml_correct"].mean()
    ats_games = test_df["model_covers"].dropna()
    ats = ats_games.mean() if len(ats_games) > 0 else float("nan")
    print(f"    Accuracy: {acc:.1%} | ATS: {ats:.1%} ({len(ats_games)} games)")

    all_results.append(test_df)

# ── Combine all results ──
results = pd.concat(all_results, ignore_index=True)

print(f"\n{'=' * 70}")
print(f"  RESULTS — TRUE WALK-FORWARD (out-of-sample)")
print(f"{'=' * 70}")

total = len(results)
accuracy = results["ml_correct"].mean()
print(f"\nTotal games predicted: {total:,}")
print(f"Overall accuracy: {accuracy:.1%}")

# By season
print(f"\nBy season:")
print(f"{'Season':<10} {'Games':>8} {'Train':>8} {'Accuracy':>10} {'ATS':>8} {'ATS N':>8}")
print("-" * 58)
for season in sorted(results["season"].unique()):
    s = results[results["season"] == season]
    acc = s["ml_correct"].mean()
    ats_g = s["model_covers"].dropna()
    ats = ats_g.mean() if len(ats_g) > 0 else float("nan")
    # Estimate training size from prior seasons
    train_n = len(df_raw[df_raw["season"] < season])
    print(f"{int(season):<10} {len(s):>8,} {train_n:>8,} {acc:>10.1%} {ats:>8.1%} {len(ats_g):>8,}")

# Calibration
print(f"\nCalibration:")
print(f"{'Prob Range':<15} {'Games':>8} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
print("-" * 55)
bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00)]
for lo, hi in bins:
    fav_prob = np.maximum(results["cal_prob"].values, 1 - results["cal_prob"].values)
    mask = (fav_prob >= lo) & (fav_prob < hi)
    if mask.sum() == 0:
        continue
    pred_avg = fav_prob[mask].mean()
    home_fav = results["cal_prob"].values[mask] >= 0.5
    outcomes = results["home_win"].values[mask]
    fav_won = (home_fav & (outcomes == 1)) | (~home_fav & (outcomes == 0))
    actual_avg = fav_won.mean()
    gap = actual_avg - pred_avg
    print(f"{int(lo*100)}%–{int(hi*100)}%{mask.sum():>10,} {pred_avg:>10.1%} {actual_avg:>10.1%} {gap:>+8.1%}")

# ATS by probability threshold
print(f"\nATS by probability threshold (true out-of-sample):")
print(f"{'Min Prob':<12} {'Games':>8} {'ATS':>8} {'vs Break-even':>14}")
print("-" * 45)
for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    fav_prob = np.maximum(results["cal_prob"].values, 1 - results["cal_prob"].values)
    fav_mask = fav_prob >= thresh
    ats_sub = results.loc[fav_mask, "model_covers"].dropna()
    if len(ats_sub) > 0:
        ats_pct = ats_sub.mean()
        edge = ats_pct - 0.524
        print(f"{int(thresh*100)}%+{len(ats_sub):>10,} {ats_pct:>8.1%} {edge:>+14.1%}")

# Brier
brier = np.mean((results["cal_prob"].values - results["home_win"].values.astype(float)) ** 2)
print(f"\nBrier score: {brier:.4f}")

# Save
results.to_csv("true_walkforward_results.csv", index=False)
summary = {
    "type": "true_walk_forward",
    "total_games": total,
    "accuracy": round(accuracy, 4),
    "brier": round(brier, 4),
    "seasons_predicted": sorted([int(s) for s in results["season"].unique()]),
    "min_train_seasons": MIN_TRAIN_SEASONS,
}
with open("true_walkforward_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to true_walkforward_results.csv")
print("Done!")
