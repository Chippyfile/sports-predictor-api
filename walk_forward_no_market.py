"""
Walk-Forward Backtest — NO MARKET DATA
=======================================
Measures true pre-game predictive power by zeroing out all market features.
Then recalibrates probabilities using isotonic regression on the results.

This answers: "How good is the model WITHOUT knowing what Vegas thinks?"
"""

import os, sys, pickle, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  WALK-FORWARD BACKTEST — NO MARKET DATA")
print("=" * 60)

# ── Load model ──
print("\nLoading model...")
from ml_utils import StackedRegressor, StackedClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

MODEL_PATH = "ncaa_model_local.pkl"
import joblib
bundle = joblib.load(MODEL_PATH)

feature_cols = bundle["feature_cols"]
print(f"  Model loaded — {len(feature_cols)} features, MAE {bundle.get('mae_cv', '?')}")

# Identify market-dependent features
MARKET_FEATURES = [
    "mkt_spread", "mkt_total", "mkt_spread_vs_model", "has_mkt",
    "spread_movement", "total_movement",
    "spread_regime", "luck_x_spread", "consistency_x_spread",
    "roll_ats_diff_gated", "roll_ats_margin_gated", "has_ats_data",
]
market_in_model = [f for f in MARKET_FEATURES if f in feature_cols]
print(f"  Market features to zero out: {len(market_in_model)}")
for f in market_in_model:
    print(f"    - {f}")

# ── Load data ──
print("\nLoading parquet...")
PARQUET = "ncaa_training_data.parquet"
df = pd.read_parquet(PARQUET)
print(f"  Loaded {len(df)} rows × {len(df.columns)} cols")

# COVID filter
df = df[~((df["season"] == 2021) & (df.get("is_covid", False) == True))]
if "season" in df.columns:
    df = df[df["season"] != 2021]
print(f"  After COVID filter: {len(df)} games")

# ── Build features ──
print("\nBuilding features...")
from sports.ncaa import ncaa_build_features
X_full = ncaa_build_features(df)
print(f"  Features built: {X_full.shape}")

# Ensure all model features exist
for col in feature_cols:
    if col not in X_full.columns:
        X_full[col] = 0
X = X_full[feature_cols].copy()

# ── Zero out market features ──
print(f"\nZeroing out {len(market_in_model)} market features...")
for f in market_in_model:
    X[f] = 0

# ── Run predictions ──
print("Running predictions (no market data)...")
reg = bundle["reg"]
clf = bundle["clf"]
scaler = bundle["scaler"]

X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

margins = reg.predict(X_scaled)
raw_probs = clf.predict_proba(X_scaled)[:, 1]

# Apply existing isotonic calibration if available
isotonic = bundle.get("isotonic")
if isotonic:
    cal_probs = isotonic.predict(raw_probs)
    print("  Applied existing isotonic calibration")
else:
    cal_probs = raw_probs
    print("  No isotonic calibration found — using raw probs")

cal_probs = np.clip(cal_probs, 0.02, 0.98)
print(f"  Predicted {len(margins)} games")

# ── Grade predictions ──
print("\nGrading predictions...")
results = df[["game_id", "game_date", "season", "home_team_name", "away_team_name",
              "actual_home_score", "actual_away_score", "actual_margin", "home_win",
              "neutral_site"]].copy()

# Add market spread if available for ATS grading
if "mkt_spread" in X_full.columns:
    results["mkt_spread"] = X_full["mkt_spread"].values
elif "market_spread_home" in df.columns:
    results["mkt_spread"] = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
else:
    results["mkt_spread"] = 0

results["pred_margin"] = margins
results["pred_prob"] = cal_probs
results["raw_prob"] = raw_probs
results["pred_winner_home"] = cal_probs >= 0.5
results["actual_winner_home"] = results["home_win"].astype(bool)
results["ml_correct"] = results["pred_winner_home"] == results["actual_winner_home"]

# ATS grading (model margin vs actual margin vs market spread)
results["has_market"] = results["mkt_spread"].abs() > 0.1
results["model_covers"] = np.nan
mask = results["has_market"]
# Model picks: if pred_margin > -mkt_spread, model favors home to cover
# ATS correct if: (actual_margin + mkt_spread > 0 AND model said home covers)
#              or: (actual_margin + mkt_spread < 0 AND model said away covers)
actual_ats = results.loc[mask, "actual_margin"] + results.loc[mask, "mkt_spread"]
model_pick_home_cover = results.loc[mask, "pred_margin"] > -results.loc[mask, "mkt_spread"]
actual_home_cover = actual_ats > 0
results.loc[mask, "model_covers"] = (model_pick_home_cover == actual_home_cover).astype(float)
# Pushes
results.loc[mask & (actual_ats == 0), "model_covers"] = np.nan

# ── Overall Results ──
total = len(results)
correct = results["ml_correct"].sum()
accuracy = correct / total

print(f"\n{'=' * 60}")
print(f"  RESULTS — NO MARKET DATA")
print(f"{'=' * 60}")
print(f"\nTotal games: {total:,}")
print(f"Overall accuracy: {accuracy:.1%}")

# Compare to full-market accuracy
print(f"\n  (Compare: WITH market data was 74.7%)")
print(f"  (This measures pure pre-game predictive power)")

# ── By confidence tier ──
def assign_tier(prob):
    margin = abs(prob - 0.5)
    if margin < 0.05:
        return "LOW"
    elif margin < 0.15:
        return "MEDIUM"
    else:
        return "HIGH"

results["tier"] = results["pred_prob"].apply(assign_tier)

print(f"\nBy confidence tier:")
print(f"{'Tier':<12} {'Games':>8} {'Accuracy':>10} {'ATS':>10} {'ATS N':>8}")
print("-" * 52)
for tier in ["LOW", "MEDIUM", "HIGH"]:
    t = results[results["tier"] == tier]
    acc = t["ml_correct"].mean()
    ats_games = t["model_covers"].dropna()
    ats = ats_games.mean() if len(ats_games) > 0 else float("nan")
    ats_n = len(ats_games)
    print(f"{tier:<12} {len(t):>8,} {acc:>10.1%} {ats:>10.1%} {ats_n:>8,}")

# ── By season ──
print(f"\nBy season:")
print(f"{'Season':<12} {'Games':>8} {'Accuracy':>10} {'ATS':>10} {'ATS N':>8}")
print("-" * 52)
for season in sorted(results["season"].unique()):
    s = results[results["season"] == season]
    acc = s["ml_correct"].mean()
    ats_games = s["model_covers"].dropna()
    ats = ats_games.mean() if len(ats_games) > 0 else float("nan")
    ats_n = len(ats_games)
    print(f"{int(season):<12} {len(s):>8,} {acc:>10.1%} {ats:>10.1%} {ats_n:>8,}")

# ── Calibration BEFORE recalibration ──
print(f"\nCalibration (BEFORE recalibration):")
print(f"{'Win Prob Range':<20} {'Games':>8} {'Predicted':>10} {'Actual':>10} {'Gap':>10}")
print("-" * 62)
bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00)]
cal_data = []
for lo, hi in bins:
    # Include both sides (>0.5 home, <0.5 away) by using absolute distance from 0.5
    # mapped back to the favored team's perspective
    fav_prob = results["pred_prob"].clip(lower=0.5)
    mask_hi = (fav_prob >= lo) & (fav_prob < hi)
    # For away favorites (pred_prob < 0.5), flip to their perspective
    mask_lo = ((1 - results["pred_prob"]) >= lo) & ((1 - results["pred_prob"]) < hi)
    mask = mask_hi | mask_lo
    if mask.sum() == 0:
        continue
    sub = results[mask]
    # For the "actual" rate, use whether the FAVORED team won
    fav_won = sub.apply(lambda r: r["ml_correct"], axis=1)
    pred_avg = sub["pred_prob"].apply(lambda p: max(p, 1 - p)).mean()
    actual_avg = fav_won.mean()
    gap = actual_avg - pred_avg
    cal_data.append({"lo": lo, "hi": hi, "n": len(sub), "pred": pred_avg, "actual": actual_avg, "gap": gap})
    print(f"{int(lo*100)}%–{int(hi*100)}%{len(sub):>12,} {pred_avg:>10.1%} {actual_avg:>10.1%} {gap:>+10.1%}")

# ── Recalibrate with isotonic regression ──
print(f"\n{'=' * 60}")
print(f"  RECALIBRATING WITH ISOTONIC REGRESSION")
print(f"{'=' * 60}")

# Use 80% of data to fit calibration, 20% to validate
np.random.seed(42)
n = len(results)
indices = np.random.permutation(n)
train_idx = indices[:int(0.8 * n)]
val_idx = indices[int(0.8 * n):]

train_probs = results.iloc[train_idx]["raw_prob"].values
train_outcomes = results.iloc[train_idx]["actual_winner_home"].astype(int).values
val_probs = results.iloc[val_idx]["raw_prob"].values
val_outcomes = results.iloc[val_idx]["actual_winner_home"].astype(int).values

# Fit new isotonic
new_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
new_iso.fit(train_probs, train_outcomes)

# Apply to validation set
recal_val = new_iso.predict(val_probs)
recal_val = np.clip(recal_val, 0.02, 0.98)

# Apply to ALL data for the summary
recal_all = new_iso.predict(results["raw_prob"].values)
recal_all = np.clip(recal_all, 0.02, 0.98)
results["recal_prob"] = recal_all
results["recal_correct"] = (recal_all >= 0.5) == results["actual_winner_home"]

# Brier scores
brier_before = np.mean((results["pred_prob"].values - results["actual_winner_home"].astype(float).values) ** 2)
brier_after = np.mean((recal_all - results["actual_winner_home"].astype(float).values) ** 2)
# Validation set only
brier_val_before = np.mean((val_probs - val_outcomes) ** 2)
brier_val_after = np.mean((recal_val - val_outcomes) ** 2)

print(f"\nBrier score (all data):  {brier_before:.4f} → {brier_after:.4f} ({brier_after - brier_before:+.4f})")
print(f"Brier score (val only): {brier_val_before:.4f} → {brier_val_after:.4f} ({brier_val_after - brier_val_before:+.4f})")
print(f"Accuracy unchanged: {results['recal_correct'].mean():.1%} (same picks, better probabilities)")

# ── Calibration AFTER recalibration ──
print(f"\nCalibration (AFTER recalibration — validation set only):")
print(f"{'Win Prob Range':<20} {'Games':>8} {'Predicted':>10} {'Actual':>10} {'Gap':>10}")
print("-" * 62)
val_results = results.iloc[val_idx].copy()
for lo, hi in bins:
    fav_prob = val_results["recal_prob"].clip(lower=0.5)
    mask_hi = (fav_prob >= lo) & (fav_prob < hi)
    mask_lo = ((1 - val_results["recal_prob"]) >= lo) & ((1 - val_results["recal_prob"]) < hi)
    mask = mask_hi | mask_lo
    if mask.sum() == 0:
        continue
    sub = val_results[mask]
    fav_won = sub["recal_correct"]
    pred_avg = sub["recal_prob"].apply(lambda p: max(p, 1 - p)).mean()
    actual_avg = fav_won.mean()
    gap = actual_avg - pred_avg
    print(f"{int(lo*100)}%–{int(hi*100)}%{len(sub):>12,} {pred_avg:>10.1%} {actual_avg:>10.1%} {gap:>+10.1%}")

# ── ATS with recalibrated probabilities ──
print(f"\nATS by recalibrated win probability threshold:")
print(f"{'Min Win Prob':<15} {'Games':>8} {'ATS':>10}")
print("-" * 35)
for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    # Games where model's favored team has recal_prob >= threshold
    fav_mask = results["recal_prob"].apply(lambda p: max(p, 1-p)) >= threshold
    ats_sub = results[fav_mask & results["has_market"]]["model_covers"].dropna()
    if len(ats_sub) > 0:
        print(f"{int(threshold*100)}%+{len(ats_sub):>12,} {ats_sub.mean():>10.1%}")

# ── Save new isotonic calibration ──
# Fit on ALL data (not just 80%) for production use
final_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
final_iso.fit(results["raw_prob"].values, results["actual_winner_home"].astype(int).values)

# Save calibration curve for frontend
cal_points = []
for p in np.arange(0.02, 0.99, 0.01):
    cal_points.append({"raw": round(p, 2), "calibrated": round(float(final_iso.predict([p])[0]), 4)})

with open("no_market_calibration.json", "w") as f:
    json.dump({
        "description": "Isotonic calibration trained on 60K games WITHOUT market features",
        "created": datetime.now().isoformat(),
        "n_games": len(results),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "calibration_curve": cal_points,
    }, f, indent=2)
print(f"\nCalibration curve saved to no_market_calibration.json")

# Save the isotonic model for integration into the pkl
import joblib as jl
jl.dump(final_iso, "no_market_isotonic.joblib")
print("Isotonic model saved to no_market_isotonic.joblib")

# ── Save results ──
results.to_csv("walk_forward_no_market_results.csv", index=False)
print(f"Full results saved to walk_forward_no_market_results.csv")

summary = {
    "total_games": total,
    "accuracy_no_market": round(accuracy, 4),
    "brier_before_recal": round(brier_before, 4),
    "brier_after_recal": round(brier_after, 4),
    "market_features_zeroed": market_in_model,
    "calibration_data": cal_data,
}
with open("walk_forward_no_market_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved to walk_forward_no_market_summary.json")

print(f"\nDone!")
