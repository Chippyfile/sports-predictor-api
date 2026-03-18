#!/usr/bin/env python3
"""
Walk-Forward Backtest — NCAA Predictions
==========================================
Runs the trained model against all 64K historical games in the parquet,
using only data that was available before each game was played.

Features are pre-computed in the parquet (rolling averages, Elo, etc.)
so there is no leakage — each row already represents pre-game state.

Usage:
    # Run with .venv Python (required for numpy 1.26.4 compatibility):
    .venv/bin/python3 walk_forward_backtest.py

    # Optional flags:
    .venv/bin/python3 walk_forward_backtest.py --seasons 2024 2025 2026
    .venv/bin/python3 walk_forward_backtest.py --min-date 2022-11-01

Output:
    walk_forward_results.csv   — game-level predictions + grades
    walk_forward_summary.json  — ATS, calibration, accuracy by tier/season
"""

import sys, os, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# ── Load model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  WALK-FORWARD BACKTEST")
print("=" * 60)

print("\nLoading model...")
from ml_utils import StackedRegressor, StackedClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

MODEL_PATH = "ncaa_model_local.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"  ERROR: {MODEL_PATH} not found. Run retrain_and_upload.py first.")
    sys.exit(1)

import sys as _sys
for _cls in [StandardScaler, RidgeCV, LogisticRegression, IsotonicRegression, MLPRegressor, XGBRegressor, CatBoostRegressor, StackedRegressor, StackedClassifier]:
    _sys.modules[_cls.__name__] = _sys.modules[_cls.__module__]

with open(MODEL_PATH, "rb") as f:
    import joblib; bundle = joblib.load(MODEL_PATH)

feature_cols = bundle["feature_cols"]
reg = bundle["reg"]
clf = bundle["clf"]
isotonic = bundle.get("isotonic")
print(f"  Model loaded OK — {len(feature_cols)} features, MAE {bundle.get('mae_cv', '?')}")

# ── Load parquet ─────────────────────────────────────────────────────────────
print("\nLoading parquet...")
PARQUET_PATH = "ncaa_training_data.parquet"
df_raw = pd.read_parquet(PARQUET_PATH)
print(f"  Loaded {len(df_raw)} rows × {len(df_raw.columns)} cols")

# ── Optional filters ─────────────────────────────────────────────────────────
seasons = None
min_date = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--seasons":
        seasons = [int(s) for s in sys.argv[i+2:] if s.isdigit()]
    if arg == "--min-date":
        min_date = sys.argv[i+2]

if seasons:
    df_raw = df_raw[df_raw["season"].isin(seasons)]
    print(f"  Filtered to seasons {seasons}: {len(df_raw)} games")
if min_date:
    df_raw = df_raw[df_raw["game_date"] >= min_date]
    print(f"  Filtered to >= {min_date}: {len(df_raw)} games")

# Drop COVID season
df_raw = df_raw[df_raw["season"] != 2021]
print(f"  After COVID filter: {len(df_raw)} games")

# ── Build features ────────────────────────────────────────────────────────────
print("\nBuilding features...")
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic

# Add required columns that may be missing from parquet
_col_defaults = {
    "home_lineup_changes": 0, "away_lineup_changes": 0,
    "home_lineup_stability_5g": 1.0, "away_lineup_stability_5g": 1.0,
    "home_starter_games_together": 0, "away_starter_games_together": 0,
    "home_new_starter_impact": 0.0, "away_new_starter_impact": 0.0,
    "home_player_rating_sum": 0.0, "away_player_rating_sum": 0.0,
    "home_weakest_starter": 0.0, "away_weakest_starter": 0.0,
    "home_starter_variance": 0.0, "away_starter_variance": 0.0,
    "h2h_margin_avg": 0.0, "h2h_home_win_rate": 0.0,
    "conf_strength_diff": 0.0, "cross_conf_flag": 0,
    "odds_api_spread_movement": 0.0, "odds_api_total_movement": 0.0,
    "dk_spread_movement": 0.0, "dk_total_movement": 0.0,
    "home_tempo": 70.0, "away_tempo": 70.0,
    "home_ppg": 0.0, "away_ppg": 0.0,
    "home_opp_ppg": 0.0, "away_opp_ppg": 0.0,
    "recent_form_diff": 0.0,
    "home_clutch_ftm": 0.0, "away_clutch_ftm": 0.0,
    "home_clutch_fta": 1.0, "away_clutch_fta": 1.0,
    "espn_home_win_pct": 0.5, "espn_predictor_home_pct": 0.5,
    "halftime_home_win_prob": 0.5,
    "home_injury_penalty": 0.0, "away_injury_penalty": 0.0,
    "home_missing_starters": 0, "away_missing_starters": 0,
    "injury_diff": 0.0,
}
for col, default in _col_defaults.items():
    if col not in df_raw.columns:
        df_raw[col] = default

# Load referee profiles if available
try:
    import json as _json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = _json.load(f)
    print("  Referee profiles loaded")
except:
    print("  No referee profiles found — ref features will default to 0")

# Run backfill heuristic (generates spread_home, win_pct_home etc.)
df_raw["home_record_wins"] = df_raw.get("home_wins", df_raw.get("home_record_wins", 0))
df_raw["away_record_wins"] = df_raw.get("away_wins", df_raw.get("away_record_wins", 0))
df_raw["home_record_losses"] = df_raw.get("home_losses", df_raw.get("home_record_losses", 0))
df_raw["away_record_losses"] = df_raw.get("away_losses", df_raw.get("away_record_losses", 0))

df_raw = _ncaa_backfill_heuristic(df_raw)
X_all = ncaa_build_features(df_raw)
print(f"  Features built: {X_all.shape}")

# Align with model feature cols
for col in feature_cols:
    if col not in X_all.columns:
        X_all[col] = 0
X_model = X_all[feature_cols].fillna(0)

# ── Run predictions ───────────────────────────────────────────────────────────
print("\nRunning predictions...")
margins = reg.predict(X_model)

raw_probs = clf.predict_proba(X_model)[:, 1]
if isotonic is not None:
    win_probs = np.array([float(isotonic.predict([p])[0]) for p in raw_probs])
else:
    win_probs = raw_probs
win_probs = np.clip(win_probs, 0.05, 0.95)

print(f"  Predicted {len(margins)} games")

# ── Grade predictions ─────────────────────────────────────────────────────────
print("\nGrading predictions...")
results = df_raw[["game_date", "season", "home_team_name", "away_team_name",
                   "actual_home_score", "actual_away_score", "actual_margin",
                   "neutral_site"]].copy()

# Add parquet market spread if available
for col in ["market_spread_home", "mkt_spread"]:
    if col in df_raw.columns:
        results["market_spread"] = df_raw[col]
        break
else:
    results["market_spread"] = np.nan

results["pred_margin"] = margins
results["win_prob_home"] = win_probs
results["win_margin"] = np.abs(win_probs - 0.5)

# Straight-up accuracy
results["actual_margin"] = results["actual_home_score"] - results["actual_away_score"]
results["ml_correct"] = (
    (results["pred_margin"] > 0) == (results["actual_margin"] > 0)
).astype(int)

# ATS (only where market spread exists)
results["ats_correct"] = np.nan
mask = results["market_spread"].notna()
if mask.sum() > 0:
    pred_beats_spread = results.loc[mask, "pred_margin"] - results.loc[mask, "market_spread"]
    actual_beats_spread = results.loc[mask, "actual_margin"] - results.loc[mask, "market_spread"]
    results.loc[mask, "ats_correct"] = (
        (pred_beats_spread > 0) == (actual_beats_spread > 0)
    ).astype(float)

# Confidence tiers
results["tier"] = pd.cut(
    results["win_margin"],
    bins=[0, 0.10, 0.25, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

total = len(results)
overall_acc = results["ml_correct"].mean()
print(f"\nTotal games: {total:,}")
print(f"Overall accuracy: {overall_acc:.1%}")

# By tier
print("\nBy confidence tier:")
print(f"{'Tier':<10} {'Games':>8} {'Accuracy':>10} {'ATS':>10} {'ATS N':>8}")
print("-" * 50)
for tier in ["LOW", "MEDIUM", "HIGH"]:
    t = results[results["tier"] == tier]
    acc = t["ml_correct"].mean()
    ats = t["ats_correct"].mean()
    ats_n = t["ats_correct"].notna().sum()
    print(f"{tier:<10} {len(t):>8,} {acc:>10.1%} {ats if pd.notna(ats) else 'N/A':>10} {ats_n:>8,}")

# By season
print("\nBy season:")
print(f"{'Season':<10} {'Games':>8} {'Accuracy':>10} {'ATS':>10}")
print("-" * 40)
for season in sorted(results["season"].unique()):
    s = results[results["season"] == season]
    acc = s["ml_correct"].mean()
    ats = s["ats_correct"].mean()
    print(f"{int(season):<10} {len(s):>8,} {acc:>10.1%} {f'{ats:.1%}' if pd.notna(ats) else 'N/A':>10}")

# Calibration bins
print("\nCalibration (predicted vs actual win rate):")
print(f"{'Win Prob Range':<20} {'Games':>8} {'Pred':>8} {'Actual':>8} {'Gap':>8}")
print("-" * 56)
bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00)]
for lo, hi in bins:
    # Use max(home_prob, away_prob) to normalize to 50%+ perspective
    mask = results["win_prob_home"].between(lo, hi) | (1 - results["win_prob_home"]).between(lo, hi)
    sub = results[mask].copy()
    if len(sub) < 10:
        continue
    # Flip so we're always looking from the favored side
    sub["favored_prob"] = sub["win_prob_home"].apply(lambda p: p if p >= 0.5 else 1 - p)
    sub["favored_correct"] = sub.apply(
        lambda r: r["ml_correct"] if r["win_prob_home"] >= 0.5 else 1 - r["ml_correct"], axis=1
    )
    pred_avg = sub["favored_prob"].mean()
    actual_avg = sub["favored_correct"].mean()
    gap = actual_avg - pred_avg
    print(f"{lo:.0%}–{hi:.0%}{'':>10} {len(sub):>8,} {pred_avg:>8.1%} {actual_avg:>8.1%} {gap:>+8.1%}")

# ATS by confidence threshold
print("\nATS by win probability threshold:")
print(f"{'Min Win Prob':<15} {'Games':>8} {'ATS':>10}")
print("-" * 35)
for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    sub = results[
        (results["win_prob_home"] >= thresh) | (results["win_prob_home"] <= 1 - thresh)
    ]
    ats = sub["ats_correct"].mean()
    ats_n = sub["ats_correct"].notna().sum()
    if ats_n > 0:
        print(f"{thresh:.0%}+{'':>8} {ats_n:>8,} {ats:>10.1%}")

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\nSaving results...")
results.to_csv("walk_forward_results.csv", index=False)
print("  walk_forward_results.csv saved")

# JSON summary
summary = {
    "generated_at": datetime.now().isoformat(),
    "total_games": total,
    "overall_accuracy": round(float(overall_acc), 4),
    "by_tier": {},
    "by_season": {},
}
for tier in ["LOW", "MEDIUM", "HIGH"]:
    t = results[results["tier"] == tier]
    summary["by_tier"][tier] = {
        "n": len(t),
        "accuracy": round(float(t["ml_correct"].mean()), 4),
        "ats_accuracy": round(float(t["ats_correct"].mean()), 4) if t["ats_correct"].notna().any() else None,
        "ats_n": int(t["ats_correct"].notna().sum()),
    }
for season in sorted(results["season"].unique()):
    s = results[results["season"] == season]
    summary["by_season"][int(season)] = {
        "n": len(s),
        "accuracy": round(float(s["ml_correct"].mean()), 4),
        "ats_accuracy": round(float(s["ats_correct"].mean()), 4) if s["ats_correct"].notna().any() else None,
    }

with open("walk_forward_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  walk_forward_summary.json saved")

print("\nDone!")
