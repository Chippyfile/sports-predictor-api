"""
Recalibrate Model Probabilities
================================
Uses the walk-forward results (all features, including market)
to fit a new isotonic calibration curve that makes probabilities honest.

After this: when the model says 75%, teams actually win ~75% of the time.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  RECALIBRATE MODEL PROBABILITIES")
print("=" * 60)

# ── Load model ──
print("\nLoading model...")
from ml_utils import StackedRegressor, StackedClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

MODEL_PATH = "ncaa_model_local.pkl"
bundle = joblib.load(MODEL_PATH)
feature_cols = bundle["feature_cols"]
old_isotonic = bundle.get("isotonic")
print(f"  Model: {len(feature_cols)} features, MAE {bundle.get('mae_cv', '?')}")
print(f"  Existing isotonic: {'yes' if old_isotonic else 'no'}")

# ── Load data ──
print("\nLoading parquet...")
df = pd.read_parquet("ncaa_training_data.parquet")
if "season" in df.columns:
    df = df[df["season"] != 2021]
print(f"  {len(df)} games (excl COVID 2021)")

# ── Build features ──
print("\nBuilding features...")
from sports.ncaa import ncaa_build_features
X_full = ncaa_build_features(df)
for col in feature_cols:
    if col not in X_full.columns:
        X_full[col] = 0
X = X_full[feature_cols].copy()

# ── Get raw classifier probabilities ──
print("Getting raw probabilities...")
clf = bundle["clf"]
scaler = bundle["scaler"]
X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
raw_probs = clf.predict_proba(X_scaled)[:, 1]
outcomes = df["home_win"].astype(int).values

print(f"  Raw prob range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
print(f"  Home win rate: {outcomes.mean():.3f}")

# ── Show BEFORE calibration ──
if old_isotonic:
    old_cal = np.clip(old_isotonic.predict(raw_probs), 0.02, 0.98)
else:
    old_cal = raw_probs

brier_raw = np.mean((raw_probs - outcomes) ** 2)
brier_old = np.mean((old_cal - outcomes) ** 2)

print(f"\n  Brier (raw classifier): {brier_raw:.4f}")
print(f"  Brier (old isotonic):   {brier_old:.4f}")

# ── Split: 80% train calibration, 20% validate ──
print("\nSplitting 80/20 for calibration...")
np.random.seed(42)
n = len(df)
idx = np.random.permutation(n)
train_n = int(0.8 * n)
train_idx, val_idx = idx[:train_n], idx[train_n:]

train_raw = raw_probs[train_idx]
train_y = outcomes[train_idx]
val_raw = raw_probs[val_idx]
val_y = outcomes[val_idx]

# ── Fit new isotonic on train set ──
print("Fitting new isotonic regression...")
new_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
new_iso.fit(train_raw, train_y)

# ── Evaluate on validation set ──
val_old_cal = np.clip(old_isotonic.predict(val_raw), 0.02, 0.98) if old_isotonic else val_raw
val_new_cal = np.clip(new_iso.predict(val_raw), 0.02, 0.98)

brier_val_old = np.mean((val_old_cal - val_y) ** 2)
brier_val_new = np.mean((val_new_cal - val_y) ** 2)

print(f"\n  Validation Brier (old): {brier_val_old:.4f}")
print(f"  Validation Brier (new): {brier_val_new:.4f} ({brier_val_new - brier_val_old:+.4f})")

# ── Calibration table: BEFORE vs AFTER ──
bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00)]

def cal_table(probs, outcomes, label):
    """Compute calibration from the favored team's perspective."""
    print(f"\n{label}:")
    print(f"{'Prob Range':<15} {'Games':>8} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
    print("-" * 55)
    for lo, hi in bins:
        # Favored team's probability
        fav_prob = np.maximum(probs, 1 - probs)
        mask = (fav_prob >= lo) & (fav_prob < hi)
        if mask.sum() == 0:
            continue
        sub_prob = fav_prob[mask]
        # Favored team won = (prob >= 0.5 and home won) or (prob < 0.5 and away won)
        home_fav = probs[mask] >= 0.5
        fav_won = (home_fav & (outcomes[mask] == 1)) | (~home_fav & (outcomes[mask] == 0))
        pred_avg = sub_prob.mean()
        actual_avg = fav_won.mean()
        gap = actual_avg - pred_avg
        print(f"{int(lo*100):>3}%–{int(hi*100)}%{mask.sum():>10,} {pred_avg:>10.1%} {actual_avg:>10.1%} {gap:>+8.1%}")

print(f"\n{'=' * 60}")
print(f"  CALIBRATION COMPARISON (validation set, n={len(val_idx):,})")
print(f"{'=' * 60}")

cal_table(val_old_cal, val_y, "OLD isotonic calibration")
cal_table(val_new_cal, val_y, "NEW isotonic calibration")

# ── Fit FINAL isotonic on ALL data for production ──
print(f"\n{'=' * 60}")
print(f"  FITTING PRODUCTION CALIBRATION ON ALL {len(df):,} GAMES")
print(f"{'=' * 60}")

final_iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
final_iso.fit(raw_probs, outcomes)

# Show final calibration on full data
final_cal = np.clip(final_iso.predict(raw_probs), 0.02, 0.98)
brier_final = np.mean((final_cal - outcomes) ** 2)
print(f"\n  Final Brier (all data): {brier_final:.4f}")
cal_table(final_cal, outcomes, "PRODUCTION calibration (all data)")

# ── Accuracy and ATS with new calibration ──
print(f"\n{'=' * 60}")
print(f"  ACCURACY & ATS WITH NEW CALIBRATION")
print(f"{'=' * 60}")

pred_home = final_cal >= 0.5
actual_home = outcomes == 1
accuracy = (pred_home == actual_home).mean()
print(f"\nOverall accuracy: {accuracy:.1%}")

# ATS
if "mkt_spread" in X_full.columns:
    mkt = X_full["mkt_spread"].values
elif "market_spread_home" in df.columns:
    mkt = pd.to_numeric(df["market_spread_home"], errors="coerce").fillna(0).values
else:
    mkt = np.zeros(len(df))

has_mkt = np.abs(mkt) > 0.1
margins = bundle["reg"].predict(X_scaled)

print(f"\nATS by recalibrated probability threshold:")
print(f"{'Min Prob':<12} {'Games':>8} {'ATS':>8} {'Edge':>8}")
print("-" * 40)
for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    fav_prob = np.maximum(final_cal, 1 - final_cal)
    fav_mask = fav_prob >= thresh
    combined = fav_mask & has_mkt
    if combined.sum() == 0:
        continue
    actual_margin = df["actual_margin"].values[combined]
    game_mkt = mkt[combined]
    model_margin = margins[combined]
    # Model pick: if model_margin > -mkt_spread, model favors home to cover
    model_home_cover = model_margin > -game_mkt
    actual_ats = actual_margin + game_mkt
    actual_home_cover = actual_ats > 0
    # Exclude pushes
    non_push = actual_ats != 0
    if non_push.sum() == 0:
        continue
    ats_correct = (model_home_cover[non_push] == actual_home_cover[non_push]).mean()
    edge = ats_correct - 0.524  # breakeven at -110
    print(f"{int(thresh*100)}%+{non_push.sum():>10,} {ats_correct:>8.1%} {edge:>+8.1%}")

# ── Update model bundle ──
print(f"\n{'=' * 60}")
print(f"  UPDATING MODEL")
print(f"{'=' * 60}")

bundle["isotonic"] = final_iso
bundle["isotonic_n_games"] = len(df)
bundle["isotonic_brier"] = round(brier_final, 4)

joblib.dump(bundle, MODEL_PATH, compress=3)
print(f"\n  Updated {MODEL_PATH} with new isotonic calibration")
print(f"  Brier: {brier_old:.4f} → {brier_final:.4f}")

# Also save to models/ncaa.pkl for Railway
import shutil
shutil.copy(MODEL_PATH, "models/ncaa.pkl")
print(f"  Copied to models/ncaa.pkl")

# Save calibration curve as JSON for frontend reference
cal_points = []
for p in np.arange(0.02, 0.99, 0.01):
    cal_points.append({
        "raw": round(p, 2),
        "calibrated": round(float(final_iso.predict([p])[0]), 4)
    })

with open("calibration_curve.json", "w") as f:
    json.dump({
        "description": "Isotonic calibration on 60K walk-forward games",
        "n_games": len(df),
        "brier_before": round(brier_old, 4),
        "brier_after": round(brier_final, 4),
        "curve": cal_points,
    }, f, indent=2)
print(f"  Calibration curve saved to calibration_curve.json")

print(f"\nDeploy to Railway:")
print(f"  cd ~/Desktop/sports-predictor-api")
print(f"  git add models/ncaa.pkl && git commit -m 'model: recalibrated isotonic on 60K games' && git push")
print(f"\nDone!")
