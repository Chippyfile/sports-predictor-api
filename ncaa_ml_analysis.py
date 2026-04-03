#!/usr/bin/env python3
"""
ncaa_ml_analysis.py — Moneyline (straight-up winner) accuracy analysis
======================================================================
Uses walk-forward predictions to evaluate:
  1. Overall ML accuracy by predicted margin
  2. Accuracy by probability bucket (calibration)
  3. Home vs Away breakdown at each confidence level
  4. Per-season consistency
  5. ROI simulation at various confidence thresholds

Usage:
    python3 ncaa_ml_analysis.py
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from datetime import datetime

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42
N_FOLDS = 30

FEATURES_43 = [
    "mkt_spread", "player_rating_diff", "ref_home_whistle", "weakest_starter_diff",
    "crowd_shock_diff", "lineup_stability_diff", "lineup_changes_diff", "adj_oe_diff",
    "hca_pts", "blowout_asym_diff", "threepct_diff", "pit_sos_diff",
    "orb_pct_diff", "blocks_diff", "drb_pct_diff", "opp_to_rate_diff",
    "elo_diff", "is_early", "spread_regime", "assist_rate_diff",
    "opp_ppg_diff", "opp_suppression_diff", "roll_ats_margin_gated", "has_ats_data",
    "tempo_avg", "form_x_familiarity", "to_conversion_diff", "conf_strength_diff",
    "roll_rotation_diff", "roll_dominance_diff", "importance", "twopt_diff",
    "roll_ats_diff_gated", "overreaction_diff", "three_rate_diff",
    "ppp_diff", "to_margin_diff", "momentum_halflife_diff",
    "starter_experience_diff", "style_familiarity", "fatigue_x_quality",
    "ato_diff", "consistency_x_spread",
]

MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}


def walk_forward(X_s, y, n_folds):
    n = len(X_s)
    fold_size = n // (n_folds + 1)
    min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder()
            m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof


# ── Load data ──
print("=" * 70)
print("  NCAA MONEYLINE (WINNER) ACCURACY ANALYSIS")
print("=" * 70)

print("\n  Loading data...")
df = load_cached()
if df is None: df = dump()
df = df[df["actual_home_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()

df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
df = df[season_mask & early_mask].copy()
df = df.drop(columns=["game_date_dt"], errors="ignore")

if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]

_qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score","actual_away_score","season",
            "home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else
    0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

print("  Heuristic backfill + features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError: pass
try:
    import json
    with open("referee_profiles.json") as f: ncaa_build_features._ref_profiles = json.load(f)
except: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])
X_full = ncaa_build_features(df)
available = [f for f in FEATURES_43 if f in X_full.columns]
X = X_full[available]

y = df["actual_home_score"].values - df["actual_away_score"].values
seasons = df["season"].values
n = len(X)

print(f"  {n} games × {len(available)} features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ── Walk-forward ──
print(f"\n  {N_FOLDS}-fold walk-forward...")
t0 = time.time()
oof = walk_forward(X_s, y, N_FOLDS)
print(f"  Done in {time.time()-t0:.0f}s")

valid = ~np.isnan(oof)
pred = oof[valid]; actual = y[valid]; szn = seasons[valid]
pred_home_win = pred > 0
actual_home_win = actual > 0
not_tie = actual != 0
margin = np.abs(pred)

# Convert margin to probability (sigmoid, σ=10 for NCAA)
prob_home = 1.0 / (1.0 + np.exp(-pred / 10.0))
prob_pick = np.maximum(prob_home, 1 - prob_home)  # confidence in predicted winner

# ══════════════════════════════════════════════════════════
# 1. OVERALL ML ACCURACY BY PREDICTED MARGIN
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  1. ML ACCURACY BY PREDICTED MARGIN")
print(f"     (Does predicting a bigger margin = more likely to win?)")
print(f"{'='*70}")

print(f"\n  {'Margin':>8s} {'Side':>6s} {'Games':>7s} {'W':>5s} {'L':>5s} {'Acc':>6s}")
print("  " + "-" * 42)

for t in [0, 2, 4, 6, 8, 10, 12, 15, 20]:
    # Home wins when model predicts home
    h_mask = (pred >= t) & not_tie
    if h_mask.sum() >= 20:
        h_acc = actual_home_win[h_mask].mean()
        h_w = actual_home_win[h_mask].sum()
        h_l = h_mask.sum() - h_w
        print(f"  Home≥{t:<3d} {'HOME':>6s} {h_mask.sum():>7d} {h_w:>5.0f} {h_l:>5.0f} {h_acc:>5.1%}")

    # Away wins when model predicts away
    a_mask = (pred <= -t) & not_tie
    if a_mask.sum() >= 20:
        a_acc = (~actual_home_win[a_mask]).mean()
        a_w = (~actual_home_win[a_mask]).sum()
        a_l = a_mask.sum() - a_w
        print(f"  Away≥{t:<3d} {'AWAY':>6s} {a_mask.sum():>7d} {a_w:>5.0f} {a_l:>5.0f} {a_acc:>5.1%}")

# Combined
print(f"\n  {'Margin':>8s} {'Games':>7s} {'Correct':>8s} {'Acc':>6s}")
print("  " + "-" * 35)
for t in [0, 2, 4, 6, 8, 10, 12, 15, 20]:
    m = (margin >= t) & not_tie
    if m.sum() < 20: continue
    correct = (pred_home_win[m] == actual_home_win[m])
    acc = correct.mean()
    print(f"  {t:>7d}+ {m.sum():>7d} {correct.sum():>8.0f} {acc:>5.1%}")


# ══════════════════════════════════════════════════════════
# 2. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  2. WIN PROBABILITY CALIBRATION")
print(f"     (Does 70% predicted = 70% actual?)")
print(f"{'='*70}")

buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
           (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.00)]

print(f"\n  {'Predicted':>10s} {'Games':>7s} {'Actual':>8s} {'Error':>7s} {'Quality'}")
print("  " + "-" * 50)

for lo, hi in buckets:
    # Games where model confidence falls in this bucket
    m = (prob_pick >= lo) & (prob_pick < hi) & not_tie
    if m.sum() < 20: continue
    # Did the predicted winner actually win?
    correct = (pred_home_win[m] == actual_home_win[m])
    actual_rate = correct.mean()
    expected = (lo + hi) / 2
    error = actual_rate - expected
    quality = "GOOD" if abs(error) < 0.03 else ("over" if error > 0 else "under")
    print(f"  {lo:.0%}-{hi:.0%} {m.sum():>7d} {actual_rate:>7.1%} {error:>+6.3f}  {quality}")


# ══════════════════════════════════════════════════════════
# 3. HOME vs AWAY AT CONFIDENCE THRESHOLDS
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  3. ML ACCURACY BY CONFIDENCE THRESHOLD (Home vs Away)")
print(f"{'='*70}")

print(f"\n  {'Conf':>6s} {'Side':>6s} {'Games':>7s} {'W':>5s} {'L':>5s} {'Acc':>6s}")
print("  " + "-" * 42)

for conf_thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    # Home picks at this confidence
    h_mask = (prob_home >= conf_thresh) & not_tie
    if h_mask.sum() >= 20:
        h_acc = actual_home_win[h_mask].mean()
        h_w = actual_home_win[h_mask].sum()
        print(f"  ≥{conf_thresh:.0%}  {'HOME':>6s} {h_mask.sum():>7d} {h_w:>5.0f} {h_mask.sum()-h_w:>5.0f} {h_acc:>5.1%}")

    # Away picks at this confidence
    a_mask = ((1 - prob_home) >= conf_thresh) & not_tie
    if a_mask.sum() >= 20:
        a_acc = (~actual_home_win[a_mask]).mean()
        a_w = (~actual_home_win[a_mask]).sum()
        print(f"  ≥{conf_thresh:.0%}  {'AWAY':>6s} {a_mask.sum():>7d} {a_w:>5.0f} {a_mask.sum()-a_w:>5.0f} {a_acc:>5.1%}")


# ══════════════════════════════════════════════════════════
# 4. PER-SEASON CONSISTENCY
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  4. PER-SEASON ML ACCURACY")
print(f"{'='*70}")

print(f"\n  {'Season':>7s} {'Games':>7s} {'Correct':>8s} {'Acc':>6s} {'Acc≥60%':>8s} {'N≥60%':>7s} {'Acc≥70%':>8s} {'N≥70%':>7s}")
print("  " + "-" * 70)

for s in sorted(set(szn)):
    sm = (szn == s) & not_tie
    if sm.sum() < 50: continue
    correct = (pred_home_win[sm] == actual_home_win[sm])
    acc_all = correct.mean()

    # At 60% confidence
    m60 = sm & (prob_pick >= 0.60)
    acc60 = (pred_home_win[m60] == actual_home_win[m60]).mean() if m60.sum() >= 20 else 0
    n60 = m60.sum()

    # At 70% confidence
    m70 = sm & (prob_pick >= 0.70)
    acc70 = (pred_home_win[m70] == actual_home_win[m70]).mean() if m70.sum() >= 20 else 0
    n70 = m70.sum()

    print(f"  {s:>7d} {sm.sum():>7d} {correct.sum():>8.0f} {acc_all:>5.1%} {acc60:>7.1%} {n60:>7d} {acc70:>7.1%} {n70:>7d}")

# Combined
sm_all = not_tie
correct_all = (pred_home_win[sm_all] == actual_home_win[sm_all])
m60_all = sm_all & (prob_pick >= 0.60)
m70_all = sm_all & (prob_pick >= 0.70)
acc60_all = (pred_home_win[m60_all] == actual_home_win[m60_all]).mean() if m60_all.sum() >= 20 else 0
acc70_all = (pred_home_win[m70_all] == actual_home_win[m70_all]).mean() if m70_all.sum() >= 20 else 0
print(f"  {'ALL':>7s} {sm_all.sum():>7d} {correct_all.sum():>8.0f} {correct_all.mean():>5.1%} {acc60_all:>7.1%} {m60_all.sum():>7d} {acc70_all:>7.1%} {m70_all.sum():>7d}")


# ══════════════════════════════════════════════════════════
# 5. MONEYLINE ROI SIMULATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  5. MONEYLINE ROI SIMULATION")
print(f"     (If you bet $100 on every ML pick at standard -110 juice)")
print(f"{'='*70}")

print(f"\n  {'Confidence':>11s} {'Picks':>7s} {'Wins':>6s} {'Acc':>6s} {'Wagered':>9s} {'Returned':>9s} {'Profit':>8s} {'ROI':>7s}")
print("  " + "-" * 70)

for conf_thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    m = (prob_pick >= conf_thresh) & not_tie
    if m.sum() < 20: continue
    correct = (pred_home_win[m] == actual_home_win[m])
    n_picks = m.sum()
    n_wins = correct.sum()
    acc = correct.mean()
    wagered = n_picks * 110  # bet $110 to win $100 at -110
    returned = n_wins * 210  # each win returns $210 ($110 stake + $100 profit)
    profit = returned - wagered
    roi = (profit / wagered) * 100
    tag = "✅" if roi > 0 else "❌"
    print(f"  ≥{conf_thresh:.0%}      {n_picks:>7d} {n_wins:>6.0f} {acc:>5.1%} ${wagered:>8,d} ${returned:>8,.0f} ${profit:>+7,.0f} {roi:>+6.1f}% {tag}")

print(f"\n  Note: Actual ML odds vary. -110 is worst case (standard juice).")
print(f"  With +150 underdogs, ROI is significantly higher on away picks.")

print(f"\n  Done.")
