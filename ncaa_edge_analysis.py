#!/usr/bin/env python3
"""
ncaa_edge_analysis.py — Signed edge breakdown by season
========================================================
Walk-forward predictions → ATS accuracy at every signed edge bucket,
per season and combined. Shows where the model's edge is strongest.

Usage:
    python3 ncaa_edge_analysis.py
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
print("  NCAA SIGNED EDGE ANALYSIS BY SEASON")
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
if "espn_over_under" in df.columns:
    espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce")
    mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
    fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
    df.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

_qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score","actual_away_score","home_adj_em","away_adj_em","home_ppg","away_ppg",
            "home_tempo","away_tempo","season","home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
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
spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
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

# ── Compute edges ──
valid = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
pred = oof[valid]; actual = y[valid]; spr = spreads[valid]; szn = seasons[valid]

edge = pred - (-spr)  # positive = model says home covers
margin = actual + spr  # positive = home actually covered
not_push = margin != 0
correct = np.sign(edge) == np.sign(margin)

# ── Signed edge thresholds ──
thresholds = [-12, -10, -8, -7, -6, -4, -2, 0, 2, 4, 6, 7, 8, 10, 12]
unique_seasons = sorted(set(szn))

print(f"\n{'='*70}")
print(f"  SIGNED EDGE ANALYSIS — Cumulative ATS by Direction")
print(f"  Positive edge = model says HOME covers")
print(f"  Negative edge = model says AWAY covers")
print(f"{'='*70}")

# ── CUMULATIVE: edge >= threshold (for positive) and edge <= -threshold (for negative) ──
print(f"\n  === COMBINED (all {valid.sum()} games) ===")
print(f"  {'Edge':<10s} {'Side':<6s} {'Games':>7s} {'W':>5s} {'L':>5s} {'Acc':>6s} {'ROI':>7s}")
print("  " + "-" * 50)

for t in thresholds:
    if t >= 0:
        mask = (edge >= t) & not_push
        side = "HOME"
    else:
        mask = (edge <= t) & not_push
        side = "AWAY"

    n_games = mask.sum()
    if n_games < 10:
        continue
    wins = correct[mask].sum()
    losses = n_games - wins
    acc = wins / n_games
    roi = (acc * 1.909 - 1) * 100
    tag = "✅" if acc > 0.524 else "❌"
    label = f"{t:+d}+" if t >= 0 else f"{t:+d}−"
    print(f"  {label:<10s} {side:<6s} {n_games:>7d} {wins:>5.0f} {losses:>5.0f} {acc:>5.1%} {roi:>+6.1f}% {tag}")

# ── PER-SEASON BREAKDOWN ──
print(f"\n{'='*70}")
print(f"  PER-SEASON SIGNED EDGE BREAKDOWN")
print(f"{'='*70}")

for szn_val in unique_seasons:
    szn_mask = szn == szn_val
    szn_n = szn_mask.sum()
    if szn_n < 50:
        continue

    print(f"\n  === SEASON {szn_val} ({szn_n} graded games) ===")
    print(f"  {'Edge':<10s} {'Side':<6s} {'Games':>7s} {'W':>5s} {'L':>5s} {'Acc':>6s} {'ROI':>7s}")
    print("  " + "-" * 50)

    for t in thresholds:
        if t >= 0:
            mask = (edge >= t) & not_push & szn_mask
            side = "HOME"
        else:
            mask = (edge <= t) & not_push & szn_mask
            side = "AWAY"

        n_games = mask.sum()
        if n_games < 5:
            continue
        wins = correct[mask].sum()
        losses = n_games - wins
        acc = wins / n_games
        roi = (acc * 1.909 - 1) * 100
        tag = "✅" if acc > 0.524 else "❌"
        label = f"{t:+d}+" if t >= 0 else f"{t:+d}−"
        print(f"  {label:<10s} {side:<6s} {n_games:>7d} {wins:>5.0f} {losses:>5.0f} {acc:>5.1%} {roi:>+6.1f}% {tag}")


# ── EDGE BUCKETS (ranges, not cumulative) ──
print(f"\n{'='*70}")
print(f"  EDGE BUCKETS (non-cumulative ranges)")
print(f"{'='*70}")

buckets = [
    (-999, -12, "Strong AWAY (< -12)"),
    (-12, -10, "AWAY -12 to -10"),
    (-10, -8, "AWAY -10 to -8"),
    (-8, -6, "AWAY -8 to -6"),
    (-6, -4, "AWAY -6 to -4"),
    (-4, -2, "AWAY -4 to -2"),
    (-2, 0, "AWAY -2 to 0"),
    (0, 2, "HOME 0 to +2"),
    (2, 4, "HOME +2 to +4"),
    (4, 6, "HOME +4 to +6"),
    (6, 8, "HOME +6 to +8"),
    (8, 10, "HOME +8 to +10"),
    (10, 12, "HOME +10 to +12"),
    (12, 999, "Strong HOME (> +12)"),
]

print(f"\n  {'Bucket':<25s} {'Games':>7s} {'W':>5s} {'L':>5s} {'Acc':>6s} {'ROI':>7s}")
print("  " + "-" * 60)

for lo, hi, label in buckets:
    if lo == -999:
        mask = (edge < hi) & not_push
    elif hi == 999:
        mask = (edge >= lo) & not_push
    else:
        mask = (edge >= lo) & (edge < hi) & not_push
    
    n_games = mask.sum()
    if n_games < 10:
        continue
    wins = correct[mask].sum()
    losses = n_games - wins
    acc = wins / n_games
    roi = (acc * 1.909 - 1) * 100
    tag = "✅" if acc > 0.524 else "❌"
    print(f"  {label:<25s} {n_games:>7d} {wins:>5.0f} {losses:>5.0f} {acc:>5.1%} {roi:>+6.1f}% {tag}")

print(f"\n  Done.")
