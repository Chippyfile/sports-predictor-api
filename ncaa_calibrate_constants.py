#!/usr/bin/env python3
"""
ncaa_calibrate_constants.py — Empirically calibrate sigma + season weights
==========================================================================
1. Sigma: finds optimal value that converts margin → win probability
   with best Brier score (currently 10 backend, 16 frontend — which is right?)
2. Season weights: tests different decay functions to find optimal training window

Usage:
    python3 ncaa_calibrate_constants.py
"""
import sys, os, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from dump_training_data import load_cached, dump
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42; N_FOLDS = 20  # Fewer folds for speed

FEATURES_43 = [
    "mkt_spread","player_rating_diff","ref_home_whistle","weakest_starter_diff",
    "crowd_shock_diff","lineup_stability_diff","lineup_changes_diff","adj_oe_diff",
    "hca_pts","blowout_asym_diff","threepct_diff","pit_sos_diff","orb_pct_diff",
    "blocks_diff","drb_pct_diff","opp_to_rate_diff","elo_diff","is_early",
    "spread_regime","assist_rate_diff","opp_ppg_diff","opp_suppression_diff",
    "roll_ats_margin_gated","has_ats_data","tempo_avg","form_x_familiarity",
    "to_conversion_diff","conf_strength_diff","roll_rotation_diff","roll_dominance_diff",
    "importance","twopt_diff","roll_ats_diff_gated","overreaction_diff",
    "three_rate_diff","ppp_diff","to_margin_diff","momentum_halflife_diff",
    "starter_experience_diff","style_familiarity","fatigue_x_quality","ato_diff",
    "consistency_x_spread"
]

MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}

def walk_forward(X_s, y, w, n_folds):
    """Walk-forward with optional sample weights."""
    n = len(X_s); fold_size = n // (n_folds + 1); min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder()
            if w is not None and hasattr(m, 'fit'):
                try:
                    m.fit(X_s[:ts], y[:ts], sample_weight=w[:ts])
                except TypeError:
                    m.fit(X_s[:ts], y[:ts])
            else:
                m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof

print("=" * 70)
print("  NCAA CONSTANT CALIBRATION")
print("=" * 70)

# ── Load and prep data ──
df = load_cached()
if df is None: df = dump()
df = df[df["actual_home_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
df = df[season_mask & early_mask].copy()

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

for col in ["actual_home_score","actual_away_score","season"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

print(f"\n  {len(df):,} games loaded")

# ── Build features ──
print("  Building features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError: pass
try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
except: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])
X_full = ncaa_build_features(df)
available = [f for f in FEATURES_43 if f in X_full.columns]
X = X_full[available]
y = df["actual_home_score"].values - df["actual_away_score"].values
spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
seasons = df["season"].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print(f"  {len(available)} features, {len(y)} games")

# ══════════════════════════════════════════════════════════
# PART 1: SIGMA CALIBRATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PART 1: SIGMA CALIBRATION")
print(f"  Current: backend=10.0, frontend=16.0")
print(f"  Testing: 5.0 to 25.0 in steps of 0.5")
print(f"{'='*70}")

# Generate walk-forward predictions once
print(f"\n  Running {N_FOLDS}-fold walk-forward...")
oof = walk_forward(X_s, y, None, N_FOLDS)
valid = ~np.isnan(oof)
pred = oof[valid]
actual = y[valid]
home_won = (actual > 0).astype(float)

print(f"  {valid.sum():,} valid predictions")

# Test different sigma values
sigmas = np.arange(5.0, 25.5, 0.5)
results = []

print(f"\n  {'Sigma':>6s} {'Brier':>8s} {'LogLoss':>8s} {'Calib@50':>9s} {'Calib@70':>9s} {'Calib@90':>9s}")
print(f"  {'-'*52}")

for sigma in sigmas:
    prob_home = 1.0 / (1.0 + np.exp(-pred / sigma))
    
    # Brier score (lower = better)
    brier = np.mean((prob_home - home_won) ** 2)
    
    # Log loss (lower = better)
    eps = 1e-7
    prob_clipped = np.clip(prob_home, eps, 1 - eps)
    logloss = -np.mean(home_won * np.log(prob_clipped) + (1 - home_won) * np.log(1 - prob_clipped))
    
    # Calibration: for games where model says X%, does home actually win X%?
    # Check at 50%, 70%, 90% thresholds
    calib = {}
    for thresh in [0.50, 0.70, 0.90]:
        mask = prob_home >= thresh
        if mask.sum() >= 20:
            actual_pct = home_won[mask].mean()
            calib[thresh] = actual_pct
        else:
            calib[thresh] = None
    
    c50 = f"{calib[0.50]:.1%}" if calib[0.50] else "n/a"
    c70 = f"{calib[0.70]:.1%}" if calib[0.70] else "n/a"
    c90 = f"{calib[0.90]:.1%}" if calib[0.90] else "n/a"
    
    results.append({"sigma": sigma, "brier": brier, "logloss": logloss, **calib})
    
    marker = ""
    if sigma == 10.0: marker = " ← backend"
    elif sigma == 16.0: marker = " ← frontend"
    
    print(f"  {sigma:>6.1f} {brier:>8.5f} {logloss:>8.5f} {c50:>9s} {c70:>9s} {c90:>9s}{marker}")

# Find optimal
best = min(results, key=lambda x: x["brier"])
print(f"\n  ✅ OPTIMAL SIGMA: {best['sigma']:.1f} (Brier {best['brier']:.5f})")
print(f"  Backend (10.0): Brier {[r for r in results if r['sigma']==10.0][0]['brier']:.5f}")
print(f"  Frontend (16.0): Brier {[r for r in results if r['sigma']==16.0][0]['brier']:.5f}")

# Calibration plot at optimal sigma
opt_sigma = best["sigma"]
opt_prob = 1.0 / (1.0 + np.exp(-pred / opt_sigma))
print(f"\n  Calibration at sigma={opt_sigma:.1f}:")
print(f"  {'Predicted':>10s} {'Actual':>8s} {'Games':>7s} {'Error':>7s}")
print(f"  {'-'*35}")
for lo, hi in [(0.5,0.55),(0.55,0.6),(0.6,0.65),(0.65,0.7),(0.7,0.75),(0.75,0.8),(0.8,0.85),(0.85,0.9),(0.9,0.95),(0.95,1.0)]:
    mask = (opt_prob >= lo) & (opt_prob < hi)
    if mask.sum() >= 20:
        actual_pct = home_won[mask].mean()
        expected = (lo + hi) / 2
        error = actual_pct - expected
        print(f"  {lo:.0%}-{hi:.0%} {actual_pct:>7.1%} {mask.sum():>7d} {error:>+6.1%}")

# ══════════════════════════════════════════════════════════
# PART 2: SEASON WEIGHT CALIBRATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PART 2: SEASON WEIGHT DECAY")
print(f"  Current: 1.0/0.9/0.75/0.6/0.5 by age")
print(f"  Testing: flat, gentle, current, aggressive, recent-only")
print(f"{'='*70}")

current_year = 2026

WEIGHT_SCHEMES = {
    "flat": lambda age: 1.0,
    "gentle": lambda age: max(0.7, 1.0 - age * 0.05),
    "current": lambda age: {0: 1.0, 1: 0.9, 2: 0.75, 3: 0.6}.get(age, 0.5),
    "aggressive": lambda age: max(0.3, 1.0 - age * 0.15),
    "last_3_only": lambda age: 1.0 if age <= 2 else 0.2,
    "last_2_only": lambda age: 1.0 if age <= 1 else 0.3,
    "exponential": lambda age: 0.8 ** age,
}

print(f"\n  Running walk-forward with different weight schemes...")
print(f"  {'Scheme':<15s} {'MAE':>8s} {'Brier':>8s} {'ATS@7+':>8s} {'Picks@7':>8s}")
print(f"  {'-'*50}")

for name, weight_fn in WEIGHT_SCHEMES.items():
    weights = np.array([weight_fn(current_year - s) for s in seasons])
    
    oof_w = walk_forward(X_s, y, weights, N_FOLDS)
    valid_w = ~np.isnan(oof_w) & (np.abs(spreads) > 0.5)
    pred_w = oof_w[valid_w]; act_w = y[valid_w]; sp_w = spreads[valid_w]
    
    mae = np.mean(np.abs(pred_w - act_w))
    
    prob_w = 1.0 / (1.0 + np.exp(-pred_w / opt_sigma))
    hw_w = (act_w > 0).astype(float)
    brier = np.mean((prob_w - hw_w) ** 2)
    
    # ATS@7
    ats = pred_w + sp_w
    disagree = np.abs(pred_w + sp_w)
    mask7 = disagree >= 7
    if mask7.sum() > 0:
        covers = ((ats > 0) == (act_w + sp_w > 0))
        ats7 = covers[mask7].mean()
        n7 = mask7.sum()
    else:
        ats7 = 0; n7 = 0
    
    marker = " ← current" if name == "current" else ""
    print(f"  {name:<15s} {mae:>8.3f} {brier:>8.5f} {ats7:>7.1%} {n7:>8d}{marker}")

# ══════════════════════════════════════════════════════════
# PART 3: PARLAY CONFIDENCE GATE (sigma affects this)
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PART 3: PARLAY CONFIDENCE GATE — Impact of sigma on picks")
print(f"  At 65% confidence gate, how many picks qualify per sigma?")
print(f"{'='*70}")

for sigma in [8.0, 10.0, 12.0, 14.0, 16.0, opt_sigma]:
    prob = 1.0 / (1.0 + np.exp(-pred / sigma))
    conf = np.maximum(prob, 1 - prob)
    qualifying = (conf >= 0.65).sum()
    pct = qualifying / len(pred) * 100
    
    # Of qualifying picks, what's the actual win rate?
    mask = conf >= 0.65
    if mask.sum() > 0:
        pick_home = prob[mask] > 0.5
        actual_home_w = home_won[mask]
        correct = ((pick_home & (actual_home_w == 1)) | (~pick_home & (actual_home_w == 0))).mean()
    else:
        correct = 0
    
    tag = ""
    if sigma == 10.0: tag = " ← backend"
    elif sigma == 16.0: tag = " ← frontend"
    elif sigma == opt_sigma: tag = " ← OPTIMAL"
    
    print(f"  σ={sigma:>5.1f}: {qualifying:>5d} picks ({pct:>4.1f}%), ML accuracy: {correct:.1%}{tag}")

print(f"\n  Done.")
