#!/usr/bin/env python3
"""
ncaa_open_vs_close.py — Does betting early (opening line) give better ATS?
==========================================================================
Grades the same walk-forward predictions against:
  A) Closing spread (what backtest uses)
  B) Opening spread (what you'd bet if syncing early)

Only games with BOTH opening and closing spreads are compared.
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


def grade_ats(oof, y, spreads, label=""):
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
    p = oof[valid]; a = y[valid]; s = spreads[valid]
    edge = p - (-s); margin = a + s
    np_ = margin != 0; cor = np.sign(edge) == np.sign(margin)

    print(f"\n  === {label} ({valid.sum()} games) ===")
    print(f"  {'Edge':>6s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s}")
    print("  " + "-" * 30)

    for t in [0, 2, 4, 6, 7, 8, 10]:
        m = (np.abs(edge) >= t) & np_; n = m.sum()
        if n < 10: continue
        acc = float(cor[m].mean())
        roi = round((acc * 1.909 - 1) * 100, 1)
        tag = "✅" if acc > 0.524 else "❌"
        print(f"  {t:>5d}+ {n:>7d} {acc:>5.1%} {roi:>+6.1f}% {tag}")


# ── Load data ──
print("=" * 70)
print("  OPENING vs CLOSING SPREAD — ATS Comparison")
print("=" * 70)

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
closing = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
seasons = df["season"].values

# Get opening spreads (DraftKings or Odds API)
opening = np.zeros(len(df))
for col in ["dk_spread_open", "odds_api_spread_open", "opening_spread_home", "opening_spread"]:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        # Fill where we don't have opening yet
        mask = (np.abs(opening) < 0.1) & (np.abs(vals) > 0.1)
        opening[mask] = vals[mask]
        print(f"  Opening spread from '{col}': {(np.abs(vals) > 0.1).sum()} games")

has_both = (np.abs(closing) > 0.1) & (np.abs(opening) > 0.1)
print(f"\n  Total games: {len(df)}")
print(f"  Games with closing spread: {(np.abs(closing) > 0.1).sum()}")
print(f"  Games with opening spread: {(np.abs(opening) > 0.1).sum()}")
print(f"  Games with BOTH: {has_both.sum()}")

# Per-season opening coverage
for s in sorted(set(seasons)):
    sm = seasons == s
    n_open = ((np.abs(opening) > 0.1) & sm).sum()
    n_close = ((np.abs(closing) > 0.1) & sm).sum()
    n_total = sm.sum()
    print(f"    {s}: open={n_open}/{n_total} ({100*n_open/max(n_total,1):.0f}%), close={n_close}/{n_total} ({100*n_close/max(n_total,1):.0f}%)")

print(f"\n  {len(X)} games × {len(available)} features")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ── Walk-forward ──
print(f"\n  {N_FOLDS}-fold walk-forward...")
t0 = time.time()
oof = walk_forward(X_s, y, N_FOLDS)
print(f"  Done in {time.time()-t0:.0f}s")

# ── Grade against closing spread (all games) ──
grade_ats(oof, y, closing, "ALL GAMES — CLOSING SPREAD")

# ── Grade against opening spread (subset with both) ──
oof_both = oof.copy()
oof_both[~has_both] = np.nan  # mask out games without opening spread

grade_ats(oof_both, y, opening, "OPENING SPREAD (games with both lines)")
grade_ats(oof_both, y, closing, "CLOSING SPREAD (same games, for comparison)")

# ── Line movement analysis ──
if has_both.sum() > 100:
    movement = closing[has_both] - opening[has_both]
    print(f"\n  === LINE MOVEMENT ANALYSIS ({has_both.sum()} games) ===")
    print(f"  Avg movement: {np.mean(movement):+.2f} pts")
    print(f"  Std movement: {np.std(movement):.2f} pts")
    print(f"  Games that moved ≥1pt: {(np.abs(movement) >= 1).sum()} ({100*(np.abs(movement)>=1).sum()/has_both.sum():.0f}%)")
    print(f"  Games that moved ≥2pt: {(np.abs(movement) >= 2).sum()} ({100*(np.abs(movement)>=2).sum()/has_both.sum():.0f}%)")
    
    # Does the model agree with the direction of line movement?
    model_pred = oof[has_both]
    valid_mv = ~np.isnan(model_pred) & (np.abs(movement) >= 0.5)
    if valid_mv.sum() > 50:
        model_side = model_pred[valid_mv] - (-opening[has_both][valid_mv])  # edge vs opening
        move_dir = movement[valid_mv]  # positive = line moved toward home
        agree = np.sign(model_side) == np.sign(move_dir)
        print(f"\n  Model agrees with line movement direction: {agree.mean():.1%} ({valid_mv.sum()} games)")
        print(f"  (>50% = model and sharp money on same side = good)")

print(f"\n  Done.")
