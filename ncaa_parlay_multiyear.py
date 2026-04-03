#!/usr/bin/env python3
"""
ncaa_parlay_multiyear.py — Validate parlay strategies across 2019-2026
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

SEED = 42; N_FOLDS = 30
FEATURES_43 = ["mkt_spread","player_rating_diff","ref_home_whistle","weakest_starter_diff","crowd_shock_diff","lineup_stability_diff","lineup_changes_diff","adj_oe_diff","hca_pts","blowout_asym_diff","threepct_diff","pit_sos_diff","orb_pct_diff","blocks_diff","drb_pct_diff","opp_to_rate_diff","elo_diff","is_early","spread_regime","assist_rate_diff","opp_ppg_diff","opp_suppression_diff","roll_ats_margin_gated","has_ats_data","tempo_avg","form_x_familiarity","to_conversion_diff","conf_strength_diff","roll_rotation_diff","roll_dominance_diff","importance","twopt_diff","roll_ats_diff_gated","overreaction_diff","three_rate_diff","ppp_diff","to_margin_diff","momentum_halflife_diff","starter_experience_diff","style_familiarity","fatigue_x_quality","ato_diff","consistency_x_spread"]
MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED),
}

def walk_forward(X_s, y, n_folds):
    n = len(X_s); fold_size = n // (n_folds + 1); min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder(); m.fit(X_s[:ts], y[:ts]); preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof

def spread_to_ml(spread):
    s = abs(spread)
    if s < 0.5: return (-110, -110)
    pairs = [(1,-120,100),(2,-140,120),(3,-160,135),(4,-185,155),(5,-210,175),(6,-245,200),(7,-280,230),(8,-320,260),(9,-370,295),(10,-420,330),(12,-550,410),(14,-700,500),(16,-900,600),(18,-1200,750),(20,-1500,900)]
    fav_ml, dog_ml = -2000, 1200
    for lim, f, d in pairs:
        if s <= lim: fav_ml = f; dog_ml = d; break
    return (fav_ml, dog_ml) if spread < 0 else (dog_ml, fav_ml)

def ml_to_decimal(ml):
    return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1

month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

# ── Load data ──
print("=" * 70)
print("  MULTI-YEAR PARLAY VALIDATION (2019, 2022-2026)")
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

for col in ["actual_home_score","actual_away_score","season","home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]
df["season_weight"] = df["season"].apply(lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else 0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

print("\n  Loading + building features...")
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
dates = df["game_date_dt"].values if "game_date_dt" in df.columns else pd.to_datetime(df.get("game_date", "")).values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
print(f"  {len(X)} total games")
print(f"\n  {N_FOLDS}-fold walk-forward...")
t0 = time.time()
oof = walk_forward(X_s, y, N_FOLDS)
print(f"  Done in {time.time()-t0:.0f}s")

# Precompute
valid_mask = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
pred_home_win_all = oof > 0
actual_home_win_all = y > 0
not_tie_all = y != 0
confidence_all = np.abs(oof)
prob_home_all = 1.0 / (1.0 + np.exp(-oof / 10.0))
prob_pick_all = np.maximum(prob_home_all, 1 - prob_home_all)
home_ml_all = np.zeros(len(oof)); away_ml_all = np.zeros(len(oof))
for i, sp in enumerate(spreads):
    hml, aml = spread_to_ml(sp); home_ml_all[i] = hml; away_ml_all[i] = aml
pick_ml_all = np.where(pred_home_win_all, home_ml_all, away_ml_all)
pick_won_all = (pred_home_win_all == actual_home_win_all) & not_tie_all
pick_dec_all = np.array([ml_to_decimal(ml) for ml in pick_ml_all])

# Skip extreme chalk — legs worse than -1000 add almost no value to parlays
ML_CAP = -1000
ml_eligible = (pick_ml_all > ML_CAP) | (pick_ml_all > 0)  # skip favorites beyond -1000
print(f"  ML cap: skip legs worse than {ML_CAP} ({(~ml_eligible).sum()} legs excluded)")
dates_pd = pd.to_datetime(pd.Series(dates))
date_strs_all = dates_pd.dt.strftime("%Y-%m-%d").values
months_all = dates_pd.dt.month.values

MIN_LEGS = 3
TEST_SEASONS = [2019, 2022, 2023, 2024, 2025, 2026]
STRATEGIES = [("65% flex→5", 0.65, 5), ("65% flex→4", 0.65, 4), ("65% flex→3", 0.65, 3),
              ("60% flex→5", 0.60, 5), ("70% flex→5", 0.70, 5), ("75% flex→5", 0.75, 5), ("80% flex→5", 0.80, 5)]

def run_season(season, conf_thresh, max_legs):
    mask = (seasons == season) & valid_mask
    if mask.sum() < 100: return None
    szn_dates = date_strs_all[mask]; szn_not_tie = not_tie_all[mask]
    szn_prob = prob_pick_all[mask]; szn_conf = confidence_all[mask]
    szn_won = pick_won_all[mask]; szn_dec = pick_dec_all[mask]; szn_months = months_all[mask]
    szn_ml_ok = ml_eligible[mask]
    unique_dates = sorted(set(szn_dates))
    tw = 0; tr = 0; np_ = 0; nw = 0; gd = 0; rd = 0
    ld = {}; monthly = {}
    for date in unique_dates:
        dm = (szn_dates == date) & szn_not_tie & (szn_prob >= conf_thresh) & szn_ml_ok
        di = np.where(dm)[0]
        if len(di) < MIN_LEGS: continue
        si = di[np.argsort(-szn_conf[di])]; al = min(len(si), max_legs); tp = si[:al]
        ld[al] = ld.get(al, 0) + 1
        month = pd.to_datetime(date).month
        if month not in monthly: monthly[month] = {"w": 0, "r": 0, "wi": 0, "p": 0}
        pd_ = 1.0; aw = True
        for idx in tp:
            pd_ *= szn_dec[idx]
            if not szn_won[idx]: aw = False
        w = 20; tw += w; np_ += 1; monthly[month]["w"] += w; monthly[month]["p"] += 1
        if aw:
            pay = w * pd_; tr += pay; nw += 1; monthly[month]["r"] += pay; monthly[month]["wi"] += 1; gd += 1
        else: rd += 1
    if np_ == 0: return None
    profit = tr - tw; roi = (profit / tw * 100) if tw > 0 else 0; hit = (nw / np_ * 100) if np_ > 0 else 0
    return {"season": season, "days": np_, "hit": hit, "wagered": tw, "profit": profit, "roi": roi,
            "green": gd, "red": rd, "wins": nw, "legs_dist": ld, "monthly": monthly}

def run_stack_season(season, configs):
    mask = (seasons == season) & valid_mask
    if mask.sum() < 100: return None
    szn_dates = date_strs_all[mask]; szn_not_tie = not_tie_all[mask]
    szn_prob = prob_pick_all[mask]; szn_conf = confidence_all[mask]
    szn_won = pick_won_all[mask]; szn_dec = pick_dec_all[mask]
    szn_ml_ok = ml_eligible[mask]
    unique_dates = sorted(set(szn_dates))
    tw = 0; tr = 0; nb = 0
    for date in unique_dates:
        for ct, nl in configs:
            dm = (szn_dates == date) & szn_not_tie & (szn_prob >= ct) & szn_ml_ok
            di = np.where(dm)[0]
            if len(di) < nl: continue
            si = di[np.argsort(-szn_conf[di])]; tp = si[:nl]
            pd_ = 1.0; aw = True
            for idx in tp:
                pd_ *= szn_dec[idx]
                if not szn_won[idx]: aw = False
            tw += 20; nb += 1
            if aw: tr += 20 * pd_
    if nb == 0: return None
    profit = tr - tw; roi = (profit / tw * 100) if tw > 0 else 0
    return {"season": season, "bets": nb, "wagered": tw, "profit": profit, "roi": roi}

# ══════════════════════════════════════════════════════════
# 1. PER-SEASON — SINGLE PARLAYS
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  1. SINGLE PARLAY — PER SEASON")
print(f"{'='*70}")

for label, conf, max_legs in STRATEGIES:
    print(f"\n  --- {label} ---")
    print(f"  {'Season':>7s} {'Days':>5s} {'Hit%':>6s} {'G':>4s} {'R':>4s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s} {'Legs'}")
    print("  " + "-" * 75)
    tp = 0; tw = 0; td = 0; twi = 0; ps = 0
    for season in TEST_SEASONS:
        r = run_season(season, conf, max_legs)
        if r is None: print(f"  {season:>7d}   (insufficient data)"); continue
        ld = r["legs_dist"]; legs_str = " ".join(f"{k}L:{v}" for k,v in sorted(ld.items()) if v > 0)
        tag = "✅" if r["profit"] > 0 else "❌"
        print(f"  {season:>7d} {r['days']:>5d} {r['hit']:>5.1f}% {r['green']:>4d} {r['red']:>4d} "
              f"${r['wagered']:>8,d} ${r['profit']:>+8,.0f} {r['roi']:>+6.1f}% {legs_str} {tag}")
        tp += r["profit"]; tw += r["wagered"]; td += r["days"]; twi += r["wins"]
        if r["profit"] > 0: ps += 1
    if tw > 0:
        print(f"  {'TOTAL':>7s} {td:>5d} {twi/td*100:>5.1f}% {'':>4s} {'':>4s} "
              f"${tw:>8,d} ${tp:>+8,.0f} {tp/tw*100:>+6.1f}% ({ps}/{len(TEST_SEASONS)} profitable)")

# ══════════════════════════════════════════════════════════
# 2. PER-SEASON — STACKING
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  2. STACKING — PER SEASON")
print(f"{'='*70}")

STACKS = [("3+5 leg ≥65%", [(0.65,3),(0.65,5)]),
          ("3+4+5 leg ≥65%", [(0.65,3),(0.65,4),(0.65,5)]),
          ("3+4+5 leg ≥70%", [(0.70,3),(0.70,4),(0.70,5)]),
          ("3+4 leg ≥65%", [(0.65,3),(0.65,4)]),
          ("4+5 leg ≥65%", [(0.65,4),(0.65,5)])]

for label, configs in STACKS:
    print(f"\n  --- {label} ($20/parlay) ---")
    print(f"  {'Season':>7s} {'Bets':>6s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s}")
    print("  " + "-" * 45)
    tp = 0; tw = 0; ps = 0
    for season in TEST_SEASONS:
        r = run_stack_season(season, configs)
        if r is None: continue
        tag = "✅" if r["profit"] > 0 else "❌"
        print(f"  {r['season']:>7d} {r['bets']:>6d} ${r['wagered']:>8,d} ${r['profit']:>+8,.0f} {r['roi']:>+6.1f}% {tag}")
        tp += r["profit"]; tw += r["wagered"]
        if r["profit"] > 0: ps += 1
    if tw > 0:
        print(f"  {'TOTAL':>7s} {'':>6s} ${tw:>8,d} ${tp:>+8,.0f} {tp/tw*100:>+6.1f}% ({ps}/{len(TEST_SEASONS)} profitable)")

# ══════════════════════════════════════════════════════════
# 3. MONTHLY HEATMAP
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  3. MONTHLY HEATMAP — ≥65% flex→5")
print(f"{'='*70}")
print(f"\n  {'Season':>7s}", end="")
for m in [11,12,1,2,3,4]: print(f" {month_names[m]:>8s}", end="")
print(f" {'TOTAL':>9s}")
print("  " + "-" * 70)
gm = {}
for season in TEST_SEASONS:
    r = run_season(season, 0.65, 5)
    if r is None: continue
    print(f"  {season:>7d}", end=""); sp = 0
    for m in [11,12,1,2,3,4]:
        md = r["monthly"].get(m)
        if md and md["p"] > 0:
            mp = md["r"] - md["w"]; sp += mp; gm[m] = gm.get(m, 0) + mp
            print(f" ${mp:>+7,.0f}", end="")
        else: print(f" {'—':>8s}", end="")
    print(f" ${sp:>+8,.0f}")
print(f"  {'TOTAL':>7s}", end=""); gt = 0
for m in [11,12,1,2,3,4]:
    mp = gm.get(m, 0); gt += mp; print(f" ${mp:>+7,.0f}", end="")
print(f" ${gt:>+8,.0f}")

# ══════════════════════════════════════════════════════════
# 4. CUMULATIVE P&L
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  4. CUMULATIVE P&L — ≥65% flex→5, $20/parlay")
print(f"{'='*70}")
cum = 0
print(f"\n  {'Season':>7s} {'Profit':>9s} {'Cumulative':>11s}")
print("  " + "-" * 30)
for season in TEST_SEASONS:
    r = run_season(season, 0.65, 5)
    if r is None: continue
    cum += r["profit"]
    tag = "✅" if cum > 0 else "❌"
    print(f"  {season:>7d} ${r['profit']:>+8,.0f} ${cum:>+10,.0f} {tag}")
print(f"\n  Avg/season: ${cum/len(TEST_SEASONS):>+,.0f}")
print(f"\n  Done.")
