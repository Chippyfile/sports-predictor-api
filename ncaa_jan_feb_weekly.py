#!/usr/bin/env python3
"""
ncaa_jan_feb_weekly.py — Weekly breakdown for Jan/Feb
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

# ── Load data ──
print("=" * 70)
print("  JANUARY / FEBRUARY WEEKLY BREAKDOWN")
print("  3-leg vs 5-leg performance by week")
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
CONF = 0.65; ML_CAP = -500
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
ml_eligible = (pick_ml_all > ML_CAP) | (pick_ml_all > 0)
dates_pd = pd.to_datetime(pd.Series(dates))
date_strs_all = dates_pd.dt.strftime("%Y-%m-%d").values
days_all = dates_pd.dt.day.values
months_all = dates_pd.dt.month.values

TEST_SEASONS = [2019, 2022, 2023, 2024, 2025, 2026]

# ── Define week bins ──
def get_week_label(month, day):
    if month == 1:
        if day <= 7: return "Jan W1 (1-7)"
        elif day <= 14: return "Jan W2 (8-14)"
        elif day <= 21: return "Jan W3 (15-21)"
        else: return "Jan W4 (22-31)"
    elif month == 2:
        if day <= 7: return "Feb W1 (1-7)"
        elif day <= 14: return "Feb W2 (8-14)"
        elif day <= 21: return "Feb W3 (15-21)"
        else: return "Feb W4 (22-28)"
    else:
        return None

WEEK_ORDER = [
    "Jan W1 (1-7)", "Jan W2 (8-14)", "Jan W3 (15-21)", "Jan W4 (22-31)",
    "Feb W1 (1-7)", "Feb W2 (8-14)", "Feb W3 (15-21)", "Feb W4 (22-28)",
]

# ══════════════════════════════════════════════════════════
# WEEKLY RESULTS — ALL SEASONS COMBINED
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  WEEKLY BREAKDOWN — ALL SEASONS COMBINED")
print(f"  ≥65% conf, -500 ML cap, $50 on 3-leg + $25 on 5-leg")
print(f"{'='*70}")

# Collect results by week
week_totals = {}
for wk in WEEK_ORDER:
    week_totals[wk] = {"w3": 0, "l3": 0, "w5": 0, "l5": 0, "tw": 0, "tr": 0, "days": 0}

for season in TEST_SEASONS:
    mask = (seasons == season) & valid_mask
    if mask.sum() < 100: continue
    szn_dates = date_strs_all[mask]; szn_not_tie = not_tie_all[mask]
    szn_prob = prob_pick_all[mask]; szn_conf = confidence_all[mask]
    szn_won = pick_won_all[mask]; szn_dec = pick_dec_all[mask]
    szn_ml_ok = ml_eligible[mask]; szn_months = months_all[mask]; szn_days = days_all[mask]
    unique_dates = sorted(set(szn_dates))

    for date in unique_dates:
        dt = pd.to_datetime(date)
        if dt.month not in [1, 2]: continue
        wk = get_week_label(dt.month, dt.day)
        if wk is None: continue

        dm = (szn_dates == date) & szn_not_tie & (szn_prob >= CONF) & szn_ml_ok
        di = np.where(dm)[0]
        if len(di) < 3: continue
        si = di[np.argsort(-szn_conf[di])]

        week_totals[wk]["days"] += 1

        # 3-leg
        top3 = si[:3]
        pd3 = 1.0; aw3 = True
        for idx in top3:
            pd3 *= szn_dec[idx]
            if not szn_won[idx]: aw3 = False
        week_totals[wk]["tw"] += 50
        if aw3:
            week_totals[wk]["tr"] += 50 * pd3
            week_totals[wk]["w3"] += 1
        else:
            week_totals[wk]["l3"] += 1

        # 5-leg
        if len(si) >= 5:
            top5 = si[:5]
            pd5 = 1.0; aw5 = True
            for idx in top5:
                pd5 *= szn_dec[idx]
                if not szn_won[idx]: aw5 = False
            week_totals[wk]["tw"] += 25
            if aw5:
                week_totals[wk]["tr"] += 25 * pd5
                week_totals[wk]["w5"] += 1
            else:
                week_totals[wk]["l5"] += 1

print(f"\n  {'Week':<18s} {'Days':>5s} {'3L W/L':>8s} {'3L Hit':>7s} "
      f"{'5L W/L':>8s} {'5L Hit':>7s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 90)

for wk in WEEK_ORDER:
    w = week_totals[wk]
    if w["days"] == 0: continue
    h3 = w["w3"] / max(w["w3"] + w["l3"], 1) * 100
    h5 = w["w5"] / max(w["w5"] + w["l5"], 1) * 100
    n5 = w["w5"] + w["l5"]
    profit = w["tr"] - w["tw"]
    roi = (profit / w["tw"] * 100) if w["tw"] > 0 else 0
    tag = "✅" if profit > 0 else "❌"
    h5_str = f"{h5:>6.1f}%" if n5 > 0 else "   n/a"
    print(f"  {wk:<18s} {w['days']:>5d} {w['w3']:>3d}/{w['l3']:<3d} {h3:>6.1f}% "
          f"{w['w5']:>3d}/{w['l5']:<3d} {h5_str} "
          f"${w['tw']:>8,d} ${profit:>+8,.0f} {roi:>+6.1f}% {tag}")

# Jan/Feb totals
for month_label, month_weeks in [("JANUARY", WEEK_ORDER[:4]), ("FEBRUARY", WEEK_ORDER[4:])]:
    tw = sum(week_totals[w]["tw"] for w in month_weeks)
    tr = sum(week_totals[w]["tr"] for w in month_weeks)
    w3 = sum(week_totals[w]["w3"] for w in month_weeks)
    l3 = sum(week_totals[w]["l3"] for w in month_weeks)
    w5 = sum(week_totals[w]["w5"] for w in month_weeks)
    l5 = sum(week_totals[w]["l5"] for w in month_weeks)
    days = sum(week_totals[w]["days"] for w in month_weeks)
    profit = tr - tw
    roi = (profit / tw * 100) if tw > 0 else 0
    h3 = w3 / max(w3 + l3, 1) * 100
    h5 = w5 / max(w5 + l5, 1) * 100
    print(f"  {month_label:<18s} {days:>5d} {w3:>3d}/{l3:<3d} {h3:>6.1f}% "
          f"{w5:>3d}/{l5:<3d} {h5:>6.1f}% "
          f"${tw:>8,d} ${profit:>+8,.0f} {roi:>+6.1f}%")

# ══════════════════════════════════════════════════════════
# PER-SEASON WEEKLY
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PER-SEASON WEEKLY — January & February")
print(f"{'='*70}")

for season in TEST_SEASONS:
    mask = (seasons == season) & valid_mask
    if mask.sum() < 100: continue
    szn_dates = date_strs_all[mask]; szn_not_tie = not_tie_all[mask]
    szn_prob = prob_pick_all[mask]; szn_conf = confidence_all[mask]
    szn_won = pick_won_all[mask]; szn_dec = pick_dec_all[mask]
    szn_ml_ok = ml_eligible[mask]
    unique_dates = sorted(set(szn_dates))

    season_weeks = {}
    for wk in WEEK_ORDER:
        season_weeks[wk] = {"w3": 0, "l3": 0, "w5": 0, "l5": 0, "tw": 0, "tr": 0}

    for date in unique_dates:
        dt = pd.to_datetime(date)
        if dt.month not in [1, 2]: continue
        wk = get_week_label(dt.month, dt.day)
        if wk is None: continue

        dm = (szn_dates == date) & szn_not_tie & (szn_prob >= CONF) & szn_ml_ok
        di = np.where(dm)[0]
        if len(di) < 3: continue
        si = di[np.argsort(-szn_conf[di])]

        top3 = si[:3]
        pd3 = 1.0; aw3 = True
        for idx in top3:
            pd3 *= szn_dec[idx]
            if not szn_won[idx]: aw3 = False
        season_weeks[wk]["tw"] += 50
        if aw3: season_weeks[wk]["tr"] += 50 * pd3; season_weeks[wk]["w3"] += 1
        else: season_weeks[wk]["l3"] += 1

        if len(si) >= 5:
            top5 = si[:5]
            pd5 = 1.0; aw5 = True
            for idx in top5:
                pd5 *= szn_dec[idx]
                if not szn_won[idx]: aw5 = False
            season_weeks[wk]["tw"] += 25
            if aw5: season_weeks[wk]["tr"] += 25 * pd5; season_weeks[wk]["w5"] += 1
            else: season_weeks[wk]["l5"] += 1

    has_data = any(season_weeks[w]["tw"] > 0 for w in WEEK_ORDER)
    if not has_data: continue

    print(f"\n  --- {season} ---")
    print(f"  {'Week':<18s} {'3L':>6s} {'5L':>6s} {'Profit':>9s}")
    print("  " + "-" * 42)

    for wk in WEEK_ORDER:
        w = season_weeks[wk]
        if w["tw"] == 0: continue
        profit = w["tr"] - w["tw"]
        h3 = f"{w['w3']}/{w['w3']+w['l3']}" if (w['w3'] + w['l3']) > 0 else "—"
        n5 = w["w5"] + w["l5"]
        h5 = f"{w['w5']}/{n5}" if n5 > 0 else "—"
        tag = "✅" if profit > 0 else "❌"
        print(f"  {wk:<18s} {h3:>6s} {h5:>6s} ${profit:>+8,.0f} {tag}")


# ══════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  5-LEG OVERRIDE ANALYSIS")
print(f"  Which weeks should drop the 5-leg and go $75 3-leg only?")
print(f"{'='*70}")

print(f"\n  A week should drop 5-leg if 5L ROI is negative across seasons.")
print(f"\n  {'Week':<18s} {'5L Wagered':>11s} {'5L Profit':>10s} {'5L ROI':>7s} {'Recommendation'}")
print("  " + "-" * 65)

for wk in WEEK_ORDER:
    w = week_totals[wk]
    n5 = w["w5"] + w["l5"]
    if n5 == 0:
        print(f"  {wk:<18s} {'$0':>11s} {'$0':>10s} {'n/a':>7s} Skip (no data)")
        continue
    
    # Calculate 5-leg P&L only
    # We need to separate 3-leg and 5-leg returns
    # Approximate: 5-leg wagered = n5 * 25, returned = w5 * avg_payout
    w5_wagered = n5 * 25
    # For the 5-leg profit, we need actual returns. Approximate from overall:
    # total_wagered = 3L_wagered + 5L_wagered = days*50 + n5*25
    # We can estimate 5L contribution
    w3_wagered = (w["w3"] + w["l3"]) * 50
    w5_return = w["tr"] - (w["w3"] * 50 * 1.5)  # rough: 3L wins return ~1.5x on avg
    # Better approach: just look at hit rate vs break-even
    h5 = w["w5"] / n5
    # At ~1.7x avg odds, break-even is ~59%
    be = 1.0 / 1.7  # ~58.8%
    is_profitable = h5 >= be
    
    roi_est = (h5 * 1.7 - 1) * 100  # estimated ROI from hit rate and avg odds
    rec = "KEEP 5-leg" if is_profitable else "DROP → $75 3-leg only"
    tag = "✅" if is_profitable else "⛔"
    print(f"  {wk:<18s} {n5*25:>10d} {'':>10s} {roi_est:>+6.1f}% {rec} {tag}")

print(f"\n  Done.")
