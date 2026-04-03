#!/usr/bin/env python3
"""
ncaa_final_strategy.py — Final strategy validation
===================================================
3+5 stack, ≥65% conf, -500 ML cap, $50 on 3-leg + $25 on 5-leg
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

# ── STRATEGY PARAMS ──
CONF = 0.65
ML_CAP = -500
BET_3LEG = 50
BET_5LEG = 25
MIN_LEGS = 3
TEST_SEASONS = [2019, 2022, 2023, 2024, 2025, 2026]

# ── Load data ──
print("=" * 70)
print(f"  FINAL STRATEGY VALIDATION")
print(f"  3+5 stack | ≥{CONF:.0%} conf | ML cap {ML_CAP}")
print(f"  ${BET_3LEG} on 3-leg + ${BET_5LEG} on 5-leg = ${BET_3LEG+BET_5LEG}/day")
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
ml_eligible = (pick_ml_all > ML_CAP) | (pick_ml_all > 0)
dates_pd = pd.to_datetime(pd.Series(dates))
date_strs_all = dates_pd.dt.strftime("%Y-%m-%d").values
months_all = dates_pd.dt.month.values

print(f"  ML cap {ML_CAP}: {(~ml_eligible).sum()} legs excluded")

# ══════════════════════════════════════════════════════════
# RUN STRATEGY PER SEASON
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PER-SEASON RESULTS")
print(f"{'='*70}")

print(f"\n  {'Season':>7s} {'Days':>5s} {'3L W':>5s} {'3L L':>5s} {'3L Hit':>7s} "
      f"{'5L W':>5s} {'5L L':>5s} {'5L Hit':>7s} "
      f"{'Wagered':>9s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 95)

cumulative = 0
all_season_data = []

for season in TEST_SEASONS:
    mask = (seasons == season) & valid_mask
    if mask.sum() < 100: continue
    szn_dates = date_strs_all[mask]; szn_not_tie = not_tie_all[mask]
    szn_prob = prob_pick_all[mask]; szn_conf = confidence_all[mask]
    szn_won = pick_won_all[mask]; szn_dec = pick_dec_all[mask]
    szn_ml_ok = ml_eligible[mask]; szn_months = months_all[mask]
    unique_dates = sorted(set(szn_dates))

    tw = 0; tr = 0
    w3 = 0; l3 = 0; w5 = 0; l5 = 0
    daily_data = []
    monthly = {}

    for date in unique_dates:
        dm = (szn_dates == date) & szn_not_tie & (szn_prob >= CONF) & szn_ml_ok
        di = np.where(dm)[0]
        if len(di) < MIN_LEGS: continue
        si = di[np.argsort(-szn_conf[di])]
        month = pd.to_datetime(date).month
        if month not in monthly:
            monthly[month] = {"w": 0, "r": 0, "w3": 0, "l3": 0, "w5": 0, "l5": 0}

        day_wagered = 0; day_returned = 0

        # 3-leg parlay
        top3 = si[:3]
        pd3 = 1.0; aw3 = True
        for idx in top3:
            pd3 *= szn_dec[idx]
            if not szn_won[idx]: aw3 = False
        tw += BET_3LEG; day_wagered += BET_3LEG
        monthly[month]["w"] += BET_3LEG
        if aw3:
            pay3 = BET_3LEG * pd3; tr += pay3; day_returned += pay3
            w3 += 1; monthly[month]["w3"] += 1; monthly[month]["r"] += pay3
        else:
            l3 += 1; monthly[month]["l3"] += 1

        # 5-leg parlay (if enough picks)
        if len(si) >= 5:
            top5 = si[:5]
            pd5 = 1.0; aw5 = True
            for idx in top5:
                pd5 *= szn_dec[idx]
                if not szn_won[idx]: aw5 = False
            tw += BET_5LEG; day_wagered += BET_5LEG
            monthly[month]["w"] += BET_5LEG
            if aw5:
                pay5 = BET_5LEG * pd5; tr += pay5; day_returned += pay5
                w5 += 1; monthly[month]["w5"] += 1; monthly[month]["r"] += pay5
            else:
                l5 += 1; monthly[month]["l5"] += 1
        elif len(si) >= 3:
            # Flex: only had 3-4 picks, already placed 3-leg
            pass

        daily_data.append({
            "date": date, "month": month,
            "wagered": day_wagered, "returned": day_returned,
            "profit": day_returned - day_wagered,
        })

    n_days = len(daily_data)
    profit = tr - tw
    roi = (profit / tw * 100) if tw > 0 else 0
    cumulative += profit
    hit3 = w3 / max(w3 + l3, 1) * 100
    hit5 = w5 / max(w5 + l5, 1) * 100
    tag = "✅" if profit > 0 else "❌"

    print(f"  {season:>7d} {n_days:>5d} {w3:>5d} {l3:>5d} {hit3:>6.1f}% "
          f"{w5:>5d} {l5:>5d} {hit5:>6.1f}% "
          f"${tw:>8,d} ${profit:>+8,.0f} {roi:>+6.1f}% {tag}")

    all_season_data.append({
        "season": season, "days": n_days, "wagered": tw, "profit": profit, "roi": roi,
        "w3": w3, "l3": l3, "w5": w5, "l5": l5, "monthly": monthly, "daily": daily_data,
    })

# Total
tw_all = sum(s["wagered"] for s in all_season_data)
tp_all = sum(s["profit"] for s in all_season_data)
w3_all = sum(s["w3"] for s in all_season_data)
l3_all = sum(s["l3"] for s in all_season_data)
w5_all = sum(s["w5"] for s in all_season_data)
l5_all = sum(s["l5"] for s in all_season_data)
days_all = sum(s["days"] for s in all_season_data)

print(f"  {'TOTAL':>7s} {days_all:>5d} {w3_all:>5d} {l3_all:>5d} "
      f"{w3_all/(w3_all+l3_all)*100:>6.1f}% "
      f"{w5_all:>5d} {l5_all:>5d} {w5_all/(w5_all+l5_all)*100:>6.1f}% "
      f"${tw_all:>8,d} ${tp_all:>+8,.0f} {tp_all/tw_all*100:>+6.1f}%")

# ══════════════════════════════════════════════════════════
# MONTHLY DETAIL PER SEASON
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  MONTHLY BREAKDOWN")
print(f"{'='*70}")

for sd in all_season_data:
    print(f"\n  --- {sd['season']} (Profit ${sd['profit']:+,.0f}, ROI {sd['roi']:+.1f}%) ---")
    print(f"  {'Month':<6s} {'Days':>5s} {'3L W/L':>8s} {'5L W/L':>8s} {'Wagered':>9s} {'Profit':>9s}")
    print("  " + "-" * 50)

    for m_num in [11, 12, 1, 2, 3, 4]:
        m = sd["monthly"].get(m_num)
        if not m or m["w"] == 0: continue
        mp = m["r"] - m["w"]
        m_name = month_names.get(m_num, str(m_num))
        print(f"  {m_name:<6s} {'':>5s} {m['w3']:>3d}/{m['l3']:<3d} {m['w5']:>3d}/{m['l5']:<3d} "
              f"${m['w']:>8,d} ${mp:>+8,.0f}")

# ══════════════════════════════════════════════════════════
# MONTHLY HEATMAP
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  MONTHLY HEATMAP (profit by month × season)")
print(f"{'='*70}")

print(f"\n  {'Season':>7s}", end="")
for m in [11,12,1,2,3,4]: print(f" {month_names[m]:>8s}", end="")
print(f" {'TOTAL':>9s}")
print("  " + "-" * 70)

gm = {}
for sd in all_season_data:
    print(f"  {sd['season']:>7d}", end="")
    for m in [11,12,1,2,3,4]:
        md = sd["monthly"].get(m)
        if md and md["w"] > 0:
            mp = md["r"] - md["w"]; gm[m] = gm.get(m, 0) + mp
            print(f" ${mp:>+7,.0f}", end="")
        else:
            print(f" {'—':>8s}", end="")
    print(f" ${sd['profit']:>+8,.0f}")

print(f"  {'TOTAL':>7s}", end=""); gt = 0
for m in [11,12,1,2,3,4]:
    mp = gm.get(m, 0); gt += mp; print(f" ${mp:>+7,.0f}", end="")
print(f" ${gt:>+8,.0f}")

# ══════════════════════════════════════════════════════════
# CUMULATIVE P&L + DAILY STATS
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  CUMULATIVE P&L + DAILY STATS")
print(f"{'='*70}")

cum = 0
print(f"\n  {'Season':>7s} {'Profit':>9s} {'Cumul':>9s} {'Green Days':>11s} {'Red Days':>9s} "
      f"{'Best Day':>9s} {'Worst Day':>10s}")
print("  " + "-" * 70)

for sd in all_season_data:
    cum += sd["profit"]
    dd = sd["daily"]
    green = sum(1 for d in dd if d["profit"] > 0)
    red = sum(1 for d in dd if d["profit"] < 0)
    push = sum(1 for d in dd if d["profit"] == 0)
    best = max(d["profit"] for d in dd) if dd else 0
    worst = min(d["profit"] for d in dd) if dd else 0
    tag = "✅" if cum > 0 else "❌"
    print(f"  {sd['season']:>7d} ${sd['profit']:>+8,.0f} ${cum:>+8,.0f} "
          f"{green:>6d}g/{red}r "
          f"${best:>+8,.0f} ${worst:>+9,.0f} {tag}")

print(f"\n  Total profit:    ${cum:>+,.0f}")
print(f"  Avg per season:  ${cum/len(all_season_data):>+,.0f}")
print(f"  Total wagered:   ${tw_all:>,.0f}")
print(f"  Overall ROI:     {cum/tw_all*100:>+.1f}%")
print(f"  Daily risk:      ${BET_3LEG + BET_5LEG}/day")
print(f"  Total days:      {days_all}")

print(f"\n  Done.")
