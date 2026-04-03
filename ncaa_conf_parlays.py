#!/usr/bin/env python3
"""
ncaa_conf_parlays.py — Confidence-gated parlays
================================================
Only includes legs above a confidence threshold.
Tests every combo of threshold × leg count.
Days with fewer qualifying picks than legs are skipped.
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


def spread_to_ml(spread):
    s = abs(spread)
    if s < 0.5: return (-110, -110)
    if s <= 1:    fav_ml = -120; dog_ml = 100
    elif s <= 2:  fav_ml = -140; dog_ml = 120
    elif s <= 3:  fav_ml = -160; dog_ml = 135
    elif s <= 4:  fav_ml = -185; dog_ml = 155
    elif s <= 5:  fav_ml = -210; dog_ml = 175
    elif s <= 6:  fav_ml = -245; dog_ml = 200
    elif s <= 7:  fav_ml = -280; dog_ml = 230
    elif s <= 8:  fav_ml = -320; dog_ml = 260
    elif s <= 9:  fav_ml = -370; dog_ml = 295
    elif s <= 10: fav_ml = -420; dog_ml = 330
    elif s <= 12: fav_ml = -550; dog_ml = 410
    elif s <= 14: fav_ml = -700; dog_ml = 500
    elif s <= 16: fav_ml = -900; dog_ml = 600
    elif s <= 18: fav_ml = -1200; dog_ml = 750
    elif s <= 20: fav_ml = -1500; dog_ml = 900
    else:         fav_ml = -2000; dog_ml = 1200
    if spread < 0: return (fav_ml, dog_ml)
    else: return (dog_ml, fav_ml)


def ml_to_decimal(ml):
    if ml > 0: return ml / 100 + 1
    else: return 100 / abs(ml) + 1


# ── Load data ──
print("=" * 70)
print("  CONFIDENCE-GATED PARLAYS — 2026 Season")
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

for col in ["actual_home_score","actual_away_score","season",
            "home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else
    0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

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

print(f"\n  {N_FOLDS}-fold walk-forward...")
t0 = time.time()
oof = walk_forward(X_s, y, N_FOLDS)
print(f"  Done in {time.time()-t0:.0f}s")

# ── Filter to 2026 ──
mask_2026 = (seasons == 2026) & ~np.isnan(oof) & (np.abs(spreads) > 0.1)
pred_2026 = oof[mask_2026]
actual_2026 = y[mask_2026]
spread_2026 = spreads[mask_2026]
dates_2026 = pd.to_datetime(pd.Series(dates[mask_2026]))
date_strs = dates_2026.dt.strftime("%Y-%m-%d").values

pred_home_win = pred_2026 > 0
actual_home_win = actual_2026 > 0
not_tie = actual_2026 != 0
confidence = np.abs(pred_2026)
prob_home = 1.0 / (1.0 + np.exp(-pred_2026 / 10.0))
prob_pick = np.maximum(prob_home, 1 - prob_home)

home_ml_arr = np.zeros(len(pred_2026))
away_ml_arr = np.zeros(len(pred_2026))
for i, sp in enumerate(spread_2026):
    hml, aml = spread_to_ml(sp)
    home_ml_arr[i] = hml
    away_ml_arr[i] = aml

pick_ml = np.where(pred_home_win, home_ml_arr, away_ml_arr)
pick_won = (pred_home_win == actual_home_win) & not_tie
pick_dec = np.array([ml_to_decimal(ml) for ml in pick_ml])
months = dates_2026.dt.month.values

unique_dates = sorted(set(date_strs))
month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

print(f"  2026: {mask_2026.sum()} games")

# ── Show daily pick availability at each confidence level ──
print(f"\n{'='*70}")
print(f"  DAILY PICK AVAILABILITY BY CONFIDENCE")
print(f"{'='*70}")

for conf in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    daily_counts = []
    for date in unique_dates:
        day_mask = (date_strs == date) & not_tie & (prob_pick >= conf)
        daily_counts.append(day_mask.sum())
    dc = np.array(daily_counts)
    days_3plus = (dc >= 3).sum()
    days_4plus = (dc >= 4).sum()
    days_5plus = (dc >= 5).sum()
    print(f"  ≥{conf:.0%}: avg {dc.mean():.1f}/day, "
          f"≥3 picks: {days_3plus}/{len(unique_dates)} days ({100*days_3plus/len(unique_dates):.0f}%), "
          f"≥4: {days_4plus} days, ≥5: {days_5plus} days")


# ══════════════════════════════════════════════════════════
# MAIN SIMULATION — every threshold × leg count combo
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PARLAY RESULTS — Top picks per day (≥ threshold), $20/parlay")
print(f"{'='*70}")

print(f"\n  {'Conf':<6s} {'Legs':>4s} {'Days':>5s} {'Hit%':>6s} {'AvgOdds':>8s} {'Wagered':>9s} "
      f"{'Profit':>9s} {'ROI':>7s} {'Green':>6s} {'Red':>5s} {'AvgLegs':>8s}")
print("  " + "-" * 85)

all_results = []

for conf_thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    for n_legs in [3, 4, 5]:
        total_wagered = 0
        total_returned = 0
        n_parlays = 0
        n_wins = 0
        green_days = 0
        red_days = 0
        monthly = {}
        actual_legs_used = []

        for date in unique_dates:
            day_mask = (date_strs == date) & not_tie & (prob_pick >= conf_thresh)
            day_indices = np.where(day_mask)[0]

            if len(day_indices) < n_legs:
                continue

            # Sort by confidence, take top N
            sorted_idx = day_indices[np.argsort(-confidence[day_indices])]
            top_picks = sorted_idx[:n_legs]
            actual_legs_used.append(len(top_picks))

            month = pd.to_datetime(date).month
            if month not in monthly:
                monthly[month] = {"wagered": 0, "returned": 0, "wins": 0, "parlays": 0}

            # Calculate parlay
            parlay_dec = 1.0
            all_won = True
            for idx in top_picks:
                parlay_dec *= pick_dec[idx]
                if not pick_won[idx]:
                    all_won = False

            wager = 20
            total_wagered += wager
            n_parlays += 1
            monthly[month]["wagered"] += wager
            monthly[month]["parlays"] += 1

            if all_won:
                payout = wager * parlay_dec
                total_returned += payout
                n_wins += 1
                monthly[month]["returned"] += payout
                monthly[month]["wins"] += 1
                green_days += 1
            else:
                red_days += 1

        if n_parlays == 0:
            continue

        profit = total_returned - total_wagered
        roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
        hit_rate = (n_wins / n_parlays * 100) if n_parlays > 0 else 0
        avg_odds = total_returned / max(n_wins, 1) / 20 if n_wins > 0 else 0
        avg_legs = np.mean(actual_legs_used) if actual_legs_used else 0

        tag = "✅" if profit > 0 else "❌"
        print(f"  ≥{conf_thresh:.0%}  {n_legs:>4d} {n_parlays:>5d} {hit_rate:>5.1f}% "
              f"{avg_odds:>7.2f}x ${total_wagered:>8,d} "
              f"${profit:>+8,.0f} {roi:>+6.1f}% "
              f"{green_days:>5d}d {red_days:>4d}d {avg_legs:>7.1f} {tag}")

        all_results.append({
            "conf": conf_thresh, "legs": n_legs, "days": n_parlays,
            "hit": hit_rate, "profit": profit, "roi": roi,
            "wagered": total_wagered, "green": green_days, "red": red_days,
            "monthly": monthly,
        })

# ── Rank by ROI ──
print(f"\n{'='*70}")
print(f"  RANKING BY ROI (top 10)")
print(f"{'='*70}")

ranked = sorted(all_results, key=lambda x: x["roi"], reverse=True)
print(f"\n  {'Rank':>4s} {'Conf':<6s} {'Legs':>4s} {'Days':>5s} {'Hit%':>6s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 45)
for i, r in enumerate(ranked[:10]):
    print(f"  {i+1:>4d} ≥{r['conf']:.0%}  {r['legs']:>4d} {r['days']:>5d} {r['hit']:>5.1f}% "
          f"${r['profit']:>+8,.0f} {r['roi']:>+6.1f}%")

# ── Rank by profit ──
print(f"\n{'='*70}")
print(f"  RANKING BY PROFIT (top 10)")
print(f"{'='*70}")

ranked_p = sorted(all_results, key=lambda x: x["profit"], reverse=True)
print(f"\n  {'Rank':>4s} {'Conf':<6s} {'Legs':>4s} {'Days':>5s} {'Hit%':>6s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 45)
for i, r in enumerate(ranked_p[:10]):
    print(f"  {i+1:>4d} ≥{r['conf']:.0%}  {r['legs']:>4d} {r['days']:>5d} {r['hit']:>5.1f}% "
          f"${r['profit']:>+8,.0f} {r['roi']:>+6.1f}%")

# ── Monthly detail for top 3 ──
print(f"\n{'='*70}")
print(f"  MONTHLY DETAIL — Top 3 by Profit")
print(f"{'='*70}")

for r in ranked_p[:3]:
    print(f"\n  --- ≥{r['conf']:.0%} conf, {r['legs']}-leg parlays "
          f"(Profit ${r['profit']:+,.0f}, ROI {r['roi']:+.1f}%) ---")
    print(f"  {'Month':<8s} {'Parlays':>8s} {'Wins':>5s} {'Hit%':>6s} {'Wagered':>9s} {'Profit':>9s}")
    print("  " + "-" * 50)

    for m_num in [11, 12, 1, 2, 3, 4]:
        m = r["monthly"].get(m_num)
        if not m or m["parlays"] == 0:
            continue
        m_profit = m["returned"] - m["wagered"]
        m_hit = m["wins"] / m["parlays"] * 100
        m_name = month_names.get(m_num, str(m_num))
        print(f"  {m_name:<8s} {m['parlays']:>8d} {m['wins']:>5d} {m_hit:>5.1f}% "
              f"${m['wagered']:>8,d} ${m_profit:>+8,.0f}")

# ── Stacking: run BOTH a 3-leg and 4-leg parlay daily ──
print(f"\n{'='*70}")
print(f"  DYNAMIC LEG COUNT — Flex 3-5 legs based on daily availability")
print(f"  Cap at 5 legs, floor at 3, skip day if < 3 qualifying picks")
print(f"{'='*70}")

print(f"\n  {'Strategy':<30s} {'Days':>5s} {'Hit%':>6s} {'Avg Legs':>9s} {'Wagered':>9s} "
      f"{'Profit':>9s} {'ROI':>7s} {'Green':>6s} {'Red':>5s}")
print("  " + "-" * 90)

flex_results = []

for conf_thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    for max_legs in [5, 4, 3]:
        min_legs = 3
        total_wagered = 0
        total_returned = 0
        n_parlays = 0
        n_wins = 0
        green_days = 0
        red_days = 0
        legs_used_list = []
        monthly = {}

        for date in unique_dates:
            day_mask = (date_strs == date) & not_tie & (prob_pick >= conf_thresh)
            day_indices = np.where(day_mask)[0]

            if len(day_indices) < min_legs:
                continue

            # Sort by confidence, take up to max_legs
            sorted_idx = day_indices[np.argsort(-confidence[day_indices])]
            actual_legs = min(len(sorted_idx), max_legs)
            top_picks = sorted_idx[:actual_legs]
            legs_used_list.append(actual_legs)

            month = pd.to_datetime(date).month
            if month not in monthly:
                monthly[month] = {"wagered": 0, "returned": 0, "wins": 0, "parlays": 0,
                                  "legs_3": 0, "legs_4": 0, "legs_5": 0}

            parlay_dec = 1.0
            all_won = True
            for idx in top_picks:
                parlay_dec *= pick_dec[idx]
                if not pick_won[idx]:
                    all_won = False

            wager = 20
            total_wagered += wager
            n_parlays += 1
            monthly[month]["wagered"] += wager
            monthly[month]["parlays"] += 1
            monthly[month][f"legs_{actual_legs}"] = monthly[month].get(f"legs_{actual_legs}", 0) + 1

            if all_won:
                payout = wager * parlay_dec
                total_returned += payout
                n_wins += 1
                monthly[month]["returned"] += payout
                monthly[month]["wins"] += 1
                green_days += 1
            else:
                red_days += 1

        if n_parlays == 0:
            continue

        profit = total_returned - total_wagered
        roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
        hit_rate = (n_wins / n_parlays * 100) if n_parlays > 0 else 0
        avg_legs = np.mean(legs_used_list) if legs_used_list else 0

        # Count leg distribution
        l3 = sum(1 for l in legs_used_list if l == 3)
        l4 = sum(1 for l in legs_used_list if l == 4)
        l5 = sum(1 for l in legs_used_list if l == 5)

        label = f"≥{conf_thresh:.0%} flex→{max_legs}"
        tag = "✅" if profit > 0 else "❌"
        print(f"  {label:<30s} {n_parlays:>5d} {hit_rate:>5.1f}% {avg_legs:>8.1f} ${total_wagered:>8,d} "
              f"${profit:>+8,.0f} {roi:>+6.1f}% "
              f"{green_days:>5d}d {red_days:>4d}d {tag}  (3L:{l3} 4L:{l4} 5L:{l5})")

        flex_results.append({
            "label": label, "conf": conf_thresh, "max_legs": max_legs,
            "days": n_parlays, "hit": hit_rate, "profit": profit, "roi": roi,
            "wagered": total_wagered, "green": green_days, "red": red_days,
            "avg_legs": avg_legs, "monthly": monthly,
            "l3": l3, "l4": l4, "l5": l5,
        })

# ── Rank flex strategies ──
print(f"\n  --- FLEX RANKING BY PROFIT ---")
flex_ranked = sorted(flex_results, key=lambda x: x["profit"], reverse=True)
print(f"\n  {'Rank':>4s} {'Strategy':<25s} {'Days':>5s} {'Hit%':>6s} {'AvgL':>5s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 60)
for i, r in enumerate(flex_ranked[:10]):
    print(f"  {i+1:>4d} {r['label']:<25s} {r['days']:>5d} {r['hit']:>5.1f}% {r['avg_legs']:>4.1f} "
          f"${r['profit']:>+8,.0f} {r['roi']:>+6.1f}%")

# ── Monthly detail for top flex ──
print(f"\n  --- MONTHLY — Top Flex Strategy ---")
r = flex_ranked[0]
print(f"  {r['label']} (Profit ${r['profit']:+,.0f}, ROI {r['roi']:+.1f}%)")
print(f"  Leg distribution: 3-leg={r['l3']}, 4-leg={r['l4']}, 5-leg={r['l5']}")
print(f"\n  {'Month':<8s} {'Parlays':>8s} {'Wins':>5s} {'Hit%':>6s} {'Wagered':>9s} {'Profit':>9s}")
print("  " + "-" * 50)

for m_num in [11, 12, 1, 2, 3, 4]:
    m = r["monthly"].get(m_num)
    if not m or m["parlays"] == 0:
        continue
    m_profit = m["returned"] - m["wagered"]
    m_hit = m["wins"] / m["parlays"] * 100
    m_name = month_names.get(m_num, str(m_num))
    print(f"  {m_name:<8s} {m['parlays']:>8d} {m['wins']:>5d} {m_hit:>5.1f}% "
          f"${m['wagered']:>8,d} ${m_profit:>+8,.0f}")

# ── Fixed leg strategies for comparison ──
print(f"\n{'='*70}")
print(f"  STACKING STRATEGIES — Run multiple parlays per day")
print(f"{'='*70}")

stacks = [
    ("3+4 leg (≥65%)", [(0.65, 3), (0.65, 4)]),
    ("3+4+5 leg (≥65%)", [(0.65, 3), (0.65, 4), (0.65, 5)]),
    ("3+4 leg (≥70%)", [(0.70, 3), (0.70, 4)]),
    ("3+4+5 leg (≥70%)", [(0.70, 3), (0.70, 4), (0.70, 5)]),
    ("4-leg ≥65% + 3-leg ≥75%", [(0.65, 4), (0.75, 3)]),
    ("5-leg ≥65% + 3-leg ≥80%", [(0.65, 5), (0.80, 3)]),
]

print(f"\n  {'Stack':<30s} {'Bets':>5s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s}")
print("  " + "-" * 65)

for label, configs in stacks:
    total_wagered = 0
    total_returned = 0

    for date in unique_dates:
        for conf_thresh, n_legs in configs:
            day_mask = (date_strs == date) & not_tie & (prob_pick >= conf_thresh)
            day_indices = np.where(day_mask)[0]
            if len(day_indices) < n_legs:
                continue

            sorted_idx = day_indices[np.argsort(-confidence[day_indices])]
            top_picks = sorted_idx[:n_legs]

            parlay_dec = 1.0
            all_won = True
            for idx in top_picks:
                parlay_dec *= pick_dec[idx]
                if not pick_won[idx]:
                    all_won = False

            total_wagered += 20
            if all_won:
                total_returned += 20 * parlay_dec

    profit = total_returned - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    n_bets = int(total_wagered / 20)
    tag = "✅" if profit > 0 else "❌"
    print(f"  {label:<30s} {n_bets:>5d} ${total_wagered:>8,d} ${profit:>+8,.0f} {roi:>+6.1f}% {tag}")

print(f"\n  Done.")
