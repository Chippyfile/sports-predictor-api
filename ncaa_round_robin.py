#!/usr/bin/env python3
"""
ncaa_round_robin.py — Round Robin vs Single Parlay Comparison
=============================================================
Compares:
  A) Single top-N parlay (1 bet/day)
  B) Round robin: all K-leg combos from top-N picks (many bets/day)

Example: Top 8 picks, 4-leg round robin = C(8,4) = 70 parlays/day
  If 7/8 hit: 35 of 70 cash. If 6/8: 15 of 70 cash. If 5/8: 5 of 70 cash.
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from itertools import combinations
from math import comb
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


# ── Load data (same as other scripts) ──
print("=" * 70)
print("  ROUND ROBIN vs SINGLE PARLAY — 2026 Season")
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

print(f"  {len(X)} total games")

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

# Compute ML odds
home_ml_arr = np.zeros(len(pred_2026))
away_ml_arr = np.zeros(len(pred_2026))
for i, sp in enumerate(spread_2026):
    hml, aml = spread_to_ml(sp)
    home_ml_arr[i] = hml
    away_ml_arr[i] = aml

pick_ml = np.where(pred_home_win, home_ml_arr, away_ml_arr)
pick_won = (pred_home_win == actual_home_win) & not_tie
pick_dec = np.array([ml_to_decimal(ml) for ml in pick_ml])

print(f"  2026: {mask_2026.sum()} games with predictions + spreads")

# ══════════════════════════════════════════════════════════
# Group by date, get top N picks per day
# ══════════════════════════════════════════════════════════

unique_dates = sorted(set(date_strs))

# Build daily pick pools
daily_picks = {}
for date in unique_dates:
    day_mask = (date_strs == date) & not_tie & (prob_pick >= 0.65)
    day_indices = np.where(day_mask)[0]
    if len(day_indices) < 2:
        continue
    # Sort by confidence
    sorted_idx = day_indices[np.argsort(-confidence[day_indices])]
    daily_picks[date] = sorted_idx

print(f"  Days with ≥2 qualifying picks: {len(daily_picks)}")
avg_picks = np.mean([len(v) for v in daily_picks.values()])
print(f"  Average qualifying picks per day: {avg_picks:.1f}")


# ══════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════

def run_strategy(pool_size, leg_count, wager_per_combo, label):
    """Run a round-robin or single-parlay strategy."""
    total_wagered = 0
    total_returned = 0
    total_combos = 0
    total_wins = 0
    n_days = 0
    best_day_profit = -99999
    worst_day_profit = 99999
    daily_profits = []
    monthly = {}

    for date, indices in daily_picks.items():
        if len(indices) < pool_size:
            continue

        top_n = indices[:pool_size]
        n_days += 1
        month = pd.to_datetime(date).month
        if month not in monthly:
            monthly[month] = {"wagered": 0, "returned": 0, "days": 0, "combos": 0, "wins": 0}

        # All K-leg combos from top N
        combos = list(combinations(range(len(top_n)), leg_count))
        day_wagered = len(combos) * wager_per_combo
        day_returned = 0

        for combo in combos:
            total_combos += 1
            monthly[month]["combos"] += 1

            # Calculate parlay odds and check if all legs won
            parlay_dec = 1.0
            all_won = True
            for leg_idx in combo:
                actual_idx = top_n[leg_idx]
                parlay_dec *= pick_dec[actual_idx]
                if not pick_won[actual_idx]:
                    all_won = False

            if all_won:
                payout = wager_per_combo * parlay_dec
                day_returned += payout
                total_wins += 1
                monthly[month]["wins"] += 1

        total_wagered += day_wagered
        total_returned += day_returned
        day_profit = day_returned - day_wagered
        daily_profits.append(day_profit)
        best_day_profit = max(best_day_profit, day_profit)
        worst_day_profit = min(worst_day_profit, day_profit)

        monthly[month]["wagered"] += day_wagered
        monthly[month]["returned"] += day_returned
        monthly[month]["days"] += 1

    profit = total_returned - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    hit_rate = (total_wins / total_combos * 100) if total_combos > 0 else 0
    green_days = sum(1 for p in daily_profits if p > 0)
    red_days = sum(1 for p in daily_profits if p < 0)
    push_days = sum(1 for p in daily_profits if p == 0)

    return {
        "label": label,
        "pool": pool_size,
        "legs": leg_count,
        "days": n_days,
        "combos_total": total_combos,
        "combos_per_day": total_combos / max(n_days, 1),
        "wins": total_wins,
        "hit_rate": hit_rate,
        "wagered": total_wagered,
        "returned": total_returned,
        "profit": profit,
        "roi": roi,
        "best_day": best_day_profit,
        "worst_day": worst_day_profit,
        "green_days": green_days,
        "red_days": red_days,
        "wager_per_combo": wager_per_combo,
        "monthly": monthly,
    }


month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

# ── Run all strategies ──
# Budget: ~$20-40/day total
strategies = [
    # Single parlays (baseline)
    (4, 4, 20, "Single 4-leg ($20)"),
    (5, 5, 20, "Single 5-leg ($20)"),
    (8, 8, 20, "Single 8-leg ($20)"),
    # Round robin: top 6, all 3-leg combos = C(6,3)=20 combos
    (6, 3, 1, "RR top6 → 3-leg ($1×20)"),
    (6, 4, 2, "RR top6 → 4-leg ($2×15)"),
    # Round robin: top 8, various leg counts
    (8, 3, 0.50, "RR top8 → 3-leg ($0.50×56)"),
    (8, 4, 0.30, "RR top8 → 4-leg ($0.30×70)"),
    (8, 5, 0.40, "RR top8 → 5-leg ($0.40×56)"),
    # Round robin: top 10
    (10, 4, 0.10, "RR top10 → 4-leg ($0.10×210)"),
    (10, 5, 0.10, "RR top10 → 5-leg ($0.10×252)"),
    # Bigger round robins
    (8, 3, 1, "RR top8 → 3-leg ($1×56)"),
    (8, 4, 1, "RR top8 → 4-leg ($1×70)"),
]

print(f"\n{'='*70}")
print(f"  STRATEGY COMPARISON — 2026 Season")
print(f"{'='*70}")

print(f"\n  {'Strategy':<28s} {'Days':>5s} {'Bets':>6s} {'Hit%':>6s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s} {'Green':>6s} {'Red':>5s}")
print("  " + "-" * 90)

results = []
for pool, legs, wager, label in strategies:
    r = run_strategy(pool, legs, wager, label)
    results.append(r)
    tag = "✅" if r["profit"] > 0 else "❌"
    print(f"  {label:<28s} {r['days']:>5d} {r['combos_total']:>6d} {r['hit_rate']:>5.1f}% "
          f"${r['wagered']:>8,.0f} ${r['profit']:>+8,.0f} {r['roi']:>+6.1f}% "
          f"{r['green_days']:>5d}d {r['red_days']:>4d}d {tag}")

# ── Detailed monthly for top strategies ──
print(f"\n{'='*70}")
print(f"  MONTHLY DETAIL — Top Strategies")
print(f"{'='*70}")

# Show monthly for the most interesting strategies
for r in results:
    if abs(r["roi"]) < 5 and r["profit"] < 100:
        continue  # skip boring ones

    print(f"\n  --- {r['label']} (ROI {r['roi']:+.1f}%, Profit ${r['profit']:+,.0f}) ---")
    print(f"  {'Month':<8s} {'Days':>5s} {'Bets':>6s} {'Wins':>5s} {'Hit%':>6s} {'Wagered':>9s} {'Returned':>10s} {'Profit':>9s}")
    print("  " + "-" * 60)

    for m_num in [11, 12, 1, 2, 3, 4]:
        m = r["monthly"].get(m_num)
        if not m or m["days"] == 0:
            continue
        m_profit = m["returned"] - m["wagered"]
        m_hit = m["wins"] / max(m["combos"], 1) * 100
        m_name = month_names.get(m_num, str(m_num))
        print(f"  {m_name:<8s} {m['days']:>5d} {m['combos']:>6d} {m['wins']:>5d} {m_hit:>5.1f}% "
              f"${m['wagered']:>8,.0f} ${m['returned']:>9,.0f} ${m_profit:>+8,.0f}")

# ── Head-to-head: Single vs Round Robin ──
print(f"\n{'='*70}")
print(f"  HEAD-TO-HEAD: Single Parlay vs Round Robin")
print(f"  (Same ~$20/day budget)")
print(f"{'='*70}")

comparisons = [
    ("Single 4-leg ($20)", "RR top6 → 3-leg ($1×20)"),
    ("Single 5-leg ($20)", "RR top8 → 5-leg ($0.40×56)"),
    ("Single 8-leg ($20)", "RR top8 → 4-leg ($0.30×70)"),
]

for single_label, rr_label in comparisons:
    s = next((r for r in results if r["label"] == single_label), None)
    rr = next((r for r in results if r["label"] == rr_label), None)
    if not s or not rr:
        continue

    print(f"\n  {'Metric':<20s} {single_label:>28s} {rr_label:>28s}")
    print("  " + "-" * 80)
    print(f"  {'Days':<20s} {s['days']:>28d} {rr['days']:>28d}")
    print(f"  {'Bets/day':<20s} {1:>28d} {rr['combos_per_day']:>28.0f}")
    print(f"  {'Total bets':<20s} {s['combos_total']:>28d} {rr['combos_total']:>28d}")
    print(f"  {'Hit rate':<20s} {s['hit_rate']:>27.1f}% {rr['hit_rate']:>27.1f}%")
    print(f"  {'Total wagered':<20s} ${s['wagered']:>27,.0f} ${rr['wagered']:>27,.0f}")
    print(f"  {'Profit':<20s} ${s['profit']:>+27,.0f} ${rr['profit']:>+27,.0f}")
    print(f"  {'ROI':<20s} {s['roi']:>+27.1f}% {rr['roi']:>+27.1f}%")
    print(f"  {'Green days':<20s} {s['green_days']:>28d} {rr['green_days']:>28d}")
    print(f"  {'Red days':<20s} {s['red_days']:>28d} {rr['red_days']:>28d}")
    print(f"  {'Best day':<20s} ${s['best_day']:>+27,.0f} ${rr['best_day']:>+27,.0f}")
    print(f"  {'Worst day':<20s} ${s['worst_day']:>+27,.0f} ${rr['worst_day']:>+27,.0f}")

    winner = single_label if s["profit"] > rr["profit"] else rr_label
    print(f"\n  WINNER: {winner}")

print(f"\n  Done.")
