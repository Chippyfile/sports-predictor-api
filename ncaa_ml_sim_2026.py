#!/usr/bin/env python3
"""
ncaa_ml_sim_2026.py — 2026 Season ML Betting Simulation
========================================================
Uses walk-forward predictions + actual/estimated ML odds.
Shows: flat bet ROI, favorite vs underdog, monthly, parlays.
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
from collections import defaultdict

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
    """Convert point spread to approximate moneyline.
    NCAA basketball empirical conversion.
    spread < 0 means home is favored.
    Returns (home_ml, away_ml)."""
    s = abs(spread)
    if s < 0.5:
        return (-110, -110)
    # Empirical NCAA spread-to-ML approximation
    # Based on standard vig-adjusted conversion tables
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

    if spread < 0:  # home favored
        return (fav_ml, dog_ml)
    else:  # away favored
        return (dog_ml, fav_ml)


def ml_payout(ml, bet=100):
    """Calculate payout for a given ML and bet amount.
    Returns total return (stake + profit) on win, 0 on loss."""
    if ml > 0:
        return bet + bet * (ml / 100)
    else:
        return bet + bet * (100 / abs(ml))


def ml_risk(ml, target_win=100):
    """How much to risk to win $100 at given ML."""
    if ml > 0:
        return target_win * (100 / ml)
    else:
        return target_win * (abs(ml) / 100)


# ── Load data ──
print("=" * 70)
print("  2026 NCAA ML BETTING SIMULATION")
print("  Walk-forward predictions + spread-implied ML odds")
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

print("\n  Heuristic backfill + features...")
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

print(f"  {len(X)} total games, {(seasons == 2026).sum()} in 2026")

# ── Walk-forward ──
print(f"\n  {N_FOLDS}-fold walk-forward...")
t0 = time.time()
oof = walk_forward(X_s, y, N_FOLDS)
print(f"  Done in {time.time()-t0:.0f}s")

# ── Filter to 2026 with valid predictions and spreads ──
mask_2026 = (seasons == 2026) & ~np.isnan(oof) & (np.abs(spreads) > 0.1)
print(f"  2026 games with predictions + spreads: {mask_2026.sum()}")

pred_2026 = oof[mask_2026]
actual_2026 = y[mask_2026]
spread_2026 = spreads[mask_2026]
dates_2026 = pd.to_datetime(pd.Series(dates[mask_2026]))

# Compute ML odds from spread
home_ml_arr = np.zeros(len(pred_2026))
away_ml_arr = np.zeros(len(pred_2026))
for i, sp in enumerate(spread_2026):
    hml, aml = spread_to_ml(sp)
    home_ml_arr[i] = hml
    away_ml_arr[i] = aml

# Check for actual ML odds in data
actual_ml_cols = [c for c in df.columns if 'ml' in c.lower() and ('home' in c.lower() or 'away' in c.lower())]
print(f"  ML odds columns found: {actual_ml_cols}")

# Use actual odds if available, otherwise spread-implied
for col in ['dk_ml_home', 'odds_api_ml_home', 'espn_ml_home']:
    if col in df.columns:
        real_ml = pd.to_numeric(df.loc[mask_2026, col], errors="coerce").fillna(0).values
        has_real = np.abs(real_ml) > 10
        if has_real.sum() > 0:
            print(f"  Using {col}: {has_real.sum()}/{len(real_ml)} games have real ML odds")
            home_ml_arr[has_real] = real_ml[has_real]
            # Derive away from home
            for j in np.where(has_real)[0]:
                if home_ml_arr[j] < 0:
                    away_ml_arr[j] = int(abs(home_ml_arr[j]) * 0.75)  # approximate
                else:
                    away_ml_arr[j] = int(-home_ml_arr[j] * 1.3)
            break

# Determine picks
pred_home_win = pred_2026 > 0
actual_home_win = actual_2026 > 0
confidence = np.abs(pred_2026)
prob_home = 1.0 / (1.0 + np.exp(-pred_2026 / 10.0))
prob_pick = np.maximum(prob_home, 1 - prob_home)

# For each pick: what ML would we bet, and did it win?
pick_ml = np.where(pred_home_win, home_ml_arr, away_ml_arr)
pick_won = (pred_home_win == actual_home_win) & (actual_2026 != 0)
is_favorite = pick_ml < 0
is_underdog = pick_ml > 0

months = dates_2026.dt.month.values
month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}


def simulate_flat_bets(mask, label, bet_size=100):
    """Simulate flat $100-to-win bets on the selected picks."""
    n = mask.sum()
    if n == 0:
        return 0, 0, 0, 0
    
    total_risked = 0
    total_returned = 0
    wins = 0
    
    for i in np.where(mask)[0]:
        ml = pick_ml[i]
        risk = ml_risk(ml, target_win=bet_size)
        total_risked += risk
        if pick_won[i]:
            total_returned += risk + bet_size  # risk back + profit
            wins += 1
    
    profit = total_returned - total_risked
    roi = (profit / total_risked * 100) if total_risked > 0 else 0
    return wins, n, profit, roi


# ══════════════════════════════════════════════════════════
# 1. FLAT BET RESULTS BY CONFIDENCE
# ══════════════════════════════════════════════════════════

not_tie = actual_2026 != 0
base_mask = not_tie

print(f"\n{'='*70}")
print(f"  1. FLAT BET RESULTS ($100-to-win per pick)")
print(f"     2026 Season | Spread-implied ML odds")
print(f"{'='*70}")

print(f"\n  {'Filter':<25s} {'Picks':>6s} {'W':>5s} {'L':>5s} {'Acc':>6s} {'Risked':>10s} {'Profit':>10s} {'ROI':>7s}")
print("  " + "-" * 75)

for label, conf_min in [("All picks", 0), ("≥55% conf", 0.55), ("≥60% conf", 0.60),
                          ("≥65% conf", 0.65), ("≥70% conf", 0.70), ("≥75% conf", 0.75), ("≥80% conf", 0.80)]:
    m = base_mask & (prob_pick >= conf_min)
    w, n, profit, roi = simulate_flat_bets(m, label)
    l = n - w
    acc = w / n if n > 0 else 0
    risked = sum(ml_risk(pick_ml[i], 100) for i in np.where(m)[0])
    tag = "✅" if roi > 0 else "❌"
    print(f"  {label:<25s} {n:>6d} {w:>5d} {l:>5d} {acc:>5.1%} ${risked:>9,.0f} ${profit:>+9,.0f} {roi:>+6.1f}% {tag}")


# ══════════════════════════════════════════════════════════
# 2. FAVORITE vs UNDERDOG
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  2. FAVORITE vs UNDERDOG BREAKDOWN")
print(f"{'='*70}")

print(f"\n  {'Type':<25s} {'Picks':>6s} {'W':>5s} {'Acc':>6s} {'Avg ML':>8s} {'Profit':>10s} {'ROI':>7s}")
print("  " + "-" * 65)

for label, filt in [("All favorites", base_mask & is_favorite),
                     ("Favorites ≥65% conf", base_mask & is_favorite & (prob_pick >= 0.65)),
                     ("Favorites ≥70% conf", base_mask & is_favorite & (prob_pick >= 0.70)),
                     ("All underdogs", base_mask & is_underdog),
                     ("Underdogs ≥55% conf", base_mask & is_underdog & (prob_pick >= 0.55)),
                     ("Underdogs ≥60% conf", base_mask & is_underdog & (prob_pick >= 0.60))]:
    w, n, profit, roi = simulate_flat_bets(filt, label)
    if n == 0: continue
    acc = w / n
    avg_ml = np.mean(pick_ml[filt]) if filt.sum() > 0 else 0
    tag = "✅" if roi > 0 else "❌"
    print(f"  {label:<25s} {n:>6d} {w:>5d} {acc:>5.1%} {avg_ml:>+7.0f} ${profit:>+9,.0f} {roi:>+6.1f}% {tag}")


# ══════════════════════════════════════════════════════════
# 3. MONTHLY BREAKDOWN
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  3. MONTHLY BREAKDOWN (≥65% confidence)")
print(f"{'='*70}")

print(f"\n  {'Month':<10s} {'Picks':>6s} {'W':>5s} {'Acc':>6s} {'Fav':>5s} {'Dog':>5s} {'Profit':>10s} {'ROI':>7s}")
print("  " + "-" * 55)

conf_mask = base_mask & (prob_pick >= 0.65)
for m_num in [11, 12, 1, 2, 3, 4]:
    m_mask = conf_mask & (months == m_num)
    w, n, profit, roi = simulate_flat_bets(m_mask, "")
    if n == 0: continue
    acc = w / n
    n_fav = (m_mask & is_favorite).sum()
    n_dog = (m_mask & is_underdog).sum()
    m_name = month_names.get(m_num, str(m_num))
    tag = "✅" if roi > 0 else "❌"
    print(f"  {m_name:<10s} {n:>6d} {w:>5d} {acc:>5.1%} {n_fav:>5d} {n_dog:>5d} ${profit:>+9,.0f} {roi:>+6.1f}% {tag}")

# Total
w_tot, n_tot, profit_tot, roi_tot = simulate_flat_bets(conf_mask, "")
print(f"  {'TOTAL':<10s} {n_tot:>6d} {w_tot:>5d} {w_tot/max(n_tot,1):>5.1%} "
      f"{(conf_mask & is_favorite).sum():>5d} {(conf_mask & is_underdog).sum():>5d} "
      f"${profit_tot:>+9,.0f} {roi_tot:>+6.1f}%")


# ══════════════════════════════════════════════════════════
# 4. PARLAY SIMULATION
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  4. PARLAY SIMULATION (top picks per day, $20 risk)")
print(f"     Minimum 65% confidence per leg")
print(f"{'='*70}")

# Group by date
date_strs = dates_2026.dt.strftime("%Y-%m-%d").values
unique_dates = sorted(set(date_strs))

for n_legs in [5, 6, 7, 8, 9, 10, 11, 12]:
    total_wagered = 0
    total_returned = 0
    n_parlays = 0
    n_wins = 0
    best_payout = 0
    
    daily_results = []
    
    for date in unique_dates:
        day_mask = (date_strs == date) & base_mask & (prob_pick >= 0.65)
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) < n_legs:
            continue
        
        # Sort by confidence (highest first)
        sorted_idx = day_indices[np.argsort(-confidence[day_indices])]
        top_picks = sorted_idx[:n_legs]
        
        # Calculate parlay odds
        parlay_decimal = 1.0
        all_won = True
        leg_details = []
        
        for idx in top_picks:
            ml = pick_ml[idx]
            if ml > 0:
                dec = ml / 100 + 1
            else:
                dec = 100 / abs(ml) + 1
            parlay_decimal *= dec
            
            if not pick_won[idx]:
                all_won = False
            
            leg_details.append({
                'ml': ml,
                'conf': prob_pick[idx],
                'won': pick_won[idx],
            })
        
        wager = 20
        total_wagered += wager
        n_parlays += 1
        
        if all_won:
            payout = wager * parlay_decimal
            total_returned += payout
            n_wins += 1
            if payout > best_payout:
                best_payout = payout
        
        daily_results.append({
            'date': date,
            'legs': n_legs,
            'won': all_won,
            'odds': parlay_decimal,
            'payout': wager * parlay_decimal if all_won else 0,
        })
    
    profit = total_returned - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    hit_rate = (n_wins / n_parlays * 100) if n_parlays > 0 else 0
    avg_odds = np.mean([r['odds'] for r in daily_results]) if daily_results else 0
    
    tag = "✅" if profit > 0 else "❌"
    print(f"\n  --- {n_legs}-LEG PARLAYS ---")
    print(f"  Total parlays:    {n_parlays}")
    print(f"  Wins:             {n_wins} ({hit_rate:.1f}%)")
    print(f"  Avg parlay odds:  {avg_odds:.1f}x (decimal)")
    print(f"  Total wagered:    ${total_wagered:,.0f}")
    print(f"  Total returned:   ${total_returned:,.0f}")
    print(f"  Profit:           ${profit:>+,.0f} {tag}")
    print(f"  ROI:              {roi:>+.1f}%")
    print(f"  Best single hit:  ${best_payout:,.0f}")
    
    # Monthly parlay breakdown
    print(f"\n  {'Month':<8s} {'Parlays':>8s} {'Wins':>6s} {'Hit%':>6s} {'Wagered':>9s} {'Returned':>10s} {'Profit':>10s}")
    print("  " + "-" * 60)
    for m_num in [11, 12, 1, 2, 3, 4]:
        m_results = [r for r in daily_results 
                     if pd.to_datetime(r['date']).month == m_num]
        if not m_results: continue
        m_parlays = len(m_results)
        m_wins = sum(1 for r in m_results if r['won'])
        m_wagered = m_parlays * 20
        m_returned = sum(r['payout'] for r in m_results)
        m_profit = m_returned - m_wagered
        m_name = month_names.get(m_num, str(m_num))
        print(f"  {m_name:<8s} {m_parlays:>8d} {m_wins:>6d} {m_wins/max(m_parlays,1)*100:>5.1f}% "
              f"${m_wagered:>8,d} ${m_returned:>9,.0f} ${m_profit:>+9,.0f}")

print(f"\n  Done.")
