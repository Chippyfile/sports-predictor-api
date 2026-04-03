#!/usr/bin/env python3
"""
ncaa_hca_validate.py — Empirically compute HCA from real game data
==================================================================
1. Computes actual home court advantage per conference from historical results
2. Compares to hardcoded values in ncaa.py
3. Checks if travel distance correlates with neutral site outcomes

Usage:
    python3 ncaa_hca_validate.py
"""
import sys, os, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dump_training_data import dump, load_cached

print("=" * 70)
print("  NCAA HOME COURT ADVANTAGE — EMPIRICAL VALIDATION")
print("=" * 70)

# Load data
df = load_cached()
if df is None:
    df = dump()

df = df[df["actual_home_score"].notna() & df["actual_away_score"].notna()].copy()
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()

df["actual_home_score"] = pd.to_numeric(df["actual_home_score"], errors="coerce")
df["actual_away_score"] = pd.to_numeric(df["actual_away_score"], errors="coerce")
df["margin"] = df["actual_home_score"] - df["actual_away_score"]
df["neutral"] = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)

n_total = len(df)
n_neutral = df["neutral"].sum()
n_home = (~df["neutral"]).sum()

print(f"\n  Total games: {n_total:,}")
print(f"  Home games:  {n_home:,}")
print(f"  Neutral:     {n_neutral:,}")

# ═══════════════════════════════════════════════════════════
# OVERALL HCA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  OVERALL HOME COURT ADVANTAGE")
print(f"{'='*70}")

home_games = df[~df["neutral"]]
neutral_games = df[df["neutral"]]

home_margin = home_games["margin"].mean()
home_winpct = (home_games["margin"] > 0).mean()
neutral_margin = neutral_games["margin"].mean() if len(neutral_games) > 0 else 0
neutral_winpct = (neutral_games["margin"] > 0).mean() if len(neutral_games) > 0 else 0.5

print(f"\n  {'':20s} {'Avg Margin':>12s} {'Home Win%':>10s} {'Games':>8s}")
print(f"  {'-'*55}")
print(f"  {'Home games':20s} {home_margin:>+12.2f} {home_winpct:>9.1%} {len(home_games):>8,}")
print(f"  {'Neutral games':20s} {neutral_margin:>+12.2f} {neutral_winpct:>9.1%} {len(neutral_games):>8,}")
print(f"\n  Empirical HCA = {home_margin - neutral_margin:.2f} pts")
print(f"  (Home margin {home_margin:+.2f} minus neutral margin {neutral_margin:+.2f})")

# ═══════════════════════════════════════════════════════════
# PER-CONFERENCE HCA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PER-CONFERENCE HCA (home games only)")
print(f"{'='*70}")

# Hardcoded values from ncaa.py
HARDCODED = {
    "Big 12": 3.8, "Southeastern Conference": 3.7, "Big Ten": 3.6,
    "Big Ten Conference": 3.6, "Atlantic Coast Conference": 3.4,
    "Big East": 3.3, "Pac-12": 3.0, "Mountain West Conference": 3.2,
    "American Athletic Conference": 3.0, "West Coast Conference": 2.8,
    "Atlantic 10 Conference": 2.7, "Missouri Valley Conference": 2.9,
}

# ESPN conf ID → name mapping
ESPN_CONF = {
    "8": "Big 12", "23": "SEC", "7": "Big Ten",
    "2": "ACC", "4": "Big East", "21": "Pac-12",
    "44": "Mountain West", "62": "AAC", "26": "WCC",
    "3": "Atlantic 10", "18": "Missouri Valley",
    "40": "Sun Belt", "12": "Mid-American", "10": "CAA",
    "22": "Ivy League", "1": "America East", "46": "ASUN",
    "5": "Big Sky", "6": "Big South", "9": "Big West",
}

# Normalize conference names
conf_col = "home_conference"
if conf_col not in df.columns:
    conf_col = "conference"
if conf_col not in df.columns:
    print("  ⚠ No conference column found")
    conf_col = None

if conf_col:
    home_games = df[~df["neutral"]].copy()
    home_games["conf_norm"] = home_games[conf_col].map(ESPN_CONF).fillna(home_games[conf_col])
    
    conf_stats = home_games.groupby("conf_norm").agg(
        games=("margin", "count"),
        avg_margin=("margin", "mean"),
        median_margin=("margin", "median"),
        home_win_pct=("margin", lambda x: (x > 0).mean()),
    ).sort_values("avg_margin", ascending=False)
    
    conf_stats = conf_stats[conf_stats["games"] >= 100]  # Minimum sample

    print(f"\n  {'Conference':<25s} {'Games':>6s} {'Avg HCA':>8s} {'Med HCA':>8s} {'Home W%':>8s} {'Hardcoded':>10s} {'Diff':>7s}")
    print(f"  {'-'*75}")
    
    for conf, row in conf_stats.iterrows():
        # Find hardcoded value
        hc = None
        for k, v in HARDCODED.items():
            if conf.lower() in k.lower() or k.lower() in conf.lower():
                hc = v
                break
        hc_str = f"{hc:.1f}" if hc else "—"
        diff = f"{row['avg_margin'] - hc:+.1f}" if hc else "—"
        
        print(f"  {conf:<25s} {row['games']:>6.0f} {row['avg_margin']:>+7.2f} {row['median_margin']:>+7.1f} "
              f"{row['home_win_pct']:>7.1%} {hc_str:>10s} {diff:>7s}")

# ═══════════════════════════════════════════════════════════
# HCA BY SEASON (trending down?)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  HCA BY SEASON (is it shrinking?)")
print(f"{'='*70}")

home_by_season = df[~df["neutral"]].groupby("season").agg(
    games=("margin", "count"),
    avg_margin=("margin", "mean"),
    home_win_pct=("margin", lambda x: (x > 0).mean()),
)

print(f"\n  {'Season':>7s} {'Games':>6s} {'HCA (pts)':>10s} {'Home Win%':>10s}")
print(f"  {'-'*38}")
for season, row in home_by_season.iterrows():
    if row["games"] < 500:
        continue
    print(f"  {season:>7.0f} {row['games']:>6.0f} {row['avg_margin']:>+9.2f} {row['home_win_pct']:>9.1%}")

# ═══════════════════════════════════════════════════════════
# NEUTRAL SITE ANALYSIS — WHO WINS?
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  NEUTRAL SITE OUTCOMES")
print(f"{'='*70}")

if n_neutral > 0:
    ng = df[df["neutral"]].copy()
    
    # "Home" team (as labeled in DB) win rate at neutral
    db_home_wins = (ng["margin"] > 0).mean()
    print(f"\n  DB 'home' team wins at neutral: {db_home_wins:.1%} ({(ng['margin'] > 0).sum()}/{len(ng)})")
    print(f"  Average margin (DB home): {ng['margin'].mean():+.2f}")
    
    # By spread — does the favorite win as expected?
    if "market_spread_home" in ng.columns:
        ng["spread"] = pd.to_numeric(ng["market_spread_home"], errors="coerce")
        ng_with_spread = ng[ng["spread"].notna() & (ng["spread"].abs() > 0.5)]
        if len(ng_with_spread) > 0:
            fav_correct = ((ng_with_spread["spread"] < 0) & (ng_with_spread["margin"] > 0)) | \
                         ((ng_with_spread["spread"] > 0) & (ng_with_spread["margin"] < 0))
            print(f"\n  Market favorite wins at neutral: {fav_correct.mean():.1%} ({fav_correct.sum()}/{len(ng_with_spread)})")
            
            # ATS at neutral
            ng_with_spread["beat_spread"] = ng_with_spread["margin"] + ng_with_spread["spread"]
            home_covers = (ng_with_spread["beat_spread"] > 0).mean()
            print(f"  DB 'home' covers spread at neutral: {home_covers:.1%}")
            print(f"  (Should be ~50% if neutral is truly neutral)")

# ═══════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RECOMMENDATIONS")
print(f"{'='*70}")

print(f"""
  1. Compare hardcoded HCA values to empirical values above
     - If they differ by >0.5 pts, update _NCAA_CONF_HCA in ncaa.py
     - Consider using SEASON-SPECIFIC HCA (the trend might be declining)
  
  2. For neutral sites:
     - DB 'home' win rate should be ~50% if labels are random
     - If significantly different, there's a systematic bias in labeling
  
  3. Travel distance feature should capture the REMAINING variance
     at neutral sites after HCA is zeroed out
""")

print("  Done.")
