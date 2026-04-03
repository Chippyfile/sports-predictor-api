#!/usr/bin/env python3
"""
ncaa_hca_paired.py — Paired HCA: same team, home vs away vs neutral
====================================================================
For each team-season, computes average margin at home, away, neutral.
HCA = (home_margin - away_margin) / 2
This controls for team quality — better teams inflate raw HCA numbers.

Usage:
    python3 ncaa_hca_paired.py
"""
import sys, os, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dump_training_data import load_cached, dump

# ESPN conf ID → name mapping
ESPN_CONF = {
    "8": "Big 12", "23": "SEC", "7": "Big Ten",
    "2": "ACC", "4": "Big East", "21": "Pac-12",
    "44": "Mountain West", "62": "AAC", "26": "WCC",
    "3": "Atlantic 10", "18": "Missouri Valley",
    "40": "Sun Belt", "12": "Mid-American", "10": "CAA",
    "22": "Ivy League", "1": "America East", "46": "ASUN",
    "5": "Big Sky", "6": "Big South", "9": "Big West",
    "11": "Conference USA", "13": "Horizon", "14": "MAAC",
    "19": "Ohio Valley", "20": "Patriot", "24": "Southland",
    "25": "SWAC", "27": "Summit", "29": "WAC",
    "45": "Northeast", "49": "Southern",
}

HARDCODED = {
    "Big 12": 3.8, "SEC": 3.7, "Big Ten": 3.6,
    "ACC": 3.4, "Big East": 3.3, "Pac-12": 3.0,
    "Mountain West": 3.2, "AAC": 3.0, "WCC": 2.8,
    "Atlantic 10": 2.7, "Missouri Valley": 2.9,
}

print("=" * 70)
print("  NCAA PAIRED HCA — Same Team: Home vs Away vs Neutral")
print("=" * 70)

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

# Normalize conference columns
for col in ["home_conference", "away_conference"]:
    if col in df.columns:
        df[col] = df[col].astype(str).map(lambda x: ESPN_CONF.get(x, ESPN_CONF.get(f"conf_{x}".replace("conf_",""), x)))

# ═══════════════════════════════════════════════════════════
# Build team-season records: home margin, away margin, neutral margin
# ═══════════════════════════════════════════════════════════

print(f"\n  Building per-team-season records...")

records = []

# Home team perspective
for _, row in df.iterrows():
    season = row["season"]
    neutral = row["neutral"]
    margin = row["margin"]
    
    home_team = row.get("home_team_id") or row.get("home_team", "")
    away_team = row.get("away_team_id") or row.get("away_team", "")
    home_conf = row.get("home_conference", "Unknown")
    away_conf = row.get("away_conference", "Unknown")
    
    if neutral:
        # Both teams are "neutral" — record for both
        records.append({"team": str(home_team), "season": season, "conf": home_conf,
                       "venue": "neutral", "margin": margin})
        records.append({"team": str(away_team), "season": season, "conf": away_conf,
                       "venue": "neutral", "margin": -margin})
    else:
        # Home game for home_team, away game for away_team
        records.append({"team": str(home_team), "season": season, "conf": home_conf,
                       "venue": "home", "margin": margin})
        records.append({"team": str(away_team), "season": season, "conf": away_conf,
                       "venue": "away", "margin": -margin})

rdf = pd.DataFrame(records)
print(f"  {len(rdf):,} team-game records")

# ═══════════════════════════════════════════════════════════
# Per team-season: compute home/away/neutral averages
# ═══════════════════════════════════════════════════════════

# Pivot: for each team-season, get avg margin by venue type
team_season = rdf.groupby(["team", "season", "conf", "venue"])["margin"].agg(["mean", "count"]).reset_index()
team_season.columns = ["team", "season", "conf", "venue", "avg_margin", "games"]

# Pivot wider
ts_wide = team_season.pivot_table(
    index=["team", "season", "conf"],
    columns="venue",
    values=["avg_margin", "games"],
    fill_value=np.nan
)

# Flatten column names
ts_wide.columns = [f"{stat}_{venue}" for stat, venue in ts_wide.columns]
ts_wide = ts_wide.reset_index()

# Only keep teams with BOTH home and away games (minimum 3 each)
has_both = (ts_wide.get("games_home", pd.Series(0)) >= 3) & (ts_wide.get("games_away", pd.Series(0)) >= 3)
paired = ts_wide[has_both].copy()

print(f"  {len(paired):,} team-seasons with ≥3 home AND ≥3 away games")

# Compute paired HCA
paired["hca_raw"] = paired["avg_margin_home"] - paired["avg_margin_away"]
paired["hca"] = paired["hca_raw"] / 2  # Divide by 2: home gets +HCA, away gets -HCA

# ═══════════════════════════════════════════════════════════
# OVERALL PAIRED HCA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  OVERALL PAIRED HCA")
print(f"{'='*70}")

overall_hca = paired["hca"].mean()
overall_median = paired["hca"].median()
overall_std = paired["hca"].std()
print(f"\n  Mean paired HCA:   {overall_hca:+.2f} pts")
print(f"  Median paired HCA: {overall_median:+.2f} pts")
print(f"  Std dev:           {overall_std:.2f} pts")
print(f"  Raw difference:    {paired['hca_raw'].mean():+.2f} pts (home margin - away margin)")
print(f"  Teams analyzed:    {len(paired):,} team-seasons")

# ═══════════════════════════════════════════════════════════
# PER-CONFERENCE PAIRED HCA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PER-CONFERENCE PAIRED HCA")
print(f"{'='*70}")

conf_hca = paired.groupby("conf").agg(
    teams=("hca", "count"),
    mean_hca=("hca", "mean"),
    median_hca=("hca", "median"),
    std_hca=("hca", "std"),
    mean_raw=("hca_raw", "mean"),
    home_margin=("avg_margin_home", "mean"),
    away_margin=("avg_margin_away", "mean"),
).sort_values("mean_hca", ascending=False)

conf_hca = conf_hca[conf_hca["teams"] >= 10]  # Minimum 10 team-seasons

print(f"\n  {'Conference':<20s} {'Teams':>6s} {'HCA':>7s} {'Median':>7s} {'Home M':>8s} {'Away M':>8s} {'Raw Δ':>7s} {'Coded':>6s} {'Gap':>6s}")
print(f"  {'-'*82}")

for conf, row in conf_hca.iterrows():
    hc = HARDCODED.get(conf)
    hc_str = f"{hc:.1f}" if hc else "—"
    gap = f"{row['mean_hca'] - hc:+.1f}" if hc else "—"
    print(f"  {conf:<20s} {row['teams']:>6.0f} {row['mean_hca']:>+6.2f} {row['median_hca']:>+6.2f} "
          f"{row['home_margin']:>+7.2f} {row['away_margin']:>+7.2f} {row['mean_raw']:>+6.2f} "
          f"{hc_str:>6s} {gap:>6s}")

# ═══════════════════════════════════════════════════════════
# PER-SEASON PAIRED HCA
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PAIRED HCA BY SEASON")
print(f"{'='*70}")

season_hca = paired.groupby("season").agg(
    teams=("hca", "count"),
    mean_hca=("hca", "mean"),
    median_hca=("hca", "median"),
).sort_index()

print(f"\n  {'Season':>7s} {'Teams':>6s} {'Mean HCA':>9s} {'Median':>8s}")
print(f"  {'-'*34}")
for season, row in season_hca.iterrows():
    if row["teams"] < 100:
        continue
    print(f"  {season:>7.0f} {row['teams']:>6.0f} {row['mean_hca']:>+8.2f} {row['median_hca']:>+7.2f}")

# ═══════════════════════════════════════════════════════════
# NEUTRAL SITE PAIRED: teams with home + away + neutral
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  NEUTRAL SITE: SAME TEAMS — Home vs Away vs Neutral")
print(f"{'='*70}")

has_all = (ts_wide.get("games_home", pd.Series(0)) >= 3) & \
          (ts_wide.get("games_away", pd.Series(0)) >= 3) & \
          (ts_wide.get("games_neutral", pd.Series(0)) >= 1)
triple = ts_wide[has_all].copy()
triple["hca"] = (triple["avg_margin_home"] - triple["avg_margin_away"]) / 2

print(f"\n  {len(triple):,} team-seasons with home + away + neutral games")

if len(triple) > 0:
    print(f"\n  {'Venue':<10s} {'Avg Margin':>11s} {'Games/team':>11s}")
    print(f"  {'-'*35}")
    print(f"  {'Home':<10s} {triple['avg_margin_home'].mean():>+10.2f} {triple['games_home'].mean():>10.1f}")
    print(f"  {'Away':<10s} {triple['avg_margin_away'].mean():>+10.2f} {triple['games_away'].mean():>10.1f}")
    print(f"  {'Neutral':<10s} {triple['avg_margin_neutral'].mean():>+10.2f} {triple['games_neutral'].mean():>10.1f}")
    
    expected_neutral = (triple['avg_margin_home'].mean() + triple['avg_margin_away'].mean()) / 2
    actual_neutral = triple['avg_margin_neutral'].mean()
    
    print(f"\n  Expected neutral margin (midpoint of home+away): {expected_neutral:+.2f}")
    print(f"  Actual neutral margin:                           {actual_neutral:+.2f}")
    print(f"  Difference:                                      {actual_neutral - expected_neutral:+.2f}")
    print(f"  → If positive, 'home' label at neutral still gives ~{actual_neutral - expected_neutral:.1f} pts advantage")
    
    # By conference for triple-venue teams
    print(f"\n  Per-conference neutral analysis (teams with all 3 venue types):")
    conf_neutral = triple.groupby("conf").agg(
        teams=("hca", "count"),
        home_m=("avg_margin_home", "mean"),
        away_m=("avg_margin_away", "mean"),
        neutral_m=("avg_margin_neutral", "mean"),
        hca=("hca", "mean"),
    )
    conf_neutral = conf_neutral[conf_neutral["teams"] >= 5]
    conf_neutral["neutral_vs_expected"] = conf_neutral["neutral_m"] - (conf_neutral["home_m"] + conf_neutral["away_m"]) / 2
    conf_neutral = conf_neutral.sort_values("hca", ascending=False)
    
    print(f"\n  {'Conference':<20s} {'Teams':>6s} {'HCA':>6s} {'Home':>7s} {'Away':>7s} {'Neut':>7s} {'N vs Exp':>9s}")
    print(f"  {'-'*68}")
    for conf, row in conf_neutral.iterrows():
        print(f"  {conf:<20s} {row['teams']:>6.0f} {row['hca']:>+5.2f} {row['home_m']:>+6.2f} "
              f"{row['away_m']:>+6.2f} {row['neutral_m']:>+6.2f} {row['neutral_vs_expected']:>+8.2f}")

print(f"\n  Done.")
