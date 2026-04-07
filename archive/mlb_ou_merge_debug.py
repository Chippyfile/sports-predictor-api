#!/usr/bin/env python3
"""
mlb_ou_merge_debug.py — Diagnose why odds CSV isn't matching mlb_training_data
"""
import sys, os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

# Load training data (same as mlb_retrain_ou_v2 does)
from mlb_retrain import load_data
df = load_data()
print(f"Training data: {len(df)} games")
print(f"Seasons: {sorted(df['season'].unique()) if 'season' in df.columns else 'no season col'}")

# Check what columns exist
print(f"\nKey columns present:")
for col in ["game_date", "home_team", "away_team", "market_ou_total", "season"]:
    if col in df.columns:
        sample = df[col].dropna().iloc[:3].tolist() if len(df[col].dropna()) > 0 else []
        nulls = df[col].isna().sum()
        print(f"  {col}: {nulls} nulls, samples: {sample}")
    else:
        print(f"  {col}: ❌ MISSING")

# Load odds CSV
odds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlb_odds_2014_2021.csv")
odds = pd.read_csv(odds_path)
print(f"\nOdds CSV: {len(odds)} games")
print(f"Seasons: {sorted(odds['season'].unique())}")

# ── Check team abbreviation mismatches ──
train_teams = set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique())
odds_teams = set(odds["home_team"].dropna().unique()) | set(odds["away_team"].dropna().unique())
print(f"\nTraining teams ({len(train_teams)}): {sorted(train_teams)}")
print(f"Odds teams ({len(odds_teams)}): {sorted(odds_teams)}")
print(f"In training but not odds: {sorted(train_teams - odds_teams)}")
print(f"In odds but not training: {sorted(odds_teams - train_teams)}")

# ── Check date format ──
print(f"\nTraining game_date samples: {df['game_date'].dropna().iloc[:5].tolist()}")
print(f"Odds game_date samples: {odds['game_date'].iloc[:5].tolist()}")

# Check dtype
print(f"Training game_date dtype: {df['game_date'].dtype}")
print(f"Odds game_date dtype: {odds['game_date'].dtype}")

# ── Try the merge and show what's happening ──
# Only look at overlapping seasons (2015-2019)
odds_overlap = odds[odds["season"].isin([2015, 2016, 2017, 2018, 2019])].copy()
train_overlap = df[df["season"].isin([2015, 2016, 2017, 2018, 2019])].copy() if "season" in df.columns else df.copy()
print(f"\nOverlapping seasons:")
print(f"  Training 2015-2019: {len(train_overlap)} games")
print(f"  Odds 2015-2019: {len(odds_overlap)} games")

# Ensure string types for merge
train_overlap["_gd"] = train_overlap["game_date"].astype(str).str.strip()
train_overlap["_ht"] = train_overlap["home_team"].astype(str).str.strip().str.upper()
train_overlap["_at"] = train_overlap["away_team"].astype(str).str.strip().str.upper()

odds_overlap["_gd"] = odds_overlap["game_date"].astype(str).str.strip()
odds_overlap["_ht"] = odds_overlap["home_team"].astype(str).str.strip().str.upper()
odds_overlap["_at"] = odds_overlap["away_team"].astype(str).str.strip().str.upper()

# Build key sets
train_keys = set(zip(train_overlap["_gd"], train_overlap["_ht"], train_overlap["_at"]))
odds_keys = set(zip(odds_overlap["_gd"], odds_overlap["_ht"], odds_overlap["_at"]))
matched = train_keys & odds_keys
only_train = train_keys - odds_keys
only_odds = odds_keys - train_keys

print(f"\n  Matched keys: {len(matched)}")
print(f"  Only in training: {len(only_train)}")
print(f"  Only in odds: {len(only_odds)}")

# Show sample unmatched
if only_train:
    samples = list(only_train)[:10]
    print(f"\n  Sample keys ONLY in training (not in odds):")
    for gd, ht, at in samples:
        print(f"    {gd}  {at}@{ht}")

if only_odds:
    samples = list(only_odds)[:10]
    print(f"\n  Sample keys ONLY in odds (not in training):")
    for gd, ht, at in samples:
        print(f"    {gd}  {at}@{ht}")

# ── Check if existing market_ou_total overlaps with odds ──
has_ou = pd.to_numeric(train_overlap.get("market_ou_total", 0), errors="coerce").fillna(0) > 0
print(f"\n  Training 2015-2019 with existing market_ou_total: {has_ou.sum()}")
print(f"  Training 2015-2019 WITHOUT market_ou_total: {(~has_ou).sum()} ← these need backfill")

# ── Check for date format issues ──
# Maybe training data has different date format?
if only_train and only_odds:
    t_dates = {gd for gd, _, _ in only_train}
    o_dates = {gd for gd, _, _ in only_odds}
    # Check if same dates exist but format differs
    print(f"\n  Unique dates only-in-training: {len(t_dates)} (samples: {sorted(list(t_dates))[:5]})")
    print(f"  Unique dates only-in-odds: {len(o_dates)} (samples: {sorted(list(o_dates))[:5]})")
    
    # Check if it's a team name issue on shared dates
    shared_dates = t_dates & o_dates
    if shared_dates:
        sample_date = sorted(shared_dates)[0]
        t_games = [(ht, at) for gd, ht, at in only_train if gd == sample_date]
        o_games = [(ht, at) for gd, ht, at in only_odds if gd == sample_date]
        print(f"\n  On {sample_date}:")
        print(f"    Training teams: {t_games[:5]}")
        print(f"    Odds teams: {o_games[:5]}")
