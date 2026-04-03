#!/usr/bin/env python3
"""
compute_rolling_hca.py — Per-team rolling home court advantage
==============================================================
For each game, computes the home team's rolling HCA from their
recent home and away results. Replaces static conference HCA.

Feature: rolling_hca
- Uses last 20 home games + last 20 away games for each team
- HCA = (avg home margin - avg away margin) / 2
- Minimum 5 games each venue type, else falls back to conference avg

Can be used for:
- Training: backfill historical games with rolling_hca
- Serving: compute at prediction time from recent results

Usage:
    python3 compute_rolling_hca.py                # Analyze + show stats
    python3 compute_rolling_hca.py --backfill     # Add to training data
"""
import sys, os, warnings, argparse
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dump_training_data import load_cached, dump

WINDOW = 20  # Last N games per venue type
MIN_GAMES = 5  # Minimum games before using rolling HCA

# Empirical conference HCA (from paired analysis) — fallback only
CONF_HCA_EMPIRICAL = {
    "Big 12": 7.3, "SEC": 7.3, "Big Ten": 6.2, "ACC": 5.8,
    "Big East": 4.6, "Pac-12": 4.2, "Mountain West": 5.0,
    "AAC": 5.2, "WCC": 10.6, "Atlantic 10": 6.4,
    "Missouri Valley": 6.1, "Ivy League": 5.4,
    "Sun Belt": 6.5, "Conference USA": 6.9, "CAA": 5.5,
    "MAAC": 6.7, "ASUN": 9.6, "Big Sky": 8.6,
    "Southland": 8.1, "Ohio Valley": 7.5, "Summit": 7.1,
    "WAC": 6.2, "Patriot": 7.2, "Big South": 8.4,
    "SWAC": 6.2, "Horizon": 5.8, "Northeast": 7.4,
    "Big West": 7.0, "Southern": 7.9,
}
DEFAULT_HCA = 6.6  # Overall paired mean

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


def compute_rolling_hca(df):
    """
    Compute per-team rolling HCA for every game in the dataframe.
    Must be sorted by date. Uses only games BEFORE each row (no leakage).
    
    Returns df with new columns: rolling_hca, rolling_hca_source
    """
    df = df.copy()
    df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
    df = df.sort_values("game_date_dt").reset_index(drop=True)
    
    # Normalize conferences
    for col in ["home_conference", "away_conference"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: ESPN_CONF.get(x, x))
    
    df["actual_home_score"] = pd.to_numeric(df["actual_home_score"], errors="coerce")
    df["actual_away_score"] = pd.to_numeric(df["actual_away_score"], errors="coerce")
    df["margin"] = df["actual_home_score"] - df["actual_away_score"]
    neutral = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)
    
    # Build rolling history for each team
    # Track: team_id → deque of (margin, venue_type) for last N games
    from collections import deque, defaultdict
    
    team_home_margins = defaultdict(lambda: deque(maxlen=WINDOW))
    team_away_margins = defaultdict(lambda: deque(maxlen=WINDOW))
    
    rolling_hca = np.full(len(df), np.nan)
    rolling_hca_source = [""] * len(df)
    
    for i, row in df.iterrows():
        home_id = str(row.get("home_team_id", row.get("home_team", "")))
        away_id = str(row.get("away_team_id", row.get("away_team", "")))
        is_neutral = neutral.iloc[i] if isinstance(neutral, pd.Series) else neutral
        margin = row["margin"]
        home_conf = row.get("home_conference", "")
        
        # ── COMPUTE: rolling HCA for the HOME team (before this game) ──
        home_home_list = list(team_home_margins[home_id])
        home_away_list = list(team_away_margins[home_id])
        
        if len(home_home_list) >= MIN_GAMES and len(home_away_list) >= MIN_GAMES:
            avg_home = np.mean(home_home_list)
            avg_away = np.mean(home_away_list)
            rolling_hca[i] = (avg_home - avg_away) / 2
            rolling_hca_source[i] = "rolling"
        else:
            # Fallback to conference empirical HCA
            conf_hca = CONF_HCA_EMPIRICAL.get(home_conf, DEFAULT_HCA)
            rolling_hca[i] = conf_hca
            rolling_hca_source[i] = "conf_fallback"
        
        # ── UPDATE: add this game to history (AFTER computing feature) ──
        if pd.notna(margin):
            if is_neutral:
                # Neutral: don't update home/away history (it's neither)
                pass
            else:
                team_home_margins[home_id].append(margin)
                team_away_margins[away_id].append(-margin)  # Away team's perspective
    
    df["rolling_hca"] = rolling_hca
    df["rolling_hca_source"] = rolling_hca_source
    
    return df


def analyze(df):
    """Show stats about rolling HCA."""
    print(f"\n{'='*70}")
    print(f"  ROLLING HCA ANALYSIS")
    print(f"{'='*70}")
    
    valid = df[df["rolling_hca"].notna()]
    rolling = valid[valid["rolling_hca_source"] == "rolling"]
    fallback = valid[valid["rolling_hca_source"] == "conf_fallback"]
    
    print(f"\n  Total games:    {len(valid):,}")
    print(f"  Rolling HCA:    {len(rolling):,} ({len(rolling)/len(valid)*100:.1f}%)")
    print(f"  Conf fallback:  {len(fallback):,} ({len(fallback)/len(valid)*100:.1f}%)")
    
    print(f"\n  Rolling HCA stats:")
    print(f"    Mean:   {rolling['rolling_hca'].mean():+.2f}")
    print(f"    Median: {rolling['rolling_hca'].median():+.2f}")
    print(f"    Std:    {rolling['rolling_hca'].std():.2f}")
    print(f"    Min:    {rolling['rolling_hca'].min():+.2f}")
    print(f"    Max:    {rolling['rolling_hca'].max():+.2f}")
    
    # Correlation with actual margin
    non_neutral = valid[~valid.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)]
    if len(non_neutral) > 100:
        corr = non_neutral["rolling_hca"].corr(non_neutral["margin"])
        print(f"\n  Correlation with actual margin (non-neutral): r={corr:.4f}")
    
    # Compare: rolling vs static conference HCA
    print(f"\n  {'Metric':<30s} {'Rolling':>10s} {'Static Conf':>12s}")
    print(f"  {'-'*55}")
    
    if "hca_pts" in valid.columns:
        static_non_neutral = non_neutral[non_neutral["hca_pts"] > 0]
        if len(static_non_neutral) > 100:
            static_corr = static_non_neutral["hca_pts"].corr(static_non_neutral["margin"])
            rolling_corr = static_non_neutral["rolling_hca"].corr(static_non_neutral["margin"])
            print(f"  {'Corr with margin':<30s} {rolling_corr:>+9.4f} {static_corr:>+11.4f}")
    
    # Distribution by decile
    print(f"\n  Rolling HCA by decile — does higher HCA predict bigger home wins?")
    non_neutral_rolling = non_neutral[non_neutral["rolling_hca_source"] == "rolling"].copy()
    if len(non_neutral_rolling) > 500:
        non_neutral_rolling["hca_decile"] = pd.qcut(non_neutral_rolling["rolling_hca"], 10, labels=False, duplicates="drop")
        decile_stats = non_neutral_rolling.groupby("hca_decile").agg(
            games=("margin", "count"),
            hca=("rolling_hca", "mean"),
            actual_margin=("margin", "mean"),
            home_win_pct=("margin", lambda x: (x > 0).mean()),
        )
        print(f"\n  {'Decile':>7s} {'HCA':>6s} {'Actual Margin':>14s} {'Home Win%':>10s} {'Games':>6s}")
        print(f"  {'-'*48}")
        for dec, row in decile_stats.iterrows():
            print(f"  {dec:>7.0f} {row['hca']:>+5.1f} {row['actual_margin']:>+13.2f} {row['home_win_pct']:>9.1%} {row['games']:>6.0f}")
    
    # Per-season coverage
    print(f"\n  Per-season rolling HCA coverage:")
    season_stats = valid.groupby("season").agg(
        total=("rolling_hca_source", "count"),
        rolling=("rolling_hca_source", lambda x: (x == "rolling").sum()),
        mean_hca=("rolling_hca", "mean"),
    )
    print(f"\n  {'Season':>7s} {'Total':>6s} {'Rolling':>8s} {'Coverage':>9s} {'Mean HCA':>9s}")
    print(f"  {'-'*42}")
    for season, row in season_stats.iterrows():
        if row["total"] < 500:
            continue
        pct = row["rolling"] / row["total"] * 100
        print(f"  {season:>7.0f} {row['total']:>6.0f} {row['rolling']:>8.0f} {pct:>8.1f}% {row['mean_hca']:>+8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  PER-TEAM ROLLING HOME COURT ADVANTAGE")
    print(f"  Window: last {WINDOW} home + {WINDOW} away games")
    print(f"  Min games: {MIN_GAMES} each venue type")
    print("=" * 70)
    
    df = load_cached()
    if df is None:
        df = dump()
    
    df = df[df["actual_home_score"].notna() & df["actual_away_score"].notna()].copy()
    df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
    df = df[~df["season"].isin([2020, 2021])].copy()
    
    print(f"\n  {len(df):,} games loaded")
    
    df = compute_rolling_hca(df)
    analyze(df)
    
    if args.backfill:
        # Save enhanced dataframe
        out = "ncaa_training_with_rolling_hca.parquet"
        df.to_parquet(out)
        print(f"\n  Saved to {out}")
    
    print(f"\n  Done.")
