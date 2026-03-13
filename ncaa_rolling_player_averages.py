#!/usr/bin/env python3
"""
ncaa_rolling_player_averages.py — Compute rolling player averages from per-game data
═════════════════════════════════════════════════════════════════════════════════════
For each game, looks up both teams' PRIOR games (no leakage) and computes
rolling averages of player-derived features over a configurable window.

Input columns (per-game, already in Supabase):
  home/away_star1_pts_share, home/away_top3_pts_share,
  home/away_bench_pts, home/away_bench_pts_share,
  home/away_minutes_hhi, home/away_players_used

Output columns (rolling averages, pushed to Supabase):
  home/away_roll_star1_share, home/away_roll_top3_share,
  home/away_roll_bench_share, home/away_roll_hhi,
  home/away_roll_players_used, home/away_roll_bench_pts

Usage:
  python3 ncaa_rolling_player_averages.py --dry-run
  python3 ncaa_rolling_player_averages.py
  python3 ncaa_rolling_player_averages.py --window 10   # default is 10 games
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY env var")
    sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def sb_get(table, params=""):
    all_data = []
    offset = 0
    limit = 1000
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}&limit={limit}&offset={offset}"
        h = {**HEADERS, "Range": f"{offset}-{offset+limit-1}"}
        r = requests.get(url, headers=h, timeout=60)
        if not r.ok:
            print(f"  Error: {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
        if offset % 10000 == 0:
            print(f"  Fetched {len(all_data)} rows...")
    return all_data


def compute_rolling_averages(df, window=10):
    """
    For each game, compute rolling averages of player stats from prior games.
    
    Key insight: a team's stats appear as home_ in some games and away_ in others.
    We need to track each team's stats regardless of home/away designation.
    """
    print(f"\n  Computing rolling averages (window={window})...")
    t0 = time.time()

    # Player stat columns (per-game raw values)
    stat_cols = {
        "star1_pts_share": "roll_star1_share",
        "top3_pts_share": "roll_top3_share",
        "bench_pts_share": "roll_bench_share",
        "bench_pts": "roll_bench_pts",
        "minutes_hhi": "roll_hhi",
        "players_used": "roll_players_used",
    }

    # Sort by date for chronological processing
    df = df.sort_values("game_date").reset_index(drop=True)

    # Build per-team game history: {team_id: [(game_date, {stat: value}), ...]}
    team_history = {}

    # Initialize output columns
    for prefix in ["home_", "away_"]:
        for out_col in stat_cols.values():
            df[prefix + out_col] = np.nan

    n = len(df)
    filled = 0

    for idx in range(n):
        row = df.iloc[idx]
        game_date = row["game_date"]
        home_id = str(row.get("home_team_id", ""))
        away_id = str(row.get("away_team_id", ""))

        if not home_id or not away_id:
            continue

        # Look up rolling averages from prior games for each team
        for team_id, prefix in [(home_id, "home_"), (away_id, "away_")]:
            if team_id in team_history and len(team_history[team_id]) > 0:
                history = team_history[team_id]
                # Take last N games
                recent = history[-window:]
                for raw_col, out_col in stat_cols.items():
                    vals = [g[raw_col] for g in recent if raw_col in g and g[raw_col] is not None]
                    if vals:
                        df.at[idx, prefix + out_col] = np.mean(vals)
                        filled += 1

        # After computing rolling averages, ADD this game's stats to history
        for team_id, prefix in [(home_id, "home_"), (away_id, "away_")]:
            game_stats = {}
            for raw_col in stat_cols.keys():
                val = row.get(prefix + raw_col)
                if pd.notna(val):
                    game_stats[raw_col] = float(val)
            
            if game_stats:  # Only add if we have stats
                if team_id not in team_history:
                    team_history[team_id] = []
                team_history[team_id].append(game_stats)

        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n - idx - 1) / rate / 60
            print(f"    {idx+1}/{n} | {filled} values filled | {eta:.1f}min left")

    elapsed = time.time() - t0
    print(f"  Done: {filled} rolling values computed in {elapsed:.0f}s")
    print(f"  Teams tracked: {len(team_history)}")

    # Coverage report
    print(f"\n  ROLLING AVERAGE COVERAGE:")
    for prefix in ["home_", "away_"]:
        for out_col in stat_cols.values():
            col = prefix + out_col
            n_filled = int(df[col].notna().sum())
            print(f"    {col:<35} {n_filled:>8,}/{n} ({n_filled/n*100:.1f}%)")

    return df


def push_to_supabase(df, dry_run=False):
    """Push rolling average columns to ncaa_historical."""
    out_cols = [c for c in df.columns if "roll_" in c and ("home_" in c or "away_" in c)]
    
    # Only push rows that have at least one rolling value
    has_data = df[out_cols].notna().any(axis=1)
    to_push = df.loc[has_data, ["game_id"] + out_cols].copy()
    
    # Round floats
    for col in out_cols:
        to_push[col] = to_push[col].round(4)
    
    n = len(to_push)
    print(f"\n  Rows to push: {n}")
    print(f"  Columns: {out_cols}")

    if dry_run:
        print(f"\n  DRY RUN — showing 3 samples:")
        for _, row in to_push.head(3).iterrows():
            vals = {k: v for k, v in row.items() if pd.notna(v) and k != "game_id"}
            print(f"    {row['game_id']}: {json.dumps(vals, default=str)}")
        print(f"\n  Would push {n} rows. Run without --dry-run to execute.")
        return

    print(f"  Pushing {n:,} patches to ncaa_historical...")
    success = 0
    errors = 0
    t0 = time.time()

    for i, (_, row) in enumerate(to_push.iterrows()):
        game_id = row["game_id"]
        patch = {}
        for col in out_cols:
            val = row[col]
            if pd.notna(val):
                patch[col] = float(val)

        if not patch:
            continue

        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
        try:
            r = requests.patch(url, headers=HEADERS, json=patch, timeout=15)
            if r.ok:
                success += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"    Error {r.status_code} for {game_id}: {r.text[:100]}")
        except Exception as e:
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(f"    {i+1}/{n} | success:{success} errors:{errors} | {eta:.1f}min left")

        if (i + 1) % 50 == 0:
            time.sleep(0.2)

    elapsed = time.time() - t0
    print(f"\n  PUSH COMPLETE: {success:,} success, {errors} errors in {elapsed/60:.1f}min")


def main():
    parser = argparse.ArgumentParser(description="Compute rolling player averages")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size (default 10)")
    parser.add_argument("--compute-only", action="store_true", help="Compute and save CSV, don't push")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  ROLLING PLAYER AVERAGES (window={args.window})")
    print("=" * 70)

    # Fetch all games with player data
    print("\n  Fetching games...")
    cols = ("game_id,game_date,season,home_team_id,away_team_id,"
            "home_star1_pts_share,away_star1_pts_share,"
            "home_top3_pts_share,away_top3_pts_share,"
            "home_bench_pts,away_bench_pts,"
            "home_bench_pts_share,away_bench_pts_share,"
            "home_minutes_hhi,away_minutes_hhi,"
            "home_players_used,away_players_used")
    
    rows = sb_get("ncaa_historical",
                  f"actual_home_score=not.is.null&select={cols}&order=game_date.asc")
    
    if not rows:
        print("  ERROR: No data fetched")
        sys.exit(1)

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ["game_id", "game_date", "home_team_id", "away_team_id", "season"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Loaded {len(df)} games")

    # Compute rolling averages
    df = compute_rolling_averages(df, window=args.window)

    # Save CSV for inspection
    out_cols = [c for c in df.columns if "roll_" in c]
    csv_path = f"rolling_player_avg_w{args.window}.csv"
    df[["game_id", "game_date", "home_team_id", "away_team_id"] + out_cols].to_csv(csv_path, index=False)
    print(f"\n  Saved to {csv_path}")

    if args.compute_only:
        print("  --compute-only mode, stopping before push.")
        return

    # Push to Supabase
    push_to_supabase(df, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    print(f"  NEXT: Wire rolling columns into ncaa_build_features()")
    print(f"  New features: roll_star1_share_diff, roll_bench_share_diff,")
    print(f"                roll_hhi_diff, roll_players_used_diff, etc.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
