"""
NBA Player Ratings & Lineup Value Pipeline
Consumes nba_player_boxscores.parquet from the ESPN scraper.

Computes:
  1. Per-player season ratings (rolling PER-lite, game score, minutes share)
  2. Per-game lineup value (sum of starter ratings for each team)
  3. Star dependency (top-1, top-3 scoring concentration)
  4. Bench depth metrics
  5. Rotation size

These feed into nba_build_features_v23 as the player-level features.

Usage:
  python build_player_ratings.py                # Build from scraped data
  python build_player_ratings.py --merge        # Merge into training parquet
"""

import os, sys, time
import numpy as np
import pandas as pd


def parse_fg(fg_str):
    """Parse '5-12' into (made, attempted)."""
    if not fg_str or fg_str == "0-0" or not isinstance(fg_str, str):
        return 0, 0
    try:
        parts = fg_str.split("-")
        return int(parts[0]), int(parts[1])
    except:
        return 0, 0


def parse_minutes(min_str):
    """Parse minutes string to float."""
    if not min_str or min_str == "0":
        return 0.0
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except:
        return 0.0


def compute_game_score(row):
    """
    John Hollinger's Game Score (simplified PER for a single game).
    GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TO
    We don't have ORB/DRB/PF split, so use simplified version:
    GmSc ≈ PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*REB + STL + 0.7*AST + 0.7*BLK - TO
    """
    pts = float(row.get("PTS", 0) or 0)
    reb = float(row.get("REB", 0) or 0)
    ast = float(row.get("AST", 0) or 0)
    stl = float(row.get("STL", 0) or 0)
    blk = float(row.get("BLK", 0) or 0)
    to = float(row.get("TO", 0) or 0)

    fgm, fga = parse_fg(row.get("FG", "0-0"))
    ftm, fta = parse_fg(row.get("FT", "0-0"))

    return pts + 0.4*fgm - 0.7*fga - 0.4*(fta - ftm) + 0.7*reb + stl + 0.7*ast + 0.7*blk - to


def build_player_ratings(box_df):
    """
    Build per-player season ratings from box score data.
    Returns DataFrame with one row per player-season with rolling metrics.
    """
    df = box_df.copy()

    # Parse numeric columns
    for col in ["PTS", "REB", "AST", "STL", "BLK", "TO"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["minutes"] = df["MIN"].apply(parse_minutes)
    df["game_score"] = df.apply(compute_game_score, axis=1)
    df["plus_minus"] = pd.to_numeric(df["+/-"], errors="coerce").fillna(0)

    # Parse FG for FGA
    fg_parsed = df["FG"].apply(lambda x: parse_fg(x))
    df["fgm"] = fg_parsed.apply(lambda x: x[0])
    df["fga"] = fg_parsed.apply(lambda x: x[1])

    # Sort chronologically
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    print(f"  Processing {len(df)} player-game rows for {df['player_id'].nunique()} unique players")

    return df


def compute_lineup_features(box_df, training_df):
    """
    For each game in training_df, compute:
      - home_lineup_value / away_lineup_value (sum of starter game scores)
      - home_star1_share / away_star1_share (top scorer's points share)
      - home_top3_share / away_top3_share (top 3 scorers' share)
      - home_bench_pts / away_bench_pts
      - home_rotation_size / away_rotation_size (players with 10+ min)
      - home_starter_rating / away_starter_rating (rolling avg game score of starters)
    """
    box = box_df.copy()

    # Compute rolling player rating (last 10 games average game score)
    box["game_score"] = box.apply(compute_game_score, axis=1) if "game_score" not in box.columns else box["game_score"]
    box["minutes"] = box["MIN"].apply(parse_minutes) if "minutes" not in box.columns else box["minutes"]
    box["PTS"] = pd.to_numeric(box["PTS"], errors="coerce").fillna(0)

    # Build per-player rolling stats
    box = box.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    box["rolling_gs_10"] = box.groupby("player_id")["game_score"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    box["rolling_pts_10"] = box.groupby("player_id")["PTS"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    box["rolling_min_10"] = box.groupby("player_id")["minutes"].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )

    # For each game, compute lineup features
    results = {}
    games_grouped = box.groupby("game_id")

    for game_id, game_box in games_grouped:
        game_result = {}

        for side in ["home", "away"]:
            side_players = game_box[game_box["side"] == side].copy()
            if len(side_players) == 0:
                continue

            starters = side_players[side_players["starter"] == True]
            bench = side_players[side_players["starter"] == False]

            # Lineup value: sum of starters' rolling game score
            starter_ratings = starters["rolling_gs_10"].dropna()
            game_result[f"{side}_lineup_value"] = float(starter_ratings.sum()) if len(starter_ratings) > 0 else 0

            # Star share: top scorer's points / team total
            total_pts = side_players["PTS"].sum()
            if total_pts > 0:
                sorted_pts = side_players["PTS"].sort_values(ascending=False)
                game_result[f"{side}_star1_share"] = float(sorted_pts.iloc[0] / total_pts)
                game_result[f"{side}_top3_share"] = float(sorted_pts.iloc[:3].sum() / total_pts) if len(sorted_pts) >= 3 else float(sorted_pts.sum() / total_pts)
            else:
                game_result[f"{side}_star1_share"] = 0.2
                game_result[f"{side}_top3_share"] = 0.6

            # Bench scoring
            game_result[f"{side}_bench_pts"] = float(bench["PTS"].sum())

            # Rotation size (players with 10+ minutes)
            game_result[f"{side}_rotation_size"] = int((side_players["minutes"] >= 10).sum())

            # Starter average rating
            game_result[f"{side}_starter_avg_rating"] = float(starter_ratings.mean()) if len(starter_ratings) > 0 else 0

            # HHI (scoring concentration: Herfindahl-Hirschman Index)
            if total_pts > 0:
                shares = (side_players["PTS"] / total_pts) ** 2
                game_result[f"{side}_scoring_hhi"] = float(shares.sum())
            else:
                game_result[f"{side}_scoring_hhi"] = 0.2

        results[str(game_id)] = game_result

    # Convert to DataFrame
    lineup_df = pd.DataFrame.from_dict(results, orient="index")
    lineup_df.index.name = "game_id"
    lineup_df = lineup_df.reset_index()

    print(f"  Computed lineup features for {len(lineup_df)} games")
    print(f"    home_lineup_value mean: {lineup_df['home_lineup_value'].mean():.1f}")
    print(f"    home_star1_share mean: {lineup_df['home_star1_share'].mean():.3f}")
    print(f"    home_rotation_size mean: {lineup_df['home_rotation_size'].mean():.1f}")

    return lineup_df


def build_referee_profiles(ref_df, training_df):
    """
    Build referee tendency profiles from historical assignments.
    For each ref, compute:
      - home_whistle: home team win% in games this ref works
      - foul_rate: avg total fouls per game
      - pace_impact: avg total points per game vs league avg
      - ou_bias: avg total vs league average
    """
    # Merge ref assignments with game results
    merged = ref_df.merge(
        training_df[["game_date", "home_team", "away_team", "actual_home_score", "actual_away_score",
                      "market_spread_home", "market_ou_total"]].drop_duplicates(),
        on=["game_date", "home_team", "away_team"],
        how="left"
    )

    merged["actual_home_score"] = pd.to_numeric(merged["actual_home_score"], errors="coerce")
    merged["actual_away_score"] = pd.to_numeric(merged["actual_away_score"], errors="coerce")
    merged = merged.dropna(subset=["actual_home_score", "actual_away_score"])

    merged["home_won"] = merged["actual_home_score"] > merged["actual_away_score"]
    merged["total_pts"] = merged["actual_home_score"] + merged["actual_away_score"]
    merged["margin"] = merged["actual_home_score"] - merged["actual_away_score"]

    # League averages
    lg_home_win_rate = merged["home_won"].mean()
    lg_total_pts = merged["total_pts"].mean()

    # Per-ref profiles
    profiles = {}
    for ref_name, group in merged.groupby("ref_name"):
        if len(group) < 10:  # Need minimum sample
            continue
        profiles[ref_name] = {
            "n_games": len(group),
            "home_whistle": float(group["home_won"].mean() - lg_home_win_rate),  # Deviation from league avg
            "total_pts_avg": float(group["total_pts"].mean()),
            "ou_bias": float(group["total_pts"].mean() - lg_total_pts),
            "pace_impact": float((group["total_pts"].mean() - lg_total_pts) / lg_total_pts),
            "foul_proxy": float(group["total_pts"].mean() / lg_total_pts),  # Higher scoring ≈ more fouls
        }

    print(f"  Built profiles for {len(profiles)} referees (min 10 games)")
    print(f"    League home win rate: {lg_home_win_rate:.3f}")
    print(f"    League avg total pts: {lg_total_pts:.1f}")

    if profiles:
        # Top 5 most home-friendly refs
        sorted_refs = sorted(profiles.items(), key=lambda x: x[1]["home_whistle"], reverse=True)
        print(f"    Most home-friendly refs:")
        for name, p in sorted_refs[:5]:
            print(f"      {name}: +{p['home_whistle']:.3f} home bias, {p['n_games']} games")

    return profiles


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Player Ratings & Lineup Value Pipeline")
    print("=" * 60)

    # Check for required files
    if not os.path.exists("nba_player_boxscores.parquet"):
        print("  ERROR: nba_player_boxscores.parquet not found")
        print("  Run scrape_nba_summaries.py first")
        sys.exit(1)

    box_df = pd.read_parquet("nba_player_boxscores.parquet")
    print(f"  Loaded {len(box_df)} player-game rows")

    training_df = pd.read_parquet("nba_training_data.parquet") if os.path.exists("nba_training_data.parquet") else None

    # ── Build player ratings ──
    print("\n  Building player ratings...")
    box_df = build_player_ratings(box_df)

    # ── Build lineup features ──
    print("\n  Computing lineup features...")
    lineup_df = compute_lineup_features(box_df, training_df)
    lineup_df.to_parquet("nba_lineup_features.parquet", index=False)
    print(f"  ✅ Saved nba_lineup_features.parquet ({len(lineup_df)} games)")

    # ── Build referee profiles ──
    if os.path.exists("nba_referee_log.parquet") and training_df is not None:
        print("\n  Building referee profiles...")
        ref_df = pd.read_parquet("nba_referee_log.parquet")
        profiles = build_referee_profiles(ref_df, training_df)
        import json
        with open("nba_referee_profiles.json", "w") as f:
            json.dump(profiles, f, indent=2)
        print(f"  ✅ Saved nba_referee_profiles.json ({len(profiles)} refs)")
    else:
        print("\n  Skipping referee profiles (no ref log data)")

    # ── Merge into training data ──
    if "--merge" in sys.argv and training_df is not None:
        print("\n  Merging lineup features into training parquet...")

        # Ensure game_id types match
        if "game_id" in training_df.columns:
            training_df["game_id"] = training_df["game_id"].astype(str)
            lineup_df["game_id"] = lineup_df["game_id"].astype(str)

            merged = training_df.merge(
                lineup_df, on="game_id", how="left", suffixes=("", "_lineup")
            )
            # Fill missing lineup features with defaults
            for col in lineup_df.columns:
                if col != "game_id" and col in merged.columns:
                    merged[col] = merged[col].fillna(0)

            merged.to_parquet("nba_training_data.parquet", index=False)
            has_lineup = (merged.get("home_lineup_value", 0) > 0).sum()
            print(f"  ✅ Merged: {has_lineup}/{len(merged)} games have lineup data")
        else:
            print("  WARNING: training data has no game_id column — can't merge")
            print("  Lineup features saved separately in nba_lineup_features.parquet")
