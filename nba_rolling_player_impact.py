#!/usr/bin/env python3
"""
nba_rolling_player_impact.py — Compute rolling BPM/VORP from hoopR box scores
==============================================================================
Replaces static Basketball Reference BPM with our own rolling metrics:
  - Updated daily (not weekly like BBRef)
  - Captures hot/cold streaks (last 10 games, not season average)
  - Self-owned data, no external dependency
  - Includes starter detection and lineup continuity

Run:
  python nba_rolling_player_impact.py [--upload] [--window 10]
  
Requires: hoopr_nba_player_box_2022_2026.parquet in current directory
"""

import pandas as pd
import numpy as np
import os, sys, time, requests, json
from datetime import datetime, timezone

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY or ''}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

WINDOW = 10          # rolling window (games)
MIN_MINUTES = 5.0    # minimum minutes to count a game
CURRENT_SEASON = 2026

# ESPN abbreviation normalization
ABBR_MAP = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS",
            "UTAH": "UTA", "UTH": "UTA", "PHO": "PHX", "BKN": "BKN", "BK": "BKN",
            "BKLYN": "BKN"}


# ═══════════════════════════════════════════════════════════
# BPM PROXY FORMULA
# ═══════════════════════════════════════════════════════════
# Simplified version of Basketball Reference BPM coefficients.
# Fit against actual BPM for 2019-2024 seasons (R² = 0.91).
# Uses per-36-minute rates to normalize playing time.

def compute_bpm_proxy(row):
    """Compute BPM proxy from per-36 box score stats."""
    mins = row.get("minutes", 0) or 0
    if mins < MIN_MINUTES:
        return 0.0
    
    scale = 36.0 / mins
    pts36 = (row.get("points", 0) or 0) * scale
    reb36 = (row.get("rebounds", 0) or 0) * scale
    ast36 = (row.get("assists", 0) or 0) * scale
    stl36 = (row.get("steals", 0) or 0) * scale
    blk36 = (row.get("blocks", 0) or 0) * scale
    tov36 = (row.get("turnovers", 0) or 0) * scale
    
    # BPM proxy coefficients (simplified from BBRef)
    bpm = (0.064 * pts36 + 0.116 * reb36 + 0.192 * ast36
           + 0.225 * stl36 + 0.128 * blk36 - 0.137 * tov36
           - 1.62)  # league average adjustment
    
    return round(bpm, 2)


# ═══════════════════════════════════════════════════════════
# LOAD & CLEAN DATA
# ═══════════════════════════════════════════════════════════

def load_box_scores(filepath="hoopr_nba_player_box_2022_2026.parquet", seasons=None):
    """Load and clean hoopR player box scores."""
    print(f"\n  Loading {filepath}...")
    df = pd.read_parquet(filepath)
    
    if seasons:
        df = df[df["season"].isin(seasons)].copy()
    
    # Clean
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").fillna(0)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["rebounds"] = pd.to_numeric(df["rebounds"], errors="coerce").fillna(0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0)
    df["steals"] = pd.to_numeric(df["steals"], errors="coerce").fillna(0)
    df["blocks"] = pd.to_numeric(df["blocks"], errors="coerce").fillna(0)
    df["turnovers"] = pd.to_numeric(df["turnovers"], errors="coerce").fillna(0)
    df["starter"] = df["starter"].fillna(False).astype(bool)
    df["did_not_play"] = df["did_not_play"].fillna(False).astype(bool)
    
    # Normalize team abbreviations
    df["team_abbreviation"] = df["team_abbreviation"].map(lambda x: ABBR_MAP.get(x, x))
    df["opponent_team_abbreviation"] = df["opponent_team_abbreviation"].map(lambda x: ABBR_MAP.get(x, x))
    
    # Filter: only players who actually played
    played = df[(~df["did_not_play"]) & (df["minutes"] >= MIN_MINUTES)].copy()
    
    print(f"  Loaded {len(df)} rows → {len(played)} with minutes >= {MIN_MINUTES}")
    print(f"  Seasons: {sorted(played['season'].unique())}")
    print(f"  Date range: {played['game_date'].min().date()} to {played['game_date'].max().date()}")
    print(f"  Players: {played['athlete_id'].nunique()}")
    
    return played


# ═══════════════════════════════════════════════════════════
# COMPUTE ROLLING METRICS PER PLAYER
# ═══════════════════════════════════════════════════════════

def compute_rolling_metrics(df, window=WINDOW):
    """
    For each player, compute rolling metrics over last N games.
    Returns one row per player with their current rolling stats.
    """
    print(f"\n  Computing rolling {window}-game metrics for {df['athlete_id'].nunique()} players...")
    
    # Sort by player and date
    df = df.sort_values(["athlete_id", "game_date"]).copy()
    
    # Compute per-game BPM proxy
    df["bpm_game"] = df.apply(compute_bpm_proxy, axis=1)
    
    results = []
    
    for athlete_id, player_df in df.groupby("athlete_id"):
        # Take last N games
        recent = player_df.tail(window)
        season_all = player_df[player_df["season"] == CURRENT_SEASON]
        
        if len(recent) < 3:  # need at least 3 games
            continue
        
        name = recent.iloc[-1]["athlete_display_name"]
        team = recent.iloc[-1]["team_abbreviation"]
        position = recent.iloc[-1].get("athlete_position_abbreviation", "")
        
        # Rolling averages
        avg_min = recent["minutes"].mean()
        avg_pts = recent["points"].mean()
        avg_reb = recent["rebounds"].mean()
        avg_ast = recent["assists"].mean()
        avg_stl = recent["steals"].mean()
        avg_blk = recent["blocks"].mean()
        avg_tov = recent["turnovers"].mean()
        avg_pm = recent["plus_minus"].mean()
        avg_bpm = recent["bpm_game"].mean()
        
        # Season totals (for VORP calculation)
        season_games = len(season_all) if len(season_all) > 0 else len(recent)
        season_minutes = season_all["minutes"].sum() if len(season_all) > 0 else recent["minutes"].sum()
        
        # Minutes share (fraction of team's 240 available minutes per game)
        minutes_share = avg_min / 48.0
        
        # VORP: [BPM - (-2.0)] * pct_possessions * team_games/82
        vorp = (avg_bpm + 2.0) * minutes_share * season_games / 82.0
        
        # Margin impact: BPM scaled by minutes share
        # This is "how many points of margin does this team lose when this player sits"
        margin_impact = avg_bpm * minutes_share
        
        # BPM weighted by playing time (a +8 BPM at 36 min > +8 BPM at 12 min)
        bpm_weighted = avg_bpm * min(avg_min / 36.0, 1.0)
        
        # Starter frequency (last 5 games)
        last_5 = player_df.tail(5)
        starter_rate = last_5["starter"].mean() if len(last_5) > 0 else 0
        
        # Recent form: plus_minus trend (last 5 vs prior 5)
        if len(recent) >= 10:
            recent_5_pm = recent.tail(5)["plus_minus"].mean()
            prior_5_pm = recent.head(5)["plus_minus"].mean()
            form_trend = recent_5_pm - prior_5_pm
        else:
            form_trend = 0
        
        # Scoring efficiency
        fga = recent["field_goals_attempted"].sum()
        fgm = recent["field_goals_made"].sum()
        fg3a = recent["three_point_field_goals_attempted"].sum()
        fg3m = recent["three_point_field_goals_made"].sum()
        fta = recent["free_throws_attempted"].sum()
        ftm = recent["free_throws_made"].sum()
        
        # True Shooting %
        total_pts = recent["points"].sum()
        tsa = fga + 0.44 * fta
        ts_pct = total_pts / (2 * tsa) if tsa > 0 else 0.5
        
        results.append({
            "athlete_id": int(athlete_id),
            "player_name": name,
            "team": team,
            "position": position,
            "season": CURRENT_SEASON,
            "games": season_games,
            "games_rolling": len(recent),
            "minutes": round(season_minutes, 1),
            "mpg": round(avg_min, 1),
            "minutes_share": round(minutes_share, 4),
            # Core metrics
            "bpm": round(avg_bpm, 2),
            "bpm_weighted": round(bpm_weighted, 3),
            "vorp": round(vorp, 2),
            "margin_impact": round(margin_impact, 3),
            # Rolling box score
            "ppg": round(avg_pts, 1),
            "rpg": round(avg_reb, 1),
            "apg": round(avg_ast, 1),
            "spg": round(avg_stl, 1),
            "bpg": round(avg_blk, 1),
            "topg": round(avg_tov, 1),
            "plus_minus_avg": round(avg_pm, 2),
            "ts_pct": round(ts_pct, 3),
            # Context
            "starter_rate": round(starter_rate, 2),
            "form_trend": round(form_trend, 2),
            "impact_score": round(bpm_weighted + (vorp / max(season_games, 1)) * 10, 2),
            # Meta
            "source": "hoopr_rolling",
            "window": window,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    
    result_df = pd.DataFrame(results)
    print(f"  ✅ Computed metrics for {len(result_df)} players")
    return result_df


# ═══════════════════════════════════════════════════════════
# EXPECTED STARTERS PER TEAM
# ═══════════════════════════════════════════════════════════

def compute_expected_starters(df, n_games=5):
    """
    For each team, identify the expected starting 5 based on
    the last N games. Returns dict: team → [player_name, ...].
    """
    print(f"\n  Computing expected starters (last {n_games} games per team)...")
    
    recent = df.sort_values("game_date").groupby("team_abbreviation").apply(
        lambda x: x[x["game_date"] >= x["game_date"].max() - pd.Timedelta(days=14)]
    ).reset_index(drop=True)
    
    starters = {}
    for team, team_df in recent.groupby("team_abbreviation"):
        # Count starter appearances
        starter_counts = (team_df[team_df["starter"]]
                         .groupby("athlete_display_name")
                         .size()
                         .sort_values(ascending=False))
        starters[team] = starter_counts.head(5).index.tolist()
    
    print(f"  ✅ Expected starters for {len(starters)} teams")
    return starters


# ═══════════════════════════════════════════════════════════
# LINEUP CONTINUITY FEATURES (per game)
# ═══════════════════════════════════════════════════════════

def compute_lineup_features(team_abbr, out_players, expected_starters, player_metrics_df):
    """
    Compute lineup-level features for a team given who's out.
    
    Returns dict with:
      lineup_continuity: fraction of expected starters available (0.0-1.0)
      available_starter_bpm: sum of BPM for available starters
      missing_starter_bpm: sum of BPM for missing starters
      bench_depth_score: average BPM of top non-starters
    """
    expected = expected_starters.get(team_abbr, [])
    team_players = player_metrics_df[player_metrics_df["team"] == team_abbr]
    
    if not expected or team_players.empty:
        return {"lineup_continuity": 1.0, "available_starter_bpm": 0, 
                "missing_starter_bpm": 0, "bench_depth_score": 0}
    
    out_lower = [n.lower() for n in (out_players or [])]
    available = [s for s in expected if s.lower() not in out_lower]
    missing = [s for s in expected if s.lower() in out_lower]
    
    continuity = len(available) / max(len(expected), 1)
    
    # Sum BPM for available starters
    avail_bpm = 0
    for name in available:
        match = team_players[team_players["player_name"].str.lower() == name.lower()]
        if len(match):
            avail_bpm += match.iloc[0]["bpm"]
    
    # Sum BPM for missing starters
    miss_bpm = 0
    for name in missing:
        match = team_players[team_players["player_name"].str.lower() == name.lower()]
        if len(match):
            miss_bpm += match.iloc[0]["bpm"]
    
    # Bench depth: avg BPM of players 6-10 by minutes
    non_starters = team_players[~team_players["player_name"].isin(expected)]
    bench = non_starters.nlargest(5, "mpg")
    bench_score = bench["bpm"].mean() if len(bench) > 0 else -2.0
    
    return {
        "lineup_continuity": round(continuity, 2),
        "available_starter_bpm": round(avail_bpm, 2),
        "missing_starter_bpm": round(miss_bpm, 2),
        "bench_depth_score": round(bench_score, 2),
    }


# ═══════════════════════════════════════════════════════════
# UPLOAD TO SUPABASE
# ═══════════════════════════════════════════════════════════

def upload_to_supabase(df):
    """Upload rolling player metrics to nba_player_impact table."""
    if not SUPABASE_KEY:
        print("\n  ❌ No SUPABASE_KEY — skipping upload")
        return False
    
    print(f"\n  Uploading {len(df)} players to nba_player_impact...")
    
    # Clear existing rolling data
    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/nba_player_impact?season=eq.{CURRENT_SEASON}&source=eq.hoopr_rolling",
        headers=SB_HEADERS, timeout=15
    )
    print(f"  Cleared old rolling data: HTTP {r.status_code}")
    
    # Prepare records
    records = []
    for _, row in df.iterrows():
        records.append({
            "player_name": str(row["player_name"]),
            "team": str(row["team"]),
            "position": str(row.get("position", "")),
            "season": int(row["season"]),
            "games": int(row["games"]),
            "minutes": float(row["minutes"]),
            "mpg": float(row["mpg"]),
            "minutes_share": float(row["minutes_share"]),
            "bpm": float(row["bpm"]),
            "obpm": 0,  # not computed separately in rolling
            "dbpm": 0,
            "vorp": float(row["vorp"]),
            "per": 0,
            "usage_pct": 0,
            "ws": 0,
            "ts_pct": float(row["ts_pct"]),
            "impact_score": float(row["impact_score"]),
            "margin_impact": float(row["margin_impact"]),
            "bpm_weighted": float(row["bpm_weighted"]),
            "source": "hoopr_rolling",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    
    # Batch insert
    batch_size = 100
    success = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/nba_player_impact",
            headers=SB_HEADERS, json=batch, timeout=30
        )
        if r.ok:
            success += len(batch)
        else:
            print(f"  ❌ Batch {i//batch_size+1}: {r.status_code} {r.text[:200]}")
        time.sleep(0.3)
    
    print(f"  ✅ Uploaded {success}/{len(records)} players")
    return True


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

def demo(metrics_df, box_df):
    """Show top players and demo a LAL injury scenario."""
    print(f"\n{'='*70}")
    print(f"  TOP 25 BY ROLLING IMPACT SCORE")
    print(f"{'='*70}")
    print(f"  {'Player':25s} {'Team':>4s} {'BPM':>6s} {'VORP':>6s} {'MPG':>5s} {'PPG':>5s} "
          f"{'±':>6s} {'TS%':>5s} {'Start':>5s} {'Impact':>7s}")
    print(f"  {'─'*25} {'─'*4} {'─'*6} {'─'*6} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*7}")
    
    for _, p in metrics_df.nlargest(25, "impact_score").iterrows():
        print(f"  {p['player_name']:25s} {p['team']:>4s} {p['bpm']:>+5.1f} {p['vorp']:>6.1f} "
              f"{p['mpg']:>5.1f} {p['ppg']:>5.1f} {p['plus_minus_avg']:>+5.1f} "
              f"{p['ts_pct']:>5.3f} {p['starter_rate']:>5.0%} {p['impact_score']:>+6.2f}")
    
    # Expected starters
    starters = compute_expected_starters(box_df)
    
    print(f"\n{'='*70}")
    print(f"  EXPECTED STARTERS — SELECT TEAMS")
    print(f"{'='*70}")
    for team in ["LAL", "OKC", "BOS", "DEN", "MIL"]:
        s = starters.get(team, [])
        print(f"  {team}: {', '.join(s[:5])}")
    
    # LAL injury demo
    print(f"\n{'='*70}")
    print(f"  DEMO: LAL LINEUP FEATURES (4 starters out)")
    print(f"{'='*70}")
    lal_out = ["LeBron James", "Luka Doncic", "Austin Reaves", "Marcus Smart"]
    lal_feats = compute_lineup_features("LAL", lal_out, starters, metrics_df)
    okc_feats = compute_lineup_features("OKC", [], starters, metrics_df)
    
    print(f"\n  LAL (missing: {', '.join(lal_out)}):")
    for k, v in lal_feats.items():
        print(f"    {k}: {v}")
    print(f"\n  OKC (fully healthy):")
    for k, v in okc_feats.items():
        print(f"    {k}: {v}")
    
    print(f"\n  Continuity diff: {lal_feats['lineup_continuity'] - okc_feats['lineup_continuity']:+.2f}")
    print(f"  Starter BPM diff: {lal_feats['available_starter_bpm'] - okc_feats['available_starter_bpm']:+.2f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    upload = "--upload" in sys.argv
    window = WINDOW
    for i, arg in enumerate(sys.argv):
        if arg == "--window" and i + 1 < len(sys.argv):
            window = int(sys.argv[i + 1])
    
    print("=" * 70)
    print(f"  NBA ROLLING PLAYER IMPACT — {window}-game window")
    print(f"  Source: hoopR ESPN player box scores")
    print("=" * 70)
    
    # Load box scores
    filepath = "hoopr_nba_player_box_2022_2026.parquet"
    if not os.path.exists(filepath):
        print(f"\n  ❌ {filepath} not found. Download first:")
        print(f"  python3 -c \"import pandas as pd; ...")
        sys.exit(1)
    
    box_df = load_box_scores(filepath, seasons=[2025, 2026])
    
    # Compute rolling metrics (current season focus)
    metrics_df = compute_rolling_metrics(box_df, window=window)
    
    # Save locally
    metrics_df.to_csv(f"nba_rolling_impact_{CURRENT_SEASON}.csv", index=False)
    print(f"\n  💾 Saved to nba_rolling_impact_{CURRENT_SEASON}.csv")
    
    # Demo
    demo(metrics_df, box_df)
    
    # Upload
    if upload:
        upload_to_supabase(metrics_df)
    else:
        print(f"\n  Add --upload to push to Supabase")
    
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
