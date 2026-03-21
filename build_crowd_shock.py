"""
Build crowd_shock_diff feature from ncaa_historical attendance data.

Logic:
- For each team, compute rolling 5-game average of home attendance (from their home games only)
- For each game, look up both teams' rolling avg home attendance
- crowd_shock_diff = home_avg_crowd / away_avg_crowd
  - Value > 1: home team plays in bigger arenas → crowd favors home
  - Value = 1: similar crowd environments
  - Value < 1: away team plays in bigger arenas (rare but possible in tournaments)

Example: Duke (avg 9.3K home crowd) hosts a mid-major (avg 2.1K) → shock = 9.3/2.1 = 4.43
The mid-major is in an environment 4.4x louder than they're used to.

For neutral sites: both teams are equally "away", so shock matters less.
The feature still captures the asymmetry in crowd familiarity.

This is a REPLACEMENT for crowd_pct (which was zeroed in training because it used
the current game's attendance — a leakage-prone, noisy signal).
"""

import numpy as np
import pandas as pd


def compute_crowd_shock(df, n_games=5):
    """
    Compute rolling 5-game avg home attendance per team, then crowd_shock_diff.
    
    Args:
        df: DataFrame with columns: game_date, home_team_id, away_team_id, attendance
        n_games: rolling window size (default 5)
    
    Returns:
        DataFrame with new columns: home_avg_crowd, away_avg_crowd, crowd_shock_diff
    """
    df = df.copy()
    
    # Ensure types
    df["attendance"] = pd.to_numeric(df.get("attendance", 0), errors="coerce").fillna(0)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["away_team_id"] = df["away_team_id"].astype(str)
    
    # Sort by date for proper rolling computation
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Build per-team rolling avg home attendance from PRIOR home games
    # Key: team_id → list of (game_date, attendance) for home games with attendance > 0
    team_home_att = {}  # team_id → deque of recent home attendances
    
    home_avg_crowd = np.zeros(len(df))
    away_avg_crowd = np.zeros(len(df))
    
    for idx, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]
        att = row["attendance"]
        
        # Look up current rolling avg for both teams (from PRIOR games)
        h_history = team_home_att.get(h_id, [])
        a_history = team_home_att.get(a_id, [])
        
        # Home team's avg home crowd (what they're used to)
        if h_history:
            home_avg_crowd[idx] = np.mean(h_history[-n_games:])
        else:
            home_avg_crowd[idx] = 0  # no prior home games yet
        
        # Away team's avg home crowd (what they're used to)
        if a_history:
            away_avg_crowd[idx] = np.mean(a_history[-n_games:])
        else:
            away_avg_crowd[idx] = 0  # no prior home games yet
        
        # Update home team's history AFTER computing (no leakage)
        if att > 0:
            if h_id not in team_home_att:
                team_home_att[h_id] = []
            team_home_att[h_id].append(att)
    
    df["home_avg_crowd"] = home_avg_crowd
    df["away_avg_crowd"] = away_avg_crowd
    
    # Compute crowd_shock_diff: ratio of home crowd to away team's normal crowd
    # When away_avg = 0 (no data), default to 1.0 (no shock)
    # When home_avg = 0 (no data), default to 1.0 (no shock)
    # Log-scale the ratio to prevent extreme values and make it symmetric:
    #   log(5.0) = 1.61 (big shock), log(1.0) = 0 (no shock), log(0.2) = -1.61
    _h = np.maximum(home_avg_crowd, 1)  # avoid div by zero
    _a = np.maximum(away_avg_crowd, 1)
    
    _has_both = (home_avg_crowd > 0) & (away_avg_crowd > 0)
    raw_ratio = np.where(_has_both, _h / _a, 1.0)
    
    # Log-transform for symmetry: log(2) = 0.69, log(5) = 1.61
    df["crowd_shock_diff"] = np.where(_has_both, np.log(raw_ratio), 0.0)
    
    # Stats
    valid = df["crowd_shock_diff"] != 0
    print(f"  crowd_shock_diff: {valid.sum()}/{len(df)} non-zero ({valid.mean()*100:.1f}%)")
    print(f"    mean={df.loc[valid, 'crowd_shock_diff'].mean():.3f}, "
          f"std={df.loc[valid, 'crowd_shock_diff'].std():.3f}, "
          f"min={df.loc[valid, 'crowd_shock_diff'].min():.3f}, "
          f"max={df.loc[valid, 'crowd_shock_diff'].max():.3f}")
    print(f"    Examples of high shock (away team in unfamiliar big arena):")
    top = df.loc[valid].nlargest(5, "crowd_shock_diff")[["home_team_id", "away_team_id", "home_avg_crowd", "away_avg_crowd", "crowd_shock_diff"]]
    print(top.to_string(index=False))
    
    return df


def compute_crowd_shock_for_prediction(home_team_id, away_team_id, historical_rows):
    """
    Compute crowd_shock_diff for a single prediction from historical game data.
    
    Args:
        home_team_id: str, ESPN team ID
        away_team_id: str, ESPN team ID  
        historical_rows: list of dicts with keys: home_team_id, attendance
            (team's completed home games this season, sorted by date)
    
    Returns:
        float: crowd_shock_diff value
    """
    h_id = str(home_team_id)
    a_id = str(away_team_id)
    
    # Extract last 5 home game attendances for each team
    h_atts = [r["attendance"] for r in historical_rows 
              if str(r.get("home_team_id")) == h_id and r.get("attendance", 0) > 0]
    a_atts = [r["attendance"] for r in historical_rows 
              if str(r.get("home_team_id")) == a_id and r.get("attendance", 0) > 0]
    
    h_avg = np.mean(h_atts[-5:]) if h_atts else 0
    a_avg = np.mean(a_atts[-5:]) if a_atts else 0
    
    if h_avg > 0 and a_avg > 0:
        return float(np.log(h_avg / a_avg))
    return 0.0


if __name__ == "__main__":
    # Quick test with synthetic data
    data = {
        "game_date": pd.date_range("2025-11-10", periods=20, freq="3D"),
        "home_team_id": ["A","B","C","A","B","C","A","B","C","A",
                         "B","C","A","B","C","A","B","C","A","B"],
        "away_team_id": ["B","C","A","C","A","B","B","C","A","C",
                         "A","B","B","C","A","C","A","B","B","C"],
        "attendance":   [15000,2000,5000,14000,2500,4800,15500,1800,5200,14200,
                         2300,5100,15800,2100,4900,14500,2200,5300,15200,2400],
    }
    df = pd.DataFrame(data)
    result = compute_crowd_shock(df)
    print("\nSample output:")
    print(result[["home_team_id","away_team_id","home_avg_crowd","away_avg_crowd","crowd_shock_diff"]].to_string())
