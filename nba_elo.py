"""
NBA Elo Rating System
Computes Elo ratings from game results in nba_historical + nba_predictions.
Used by both training (historical Elo at game time) and live prediction.

Design:
  - K=20 base, scaled by margin of victory (MOV multiplier from FiveThirtyEight)
  - Home advantage = +65 Elo points (~3.5 pts spread equivalent)
  - Season carryover: regress 1/3 toward 1500 between seasons
  - Outputs: per-game Elo snapshot for both teams BEFORE the game was played

Usage:
  from nba_elo import compute_all_elo, get_current_elo

  # Training: get Elo for every historical game
  elo_df = compute_all_elo()  # Returns DataFrame with game_date, home_team, away_team, home_elo, away_elo

  # Live: get current Elo for a team
  ratings = get_current_elo()  # Returns {team_abbr: elo_rating}
"""

import numpy as np
import pandas as pd
import os, sys, json, time

# ── Configuration ──
BASE_ELO = 1500
K_FACTOR = 20.0
HCA_ELO = 65       # ~3.5 points of home advantage in Elo space
SEASON_REVERSION = 1/3  # Regress 1/3 toward mean between seasons
MOV_EXPONENT = 0.8      # Margin-of-victory dampening (FiveThirtyEight-style)


def _mov_multiplier(margin, elo_diff):
    """
    Margin-of-victory multiplier for K-factor.
    Bigger wins = bigger Elo changes, but with diminishing returns.
    Also accounts for expected blowouts (autocorrelation correction).
    Based on FiveThirtyEight's NBA Elo methodology.
    """
    abs_margin = abs(margin)
    # Autocorrelation correction: reduce K for expected blowouts
    # If the favorite wins big, that's less informative than an upset blowout
    corr = 2.2 / ((elo_diff * 0.001) + 2.2) if elo_diff > 0 else 1.0
    return np.log(max(abs_margin, 1) + 1) * corr


def _expected_score(elo_a, elo_b):
    """Expected win probability for team A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _update_elo(winner_elo, loser_elo, margin, k=K_FACTOR):
    """
    Update Elo ratings after a game.
    Returns (new_winner_elo, new_loser_elo).
    """
    expected_w = _expected_score(winner_elo, loser_elo)
    mov_mult = _mov_multiplier(margin, winner_elo - loser_elo)
    k_adj = k * mov_mult
    
    new_winner = winner_elo + k_adj * (1.0 - expected_w)
    new_loser = loser_elo + k_adj * (0.0 - (1.0 - expected_w))
    
    return round(new_winner, 1), round(new_loser, 1)


def _season_carryover(elo):
    """Regress Elo toward base between seasons."""
    return round(elo * (1 - SEASON_REVERSION) + BASE_ELO * SEASON_REVERSION, 1)


def compute_all_elo(games_df=None, save_path=None):
    """
    Compute Elo ratings for every game in chronological order.
    
    Args:
        games_df: DataFrame with columns [game_date, season, home_team, away_team, 
                  actual_home_score, actual_away_score]. If None, loads from Supabase.
        save_path: If provided, saves Elo snapshots to this path (JSON or parquet).
    
    Returns:
        DataFrame with original columns + home_elo, away_elo, elo_diff (before game)
    """
    if games_df is None:
        games_df = _load_all_games()
    
    df = games_df.copy()
    
    # Ensure numeric
    for col in ["actual_home_score", "actual_away_score", "season"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Filter to completed games with scores
    df = df.dropna(subset=["actual_home_score", "actual_away_score", "home_team", "away_team"])
    df = df.sort_values("game_date").reset_index(drop=True)
    
    if len(df) == 0:
        print("  WARNING: No completed games for Elo computation")
        return pd.DataFrame()
    
    # Initialize ratings
    ratings = {}
    elo_records = []
    
    prev_season = None
    
    for idx, row in df.iterrows():
        season = int(row.get("season", 0))
        home = row["home_team"]
        away = row["away_team"]
        h_score = float(row["actual_home_score"])
        a_score = float(row["actual_away_score"])
        
        # Season carryover
        if prev_season is not None and season != prev_season and season > prev_season:
            for team in list(ratings.keys()):
                ratings[team] = _season_carryover(ratings[team])
        prev_season = season
        
        # Initialize if new team
        if home not in ratings:
            ratings[home] = BASE_ELO
        if away not in ratings:
            ratings[away] = BASE_ELO
        
        # Snapshot BEFORE the game (this is what the model uses)
        home_elo = ratings[home]
        away_elo = ratings[away]
        
        elo_records.append({
            "game_date": row["game_date"],
            "home_team": home,
            "away_team": away,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": round(home_elo - away_elo, 1),
        })
        
        # Update ratings based on result
        margin = h_score - a_score
        
        # Add HCA to home Elo for expected score calculation
        home_elo_adj = home_elo + HCA_ELO
        
        if margin > 0:
            # Home wins
            new_home, new_away = _update_elo(home_elo_adj, away_elo, margin)
            # Remove HCA from updated rating
            ratings[home] = round(new_home - HCA_ELO + home_elo_adj - home_elo_adj, 1)
            # Actually, let's do this more cleanly:
            expected_h = _expected_score(home_elo_adj, away_elo)
            mov_mult = _mov_multiplier(margin, home_elo_adj - away_elo)
            k_adj = K_FACTOR * mov_mult
            ratings[home] = round(home_elo + k_adj * (1.0 - expected_h), 1)
            ratings[away] = round(away_elo + k_adj * (0.0 - (1.0 - expected_h)), 1)
        elif margin < 0:
            # Away wins
            expected_h = _expected_score(home_elo_adj, away_elo)
            mov_mult = _mov_multiplier(abs(margin), away_elo - home_elo_adj)
            k_adj = K_FACTOR * mov_mult
            ratings[home] = round(home_elo + k_adj * (0.0 - expected_h), 1)
            ratings[away] = round(away_elo + k_adj * (1.0 - (1.0 - expected_h)), 1)
        # else: tie (shouldn't happen in NBA, but handle gracefully — no update)
    
    elo_df = pd.DataFrame(elo_records)
    
    # Stats
    if len(elo_df) > 0:
        print(f"  NBA Elo: {len(elo_df)} games processed")
        print(f"    Rating range: [{elo_df['home_elo'].min():.0f}, {elo_df['home_elo'].max():.0f}]")
        print(f"    elo_diff std: {elo_df['elo_diff'].std():.1f}")
        
        # Predictive check: does higher Elo win more?
        merged = df.reset_index(drop=True).copy()
        merged["elo_diff"] = elo_df["elo_diff"].values
        merged["home_won"] = merged["actual_home_score"] > merged["actual_away_score"]
        merged["elo_predicted_home"] = merged["elo_diff"] > 0
        elo_accuracy = (merged["home_won"] == merged["elo_predicted_home"]).mean()
        print(f"    Elo straight-up accuracy: {elo_accuracy:.1%}")
    
    if save_path:
        if save_path.endswith(".parquet"):
            elo_df.to_parquet(save_path, index=False)
        else:
            # Save current ratings as JSON
            with open(save_path, "w") as f:
                json.dump({"ratings": ratings, "n_games": len(elo_df),
                           "last_updated": str(pd.Timestamp.now())}, f, indent=2)
        print(f"    Saved to {save_path}")
    
    return elo_df, ratings


def get_current_elo(ratings_path="nba_elo_ratings.json"):
    """
    Load current Elo ratings from saved file.
    Returns dict {team_abbr: elo_rating}.
    """
    if os.path.exists(ratings_path):
        with open(ratings_path) as f:
            data = json.load(f)
        return data.get("ratings", {})
    return {}


def merge_elo_into_df(df, elo_df):
    """
    Merge pre-game Elo ratings into a games DataFrame.
    Matches on game_date + home_team + away_team.
    """
    if elo_df is None or len(elo_df) == 0:
        df["home_elo"] = BASE_ELO
        df["away_elo"] = BASE_ELO
        df["elo_diff"] = 0.0
        return df
    
    merged = df.merge(
        elo_df[["game_date", "home_team", "away_team", "home_elo", "away_elo", "elo_diff"]],
        on=["game_date", "home_team", "away_team"],
        how="left"
    )
    # Fill any unmatched games with base Elo
    merged["home_elo"] = merged["home_elo"].fillna(BASE_ELO)
    merged["away_elo"] = merged["away_elo"].fillna(BASE_ELO)
    merged["elo_diff"] = merged["elo_diff"].fillna(0.0)
    
    return merged


def _load_all_games():
    """Load all completed games from Supabase (historical + current predictions)."""
    try:
        from db import sb_get
    except ImportError:
        # Running locally — use requests directly
        import requests
        SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
        KEY = os.environ.get("SUPABASE_ANON_KEY", "")
        
        def sb_get(table, params=""):
            all_data, offset, limit = [], 0, 1000
            while True:
                sep = "&" if params else ""
                url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
                r = requests.get(url, headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, timeout=60)
                if not r.ok:
                    break
                data = r.json()
                if not data:
                    break
                all_data.extend(data)
                if len(data) < limit:
                    break
                offset += limit
            return all_data
    
    # Historical games
    hist_rows = sb_get(
        "nba_historical",
        "is_outlier_season=eq.false&actual_home_score=not.is.null"
        "&select=game_date,season,home_team,away_team,actual_home_score,actual_away_score"
        "&order=game_date.asc"
    )
    
    # Current season predictions with results
    pred_rows = sb_get(
        "nba_predictions",
        "result_entered=eq.true&actual_home_score=not.is.null"
        "&select=game_date,home_team,away_team,actual_home_score,actual_away_score"
    )
    
    hist_df = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame()
    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
    
    # Add season to predictions if missing
    if len(pred_df) > 0 and "season" not in pred_df.columns:
        pred_df["season"] = pred_df["game_date"].apply(
            lambda d: int(d[:4]) + 1 if int(d[5:7]) >= 10 else int(d[:4])
        )
    
    if len(hist_df) > 0 and len(pred_df) > 0:
        combined = pd.concat([hist_df, pred_df], ignore_index=True)
    elif len(hist_df) > 0:
        combined = hist_df
    else:
        combined = pred_df
    
    # Deduplicate
    combined = combined.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)
    
    print(f"  Loaded {len(combined)} total games for Elo ({len(hist_df)} historical + {len(pred_df)} current)")
    return combined


# ── CLI: python nba_elo.py ──
if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Elo Rating System")
    print("=" * 60)
    
    elo_df, ratings = compute_all_elo(save_path="nba_elo_ratings.json")
    
    # Also save the per-game snapshots for training
    if elo_df is not None and len(elo_df) > 0:
        elo_df.to_parquet("nba_elo_snapshots.parquet", index=False)
        print(f"\n  Saved {len(elo_df)} Elo snapshots to nba_elo_snapshots.parquet")
    
    # Print current top/bottom teams
    if ratings:
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        print("\n  Current NBA Elo Ratings:")
        print("  " + "-" * 30)
        for team, elo in sorted_teams[:10]:
            print(f"    {team:4s}  {elo:.0f}")
        print("    ...")
        for team, elo in sorted_teams[-5:]:
            print(f"    {team:4s}  {elo:.0f}")
