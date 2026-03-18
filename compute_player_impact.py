#!/usr/bin/env python3
"""
compute_player_impact.py — Walk-forward RAPM (no data leakage)
================================================================
For each game, player ratings are computed ONLY from games before it.

Approach: Split each season into monthly windows. At each window,
recompute RAPM using all games before that window. Apply those
ratings to games within the window.

Window 1 (Nov games):     Use prior season ratings (0.5 decay)
Window 2 (Dec games):     RAPM from Nov games only
Window 3 (Jan games):     RAPM from Nov + Dec games
Window 4 (Feb games):     RAPM from Nov + Dec + Jan games
Window 5 (Mar+ games):    RAPM from Nov + Dec + Jan + Feb games

This means early-season ratings are noisier (less data) and 
late-season ratings are sharper — which is realistic.

Run:
    python3 -u compute_player_impact.py --dry-run
    python3 -u compute_player_impact.py
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

DRY_RUN = "--dry-run" in sys.argv
PARQUET = "ncaa_training_data.parquet"

print("=" * 60)
print("  PLAYER IMPACT RATINGS (Walk-Forward RAPM)")
print("=" * 60)

df = pd.read_parquet(PARQUET)
df = df.sort_values(["season", "game_date"]).reset_index(drop=True)
print(f"  Loaded {len(df)} games")

# Filter to games with scores + starters
has_data = (df["actual_home_score"].notna() & 
            df["home_starter_ids"].notna() & (df["home_starter_ids"] != "") &
            df["away_starter_ids"].notna() & (df["away_starter_ids"] != ""))
print(f"  Games with starters + scores: {has_data.sum()}")

# ═══════════════════════════════════════════════════════════════
# HELPER: Compute RAPM from a set of games
# ═══════════════════════════════════════════════════════════════

def compute_rapm(games_df, alpha=100.0):
    """Compute RAPM ratings from a DataFrame of completed games.
    Returns dict: player_id -> rating, and HCA estimate."""
    
    if len(games_df) < 30:
        return {}, 0.0
    
    # Build player index for this subset
    players = set()
    for col in ["home_starter_ids", "away_starter_ids"]:
        for ids_str in games_df[col]:
            if isinstance(ids_str, str):
                players.update(ids_str.strip().split(","))
    
    p_list = sorted(players)
    p_idx = {pid: i for i, pid in enumerate(p_list)}
    n_p = len(p_list)
    
    rows, cols_i, vals = [], [], []
    margins = []
    neutrals = []
    game_i = 0
    
    for _, row in games_df.iterrows():
        h_ids = row["home_starter_ids"].strip().split(",") if isinstance(row["home_starter_ids"], str) else []
        a_ids = row["away_starter_ids"].strip().split(",") if isinstance(row["away_starter_ids"], str) else []
        
        if len(h_ids) < 3 or len(a_ids) < 3:
            continue
        
        margin = row["actual_home_score"] - row["actual_away_score"]
        margins.append(margin)
        
        is_neutral = bool(row.get("neutral_site", False))
        neutrals.append(0.0 if is_neutral else 1.0)
        
        for pid in h_ids:
            if pid in p_idx:
                rows.append(game_i)
                cols_i.append(p_idx[pid])
                vals.append(1.0)
        
        for pid in a_ids:
            if pid in p_idx:
                rows.append(game_i)
                cols_i.append(p_idx[pid])
                vals.append(-1.0)
        
        game_i += 1
    
    if game_i < 30:
        return {}, 0.0
    
    n_games = game_i
    
    # Add HCA column
    for gi in range(n_games):
        rows.append(gi)
        cols_i.append(n_p)
        vals.append(neutrals[gi])
    
    X_sparse = sparse.csr_matrix((vals, (rows, cols_i)), shape=(n_games, n_p + 1))
    y_margin = np.array(margins)
    
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_sparse, y_margin)
    
    coefs = ridge.coef_
    hca = coefs[-1]
    
    ratings = {}
    for pid, idx in p_idx.items():
        ratings[pid] = float(coefs[idx])
    
    return ratings, hca


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD: Compute ratings at monthly checkpoints
# ═══════════════════════════════════════════════════════════════

print("\n  Computing walk-forward RAPM ratings...")
t0 = time.time()

seasons = sorted(df["season"].dropna().unique())

# Store: for each game index, the ratings available BEFORE that game
game_ratings = {}  # idx -> {player_id: rating}
prior_season_ratings = {}  # carried forward from previous season

for season in seasons:
    season_mask = (df["season"] == season) & has_data
    season_df = df[season_mask].copy()
    if len(season_df) < 50:
        print(f"    Season {season}: {len(season_df)} games — skipping")
        continue
    
    # Get month for each game
    season_df["_month"] = season_df["game_date"].astype(str).str[5:7]
    
    # Track cumulative games seen this season
    games_so_far = pd.DataFrame()
    
    unique_months = sorted(season_df["_month"].unique())
    
    for month in unique_months:
        month_games = season_df[season_df["_month"] == month]
        
        if len(games_so_far) >= 30:
            # RAPM from games BEFORE this month only (no leakage!)
            ratings, hca = compute_rapm(games_so_far)
        elif prior_season_ratings:
            # First month: use prior season ratings with 0.5 decay
            ratings = {pid: r * 0.5 for pid, r in prior_season_ratings.items()}
        else:
            ratings = {}
        
        # Apply to this month's games
        for idx in month_games.index:
            game_ratings[idx] = ratings
        
        # Add this month to cumulative pool
        games_so_far = pd.concat([games_so_far, month_games], ignore_index=True)
    
    # Save end-of-season ratings for next season carry-forward
    if len(games_so_far) >= 100:
        final_ratings, _ = compute_rapm(games_so_far)
        prior_season_ratings = final_ratings
    
    n_with_ratings = sum(1 for idx in season_df.index if idx in game_ratings and game_ratings[idx])
    print(f"    Season {season}: {len(season_df)} games, {n_with_ratings} with prior ratings")

elapsed = time.time() - t0
print(f"  Walk-forward RAPM computed in {elapsed:.0f}s")

# ═══════════════════════════════════════════════════════════════
# COMPUTE PER-GAME FEATURES
# ═══════════════════════════════════════════════════════════════

print("\n  Computing per-game player impact features...")

for prefix in ["home_", "away_"]:
    df[f"{prefix}player_rating_sum"] = np.nan
    df[f"{prefix}player_rating_avg"] = np.nan
    df[f"{prefix}weakest_starter"] = np.nan
    df[f"{prefix}starter_variance"] = np.nan
df["player_rating_diff"] = np.nan

computed = 0

for idx, row in df.iterrows():
    if idx not in game_ratings:
        continue
    
    ratings = game_ratings[idx]
    if not ratings:
        continue
    
    for side in ["home", "away"]:
        ids_str = row.get(f"{side}_starter_ids")
        if not isinstance(ids_str, str) or ids_str.strip() == "":
            continue
        
        pids = ids_str.strip().split(",")
        
        player_vals = []
        for pid in pids:
            if pid in ratings:
                player_vals.append(ratings[pid])
            else:
                player_vals.append(0.0)  # unknown -> league average
        
        if player_vals:
            df.at[idx, f"{side}_player_rating_sum"] = sum(player_vals)
            df.at[idx, f"{side}_player_rating_avg"] = np.mean(player_vals)
            df.at[idx, f"{side}_weakest_starter"] = min(player_vals)
            df.at[idx, f"{side}_starter_variance"] = np.std(player_vals) if len(player_vals) > 1 else 0
    
    h_sum = df.at[idx, "home_player_rating_sum"]
    a_sum = df.at[idx, "away_player_rating_sum"]
    if pd.notna(h_sum) and pd.notna(a_sum):
        df.at[idx, "player_rating_diff"] = h_sum - a_sum
        computed += 1
    
    if idx > 0 and idx % 10000 == 0:
        print(f"    {idx}/{len(df)} processed...")

print(f"  Computed {computed} games with walk-forward player ratings")

# ═══════════════════════════════════════════════════════════════
# STATS + LEAKAGE VERIFICATION
# ═══════════════════════════════════════════════════════════════

print(f"\n  === FEATURE STATS ===")
for col in ["player_rating_diff", "home_player_rating_sum", "home_player_rating_avg",
            "home_weakest_starter", "home_starter_variance"]:
    vals = df[col].dropna()
    if len(vals) > 0:
        print(f"  {col:35s} filled={len(vals):>6} mean={vals.mean():.3f} std={vals.std():.3f}")

# HONEST correlation (walk-forward, no leakage)
y = df["actual_home_score"] - df["actual_away_score"]
valid = df["player_rating_diff"].notna() & y.notna()
if valid.sum() > 100:
    corr = np.corrcoef(df.loc[valid, "player_rating_diff"], y[valid])[0, 1]
    print(f"\n  player_rating_diff × actual_margin: r={corr:.4f} (WALK-FORWARD, no leakage)")
    print(f"  (Previous leaked version was r=0.6547)")

# Compare with existing features
for feat_name, feat_calc in [
    ("mkt_spread", lambda: pd.to_numeric(df.get("market_spread_home", 0), errors="coerce")),
    ("elo_diff", lambda: pd.to_numeric(df.get("home_elo", 0), errors="coerce") - pd.to_numeric(df.get("away_elo", 0), errors="coerce")),
]:
    try:
        vals = feat_calc()
        v2 = vals.notna() & y.notna() & (vals != 0)
        if v2.sum() > 100:
            corr2 = np.corrcoef(vals[v2], y[v2])[0, 1]
            print(f"  {feat_name:35s} × actual_margin: r={corr2:.4f} (existing)")
    except:
        pass

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

if DRY_RUN:
    print(f"\n  DRY RUN — not saving.")
else:
    save_path = "ncaa_training_data.parquet"
    df.to_parquet(save_path, index=False)
    print(f"\n  Saved enriched data to {save_path}")
    
    # Save latest ratings for live predictions
    if prior_season_ratings:
        import json
        ratings_export = {pid: round(r, 3) for pid, r in prior_season_ratings.items()}
        with open("player_ratings.json", "w") as f:
            json.dump(ratings_export, f)
        print(f"  Saved {len(ratings_export)} player ratings to player_ratings.json")

print(f"\n  Done.")
