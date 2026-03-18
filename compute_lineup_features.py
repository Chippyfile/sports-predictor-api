#!/usr/bin/env python3
"""
compute_lineup_features.py
===========================
Computes lineup-based features from starter_ids for every game in the parquet.
Pushes results to Supabase columns.

Features computed (per team per game):
  - lineup_changes: # starters changed from previous game (0-5)
  - lineup_stability_5g: rolling 5-game avg of starter overlap (0.0-1.0)
  - starter_games_together: how many games this season the current 5 have ALL started
  - is_new_starter: 1 if any starter is making their first start of the season

Run:
    python3 -u compute_lineup_features.py              # compute + push
    python3 -u compute_lineup_features.py --dry-run    # compute only, show stats
"""

import sys, os, json, time, warnings
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

DRY_RUN = "--dry-run" in sys.argv
PARQUET = "ncaa_training_data.parquet"

print("=" * 60)
print("  LINEUP FEATURE COMPUTATION")
print("=" * 60)

# ── Load data ──
df = pd.read_parquet(PARQUET)
df = df.sort_values(["season", "game_date"]).reset_index(drop=True)
print(f"  Loaded {len(df)} games")

has_starters = df["home_starter_ids"].notna() & (df["home_starter_ids"] != "")
print(f"  Games with starter data: {has_starters.sum()} ({has_starters.mean()*100:.1f}%)")

# ── Initialize output columns ──
for prefix in ["home_", "away_"]:
    df[f"{prefix}lineup_changes"] = np.nan
    df[f"{prefix}lineup_stability_5g"] = np.nan
    df[f"{prefix}starter_games_together"] = np.nan
    df[f"{prefix}is_new_starter"] = np.nan

# ── Process each game chronologically ──
# Track per-team: previous starters, rolling overlap history, season starter counts

# team_id -> list of starter sets (chronological within season)
team_history = defaultdict(list)
# team_id -> season -> set of all player IDs who have started
team_season_starters = defaultdict(lambda: defaultdict(set))
# team_id -> season -> Counter of frozenset(5 starters) -> count
team_lineup_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

print("  Computing lineup features...")
t0 = time.time()
computed = 0

for idx, row in df.iterrows():
    season = row.get("season", 0)
    
    for side in ["home", "away"]:
        tid = str(row.get(f"{side}_team_id", ""))
        ids_str = row.get(f"{side}_starter_ids")
        
        if not isinstance(ids_str, str) or ids_str.strip() == "":
            continue
        
        current = set(ids_str.strip().split(","))
        if len(current) < 3:  # bad data
            continue
        
        history = team_history[f"{tid}_{season}"]
        season_starters = team_season_starters[tid][season]
        lineup_counts = team_lineup_counts[tid][season]
        
        # ── Feature 1: lineup_changes (vs previous game) ──
        if history:
            prev = history[-1]
            overlap = len(current & prev)
            changes = 5 - overlap
        else:
            changes = 0  # first game of season
            overlap = 5
        df.at[idx, f"{side}_lineup_changes"] = changes
        
        # ── Feature 2: lineup_stability_5g (rolling 5-game avg overlap ratio) ──
        if len(history) >= 1:
            recent = history[-5:] if len(history) >= 5 else history
            overlaps = [len(current & h) / 5.0 for h in recent]
            stability = np.mean(overlaps)
        else:
            stability = 1.0  # first game
        df.at[idx, f"{side}_lineup_stability_5g"] = round(stability, 4)
        
        # ── Feature 3: starter_games_together ──
        # How many games this season have ALL 5 of these players started together?
        lineup_key = frozenset(current)
        # Count BEFORE incrementing (how many times have we seen this exact 5 before?)
        games_together = lineup_counts[lineup_key]
        df.at[idx, f"{side}_starter_games_together"] = games_together
        lineup_counts[lineup_key] += 1
        
        # ── Feature 4: is_new_starter ──
        # Is any player making their first start of the season?
        new_starter = any(pid not in season_starters for pid in current)
        df.at[idx, f"{side}_is_new_starter"] = int(new_starter)
        
        # Update tracking
        history.append(current)
        season_starters.update(current)
        computed += 1
    
    if idx > 0 and idx % 10000 == 0:
        print(f"    {idx}/{len(df)} processed...")

elapsed = time.time() - t0
print(f"  Computed {computed} team-game observations in {elapsed:.0f}s")

# ── Stats ──
print(f"\n  === FEATURE STATS ===")
for col in ["home_lineup_changes", "home_lineup_stability_5g", 
            "home_starter_games_together", "home_is_new_starter"]:
    filled = df[col].notna().sum()
    if filled > 0:
        vals = df[col].dropna()
        print(f"  {col:40s} filled={filled:>6} mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.0f} max={vals.max():.0f}")

# Distribution of lineup changes
changes = df["home_lineup_changes"].dropna()
print(f"\n  === LINEUP CHANGE DISTRIBUTION ===")
for i in range(6):
    pct = (changes == i).mean() * 100
    print(f"  {i} changes: {pct:.1f}%")

# ── Push to Supabase ──
if DRY_RUN:
    print(f"\n  DRY RUN — not pushing. Use without --dry-run to push to Supabase.")
else:
    print(f"\n  Pushing to Supabase...")
    from dump_training_data import load_cached
    
    SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
    KEY = os.environ.get('SUPABASE_ANON_KEY', '')
    
    if not KEY:
        print("  ❌ SUPABASE_ANON_KEY not set. Export it and retry.")
        sys.exit(1)
    
    import requests
    
    # Only push games that have game_id and computed values
    push_cols = [
        "home_lineup_changes", "away_lineup_changes",
        "home_lineup_stability_5g", "away_lineup_stability_5g", 
        "home_starter_games_together", "away_starter_games_together",
        "home_is_new_starter", "away_is_new_starter",
    ]
    
    to_push = df[df["home_lineup_changes"].notna() & df["game_id"].notna()].copy()
    print(f"  {len(to_push)} games to push")
    
    pushed = 0
    errors = 0
    batch_size = 100
    
    for start in range(0, len(to_push), batch_size):
        batch = to_push.iloc[start:start+batch_size]
        
        for _, row in batch.iterrows():
            game_id = row["game_id"]
            update = {}
            for col in push_cols:
                val = row.get(col)
                if pd.notna(val):
                    update[col] = float(val) if isinstance(val, (np.floating, float)) else int(val)
            
            if not update:
                continue
                
            try:
                r = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_id=eq.{game_id}",
                    headers={
                        "apikey": KEY,
                        "Authorization": f"Bearer {KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                    json=update,
                    timeout=10,
                )
                if r.ok:
                    pushed += 1
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"    ❌ {game_id}: {r.status_code} {r.text[:100]}")
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    ❌ {game_id}: {e}")
        
        if (start // batch_size) % 10 == 0 and start > 0:
            print(f"    Pushed {pushed}/{start + len(batch)}...")
    
    print(f"  ✅ Pushed {pushed} games ({errors} errors)")

# ── Save enriched parquet ──
save_path = "ncaa_training_data_with_lineup.parquet"
df.to_parquet(save_path, index=False)
print(f"\n  Saved enriched data to {save_path}")
print(f"  Done.")
