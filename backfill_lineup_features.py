#!/usr/bin/env python3
"""
backfill_lineup_features.py — Push lineup + player rating columns from parquet to Supabase
============================================================================================

The training parquet has these 12 columns computed from starter_ids + player_ratings.json.
ncaa_historical doesn't have them yet. This script writes them back.

Step 1: Run the SQL in Supabase SQL Editor first (printed at start)
Step 2: python3 -u backfill_lineup_features.py --check    # preview
Step 3: python3 -u backfill_lineup_features.py             # push

"""
import os, sys, time, json
import pandas as pd
import numpy as np
import requests

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
PARQUET = 'ncaa_training_data.parquet'
CHECK_ONLY = '--check' in sys.argv

HEADERS = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

COLUMNS = [
    "home_player_rating_sum", "away_player_rating_sum",
    "home_weakest_starter", "away_weakest_starter",
    "home_starter_variance", "away_starter_variance",
    "home_lineup_changes", "away_lineup_changes",
    "home_lineup_stability_5g", "away_lineup_stability_5g",
    "home_starter_games_together", "away_starter_games_together",
]

print("=" * 60)
print("  BACKFILL LINEUP + PLAYER RATING FEATURES")
print("=" * 60)

# Step 1: Print SQL
print("""
  STEP 1: Run this SQL in Supabase SQL Editor FIRST:
  ─────────────────────────────────────────────────
  ALTER TABLE ncaa_historical
    ADD COLUMN IF NOT EXISTS home_player_rating_sum FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS away_player_rating_sum FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS home_weakest_starter FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS away_weakest_starter FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS home_starter_variance FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS away_starter_variance FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS home_lineup_changes FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS away_lineup_changes FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS home_lineup_stability_5g FLOAT DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS away_lineup_stability_5g FLOAT DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS home_starter_games_together FLOAT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS away_starter_games_together FLOAT DEFAULT 0;
  ─────────────────────────────────────────────────
""")

if not KEY:
    print("  ❌ SUPABASE_ANON_KEY not set")
    sys.exit(1)

# Load parquet
print(f"  Loading {PARQUET}...")
df = pd.read_parquet(PARQUET)
print(f"  Loaded {len(df)} rows")

# Check columns exist in parquet
missing = [c for c in COLUMNS if c not in df.columns]
if missing:
    print(f"  ❌ Missing from parquet: {missing}")
    sys.exit(1)

# Filter to rows that have real data (at least one nonzero lineup feature)
has_data = (df[COLUMNS].fillna(0) != 0).any(axis=1)
df_push = df[has_data][["game_id"] + COLUMNS].copy()
print(f"  Rows with lineup data: {len(df_push)} / {len(df)}")

# Clean NaN → 0
for col in COLUMNS:
    df_push[col] = pd.to_numeric(df_push[col], errors="coerce").fillna(0).round(4)

# Preview
print(f"\n  Sample data:")
for _, row in df_push.head(3).iterrows():
    print(f"    game_id={row['game_id']}: pr_sum={row['home_player_rating_sum']:.3f}, "
          f"weakest={row['home_weakest_starter']:.3f}, "
          f"lineup_chg={row['home_lineup_changes']:.0f}, "
          f"stab={row['home_lineup_stability_5g']:.3f}")

# Stats
print(f"\n  Column stats (from parquet):")
for col in COLUMNS:
    if "home" in col:  # just show home side
        s = df_push[col]
        print(f"    {col}: nonzero={( s != 0).sum()}, mean={s.mean():.4f}, std={s.std():.4f}")

if CHECK_ONLY:
    print(f"\n  CHECK ONLY — not pushing.")
    sys.exit(0)

# Push in batches
BATCH = 200
success = 0
errors = 0
t0 = time.time()

for i in range(0, len(df_push), BATCH):
    batch = df_push.iloc[i:i+BATCH]
    for _, row in batch.iterrows():
        game_id = row["game_id"]
        patch = {col: float(row[col]) for col in COLUMNS}

        try:
            url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
            r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
            if r.ok:
                success += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"    Error on {game_id}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Exception on {game_id}: {e}")

    elapsed = time.time() - t0
    rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
    eta = (len(df_push) - i - len(batch)) / rate if rate > 0 else 0
    print(f"    [{i+len(batch)}/{len(df_push)}] success={success} errors={errors} "
          f"({rate:.0f} rows/s, ETA {eta/60:.1f}min)")

print(f"\n  ✅ Done. {success} updated, {errors} errors in {time.time()-t0:.0f}s")
