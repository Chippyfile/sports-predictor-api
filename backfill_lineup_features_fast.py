#!/usr/bin/env python3
"""
backfill_lineup_features_fast.py — Batch upsert lineup features from parquet to Supabase
"""
import os, sys, time
import pandas as pd
import numpy as np
import requests

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
PARQUET = 'ncaa_training_data.parquet'

HEADERS = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates,return=minimal",
}

COLUMNS = [
    "home_player_rating_sum", "away_player_rating_sum",
    "home_weakest_starter", "away_weakest_starter",
    "home_starter_variance", "away_starter_variance",
    "home_lineup_changes", "away_lineup_changes",
    "home_lineup_stability_5g", "away_lineup_stability_5g",
    "home_starter_games_together", "away_starter_games_together",
]

if not KEY:
    print("❌ SUPABASE_ANON_KEY not set")
    sys.exit(1)

print("Loading parquet...")
df = pd.read_parquet(PARQUET)
print(f"  {len(df)} rows")

# Prep: just game_id + 12 columns, clean NaN
df_push = df[["game_id"] + COLUMNS].copy()
for col in COLUMNS:
    df_push[col] = pd.to_numeric(df_push[col], errors="coerce").fillna(0)
# Round to save bandwidth
for col in COLUMNS:
    df_push[col] = df_push[col].round(4)

# Convert to records
rows = df_push.to_dict(orient="records")
# Ensure game_id is native Python type
for r in rows:
    r["game_id"] = int(r["game_id"]) if pd.notna(r["game_id"]) else None
    for col in COLUMNS:
        r[col] = float(r[col])

# Remove rows without game_id
rows = [r for r in rows if r.get("game_id")]
print(f"  {len(rows)} rows to push")

# Batch upsert (Supabase supports up to ~1000 rows per POST)
BATCH = 500
success = 0
errors = 0
t0 = time.time()

for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    try:
        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical"
        r = requests.post(url, headers=HEADERS, json=batch, timeout=60)
        if r.ok:
            success += len(batch)
        else:
            errors += len(batch)
            print(f"  Batch error at {i}: {r.status_code} {r.text[:200]}")
    except Exception as e:
        errors += len(batch)
        print(f"  Batch exception at {i}: {e}")

    elapsed = time.time() - t0
    done = i + len(batch)
    rate = done / elapsed if elapsed > 0 else 0
    eta = (len(rows) - done) / rate if rate > 0 else 0
    print(f"  [{done}/{len(rows)}] success={success} errors={errors} ({rate:.0f} rows/s, ETA {eta:.0f}s)")

print(f"\n✅ Done. {success} updated, {errors} errors in {time.time()-t0:.0f}s")
