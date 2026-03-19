#!/usr/bin/env python3
"""
backfill_lineup_concurrent.py — Concurrent PATCH for lineup features
Uses 20 threads to push ~64K rows in ~5 minutes instead of 5 hours.
"""
import os, sys, time
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')

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

if not KEY:
    print("❌ SUPABASE_ANON_KEY not set")
    sys.exit(1)

print("Loading parquet...")
df = pd.read_parquet('ncaa_training_data.parquet')
df_push = df[["game_id"] + COLUMNS].copy()
for col in COLUMNS:
    df_push[col] = pd.to_numeric(df_push[col], errors="coerce").fillna(0).round(4)

# Skip rows where all lineup features are 0 (nothing to write)
has_data = (df_push[COLUMNS] != 0).any(axis=1)
df_push = df_push[has_data]
print(f"  {len(df_push)} rows to push")

# Build list of (game_id, patch_dict)
tasks = []
for _, row in df_push.iterrows():
    gid = row["game_id"]
    if pd.isna(gid):
        continue
    patch = {col: float(row[col]) for col in COLUMNS}
    tasks.append((int(gid), patch))

print(f"  {len(tasks)} tasks prepared")

success = 0
errors = 0
t0 = time.time()
done = 0

def patch_row(gid_patch):
    gid, patch = gid_patch
    try:
        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{gid}"
        r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
        return r.ok
    except:
        return False

WORKERS = 20
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(patch_row, t): t for t in tasks}
    for future in as_completed(futures):
        ok = future.result()
        done += 1
        if ok:
            success += 1
        else:
            errors += 1
        if done % 1000 == 0 or done == len(tasks):
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(tasks) - done) / rate if rate > 0 else 0
            print(f"  [{done}/{len(tasks)}] success={success} errors={errors} "
                  f"({rate:.0f} rows/s, ETA {eta:.0f}s)")

print(f"\n✅ Done. {success} updated, {errors} errors in {time.time()-t0:.0f}s")
