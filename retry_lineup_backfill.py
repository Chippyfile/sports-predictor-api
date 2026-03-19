#!/usr/bin/env python3
"""
retry_lineup_backfill.py — Find rows still at default and retry just those
"""
import os, sys, time
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')

HEADERS_GET = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
HEADERS_PATCH = {
    "apikey": KEY, "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
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
    print("❌ SUPABASE_ANON_KEY not set"); sys.exit(1)

# Find rows still at default (all 6 home columns = 0/default)
print("Finding rows that need retry...")
url = (f"{SUPABASE_URL}/rest/v1/ncaa_historical"
       f"?home_player_rating_sum=eq.0&home_starter_variance=eq.0"
       f"&home_lineup_stability_5g=eq.1"
       f"&select=game_id&limit=5000")
r = requests.get(url, headers=HEADERS_GET, timeout=30)
if not r.ok:
    print(f"Query failed: {r.status_code} {r.text[:200]}"); sys.exit(1)

missing_ids = set(str(row["game_id"]) for row in r.json())
print(f"  {len(missing_ids)} rows still at defaults in Supabase")

# Load parquet values for those game_ids
df = pd.read_parquet('ncaa_training_data.parquet')
for col in COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(4)
df["game_id"] = df["game_id"].astype(str)

# Filter to missing + has real data
df_retry = df[df["game_id"].isin(missing_ids)]
has_data = (df_retry[COLUMNS] != 0).any(axis=1)
df_retry = df_retry[has_data]
print(f"  {len(df_retry)} of those have real data in parquet to push")

if len(df_retry) == 0:
    print("  Nothing to retry!"); sys.exit(0)

tasks = []
for _, row in df_retry.iterrows():
    gid = int(float(row["game_id"]))
    patch = {col: float(row[col]) for col in COLUMNS}
    tasks.append((gid, patch))

def patch_row(gid_patch):
    gid, patch = gid_patch
    try:
        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{gid}"
        r = requests.patch(url, headers=HEADERS_PATCH, json=patch, timeout=30)
        return r.ok
    except:
        return False

success = errors = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {pool.submit(patch_row, t): t for t in tasks}
    for f in as_completed(futures):
        if f.result(): success += 1
        else: errors += 1

print(f"\n✅ Retry done. {success} fixed, {errors} still failed in {time.time()-t0:.0f}s")
