#!/usr/bin/env python3
"""
upload_nba_ref_profiles.py — Push referee profiles to Supabase nba_ref_profiles table.

Usage:
    python3 upload_nba_ref_profiles.py                    # Dry run
    python3 upload_nba_ref_profiles.py --push             # Upload to Supabase

Creates/updates rows in nba_ref_profiles with columns:
  ref_name, n_games, home_whistle, avg_foul_rate, ou_bias, pace_impact, total_pts_avg
"""
import json, sys, os
sys.path.insert(0, ".")

DRY_RUN = "--push" not in sys.argv

with open("nba_referee_profiles.json") as f:
    profiles = json.load(f)

print(f"Loaded {len(profiles)} referee profiles")
print(f"Mode: {'DRY RUN' if DRY_RUN else 'PUSHING TO SUPABASE'}")

rows = []
for name, stats in profiles.items():
    rows.append({
        "ref_name":       name,
        "n_games":        stats["n_games"],
        "home_whistle":   round(stats["home_whistle"], 6),
        "avg_foul_rate":  round(stats.get("foul_proxy", 1.0), 6),
        "ou_bias":        round(stats.get("ou_bias", 0), 4),
        "pace_impact":    round(stats.get("pace_impact", 0), 6),
        "total_pts_avg":  round(stats.get("total_pts_avg", 220), 2),
        "avg_home_margin": round(stats.get("home_whistle", 0) * 4.0, 4),  # approximate
    })

# Show sample
print(f"\nSample rows:")
for r in rows[:5]:
    print(f"  {r['ref_name']:25s} games={r['n_games']:4d} hw={r['home_whistle']:+.4f} "
          f"foul={r['avg_foul_rate']:.4f} ou={r['ou_bias']:+.2f} pace={r['pace_impact']:+.4f}")

if DRY_RUN:
    print(f"\n{len(rows)} rows ready. Run with --push to upload.")
    sys.exit(0)

# Upload
import requests
from config import SUPABASE_URL, SUPABASE_KEY

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

# Batch upsert (ref_name is unique key)
BATCH = 50
uploaded = 0
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/nba_ref_profiles?on_conflict=ref_name",
        headers=headers,
        json=batch,
        timeout=30,
    )
    if resp.ok or resp.status_code == 201:
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{len(rows)}")
    else:
        print(f"  ERROR: {resp.status_code} {resp.text[:200]}")
        # If table doesn't have on_conflict, try without
        if "unique" in resp.text.lower() or "conflict" in resp.text.lower():
            print("  Retrying without on_conflict (DELETE + INSERT)...")
            for row in batch:
                requests.delete(
                    f"{SUPABASE_URL}/rest/v1/nba_ref_profiles?ref_name=eq.{row['ref_name']}",
                    headers=headers, timeout=10
                )
            resp2 = requests.post(
                f"{SUPABASE_URL}/rest/v1/nba_ref_profiles",
                headers=headers, json=batch, timeout=30
            )
            if resp2.ok or resp2.status_code == 201:
                uploaded += len(batch)
                print(f"  Uploaded {uploaded}/{len(rows)} (via delete+insert)")
            else:
                print(f"  STILL FAILED: {resp2.status_code} {resp2.text[:200]}")

print(f"\nDone: {uploaded}/{len(rows)} referee profiles uploaded")
