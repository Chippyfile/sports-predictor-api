"""
dump_training_data.py — Dump ncaa_historical to local parquet for training.

Usage:
    python3 dump_training_data.py              # Dump fresh from Supabase
    python3 dump_training_data.py --check      # Check if local cache exists and age

Called by retrain_and_upload.py:
    python3 retrain_and_upload.py              # Train from local cache
    python3 retrain_and_upload.py --refresh    # Dump fresh, then train

Saves:
    ncaa_training_data.parquet   — full table (all columns)
    .ncaa_dump_timestamp         — ISO timestamp of last dump
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
PARQUET_PATH = 'ncaa_training_data.parquet'
TIMESTAMP_PATH = '.ncaa_dump_timestamp'
STALE_HOURS = 24  # Warn if cache is older than this


def sb_get_all(table, params=""):
    """Paginated Supabase GET — pulls entire table."""
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={
            "apikey": KEY,
            "Authorization": f"Bearer {KEY}"
        }, timeout=60)
        if not r.ok:
            print(f"  ❌ API error: {r.status_code} {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
        if offset % 5000 == 0:
            print(f"  ... {offset} rows fetched")
    return all_data


def dump():
    """Pull ncaa_historical from Supabase and save to local parquet."""
    if not KEY:
        print("❌ SUPABASE_ANON_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("  DUMP ncaa_historical → local parquet")
    print("=" * 60)

    t0 = time.time()
    print(f"\n  Fetching from Supabase...")
    rows = sb_get_all(
        "ncaa_historical",
        "actual_home_score=not.is.null&select=*&order=season.asc"
    )
    elapsed = time.time() - t0
    print(f"  Fetched {len(rows)} rows in {elapsed:.0f}s")

    if not rows:
        print("  ❌ No data returned")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"  Columns: {len(df.columns)}")
    print(f"  Seasons: {sorted(df['season'].dropna().unique().tolist())}")

    # Save parquet (much smaller + faster than CSV)
    df.to_parquet(PARQUET_PATH, index=False)
    size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    print(f"  Saved: {PARQUET_PATH} ({size_mb:.1f} MB)")

    # Save timestamp
    ts = datetime.now(timezone.utc).isoformat()
    with open(TIMESTAMP_PATH, 'w') as f:
        json.dump({
            "dumped_at": ts,
            "rows": len(rows),
            "columns": len(df.columns),
            "size_mb": round(size_mb, 1),
        }, f, indent=2)
    print(f"  Timestamp: {ts}")
    print(f"\n  ✅ Done. {len(rows)} rows × {len(df.columns)} cols → {size_mb:.1f} MB")
    print(f"     (vs ~132 MB over PostgREST per SELECT * query)")

    return df


def check_cache():
    """Check if local cache exists and report age."""
    if not os.path.exists(PARQUET_PATH):
        print("  No local cache found. Run: python3 dump_training_data.py")
        return False

    if os.path.exists(TIMESTAMP_PATH):
        with open(TIMESTAMP_PATH) as f:
            meta = json.load(f)
        dumped = datetime.fromisoformat(meta["dumped_at"])
        age = datetime.now(timezone.utc) - dumped
        hours = age.total_seconds() / 3600
        print(f"  Cache: {PARQUET_PATH}")
        print(f"  Dumped: {meta['dumped_at']}")
        print(f"  Age: {hours:.1f} hours")
        print(f"  Rows: {meta['rows']} × {meta['columns']} cols ({meta['size_mb']} MB)")
        if hours > STALE_HOURS:
            print(f"  ⚠️  Cache is {hours:.0f}h old (>{STALE_HOURS}h). Consider --refresh")
        else:
            print(f"  ✅ Cache is fresh")
        return True
    else:
        size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
        print(f"  Cache exists ({size_mb:.1f} MB) but no timestamp. Re-dump recommended.")
        return True


def load_cached():
    """Load cached parquet for training. Returns DataFrame or None."""
    if not os.path.exists(PARQUET_PATH):
        return None

    # Staleness warning
    if os.path.exists(TIMESTAMP_PATH):
        with open(TIMESTAMP_PATH) as f:
            meta = json.load(f)
        dumped = datetime.fromisoformat(meta["dumped_at"])
        age_hours = (datetime.now(timezone.utc) - dumped).total_seconds() / 3600
        if age_hours > STALE_HOURS:
            print(f"  ⚠️  Training cache is {age_hours:.0f}h old. Use --refresh if data changed.")

    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} cols from local cache")
    return df


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_cache()
    else:
        dump()
