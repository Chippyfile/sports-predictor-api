"""
dump_training_data.py — Dump ncaa_historical to local parquet for training.

Usage:
    python3 dump_training_data.py              # Incremental sync (new/updated rows only)
    python3 dump_training_data.py --full       # Full re-dump from Supabase
    python3 dump_training_data.py --check      # Check if local cache exists and age

Called by retrain_and_upload.py:
    python3 retrain_and_upload.py              # Train from local cache
    python3 retrain_and_upload.py --refresh    # Incremental sync, then train

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


def sb_get_all(table, params="", timeout=120):
    """Paginated Supabase GET — pulls rows matching params."""
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={
            "apikey": KEY,
            "Authorization": f"Bearer {KEY}"
        }, timeout=timeout)
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


def dump(full=False):
    """Pull ncaa_historical from Supabase. Incremental by default, full if --full."""
    if not KEY:
        print("❌ SUPABASE_ANON_KEY not set")
        sys.exit(1)

    # Try incremental sync first (unless --full or no local cache)
    if not full and os.path.exists(PARQUET_PATH):
        return _incremental_sync()
    else:
        return _full_dump()


def _incremental_sync():
    """Only pull rows newer than what's in the local cache."""
    print("=" * 60)
    print("  INCREMENTAL SYNC — ncaa_historical")
    print("=" * 60)

    t0 = time.time()

    # Load existing parquet
    df_local = pd.read_parquet(PARQUET_PATH)
    n_before = len(df_local)
    print(f"  Local cache: {n_before} rows × {len(df_local.columns)} cols")

    # Find the latest game_date in local data
    if "game_date" in df_local.columns:
        latest_date = df_local["game_date"].dropna().max()
        print(f"  Latest local game_date: {latest_date}")
    else:
        print("  ⚠️  No game_date column — falling back to full dump")
        return _full_dump()

    # Pull only rows with game_date >= latest_date (catches updates to recent games too)
    # Go back 7 days to catch any late-arriving stats/corrections
    from datetime import timedelta
    try:
        if isinstance(latest_date, str):
            cutoff_date = pd.to_datetime(latest_date) - timedelta(days=7)
        else:
            cutoff_date = latest_date - timedelta(days=7)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    except:
        cutoff_str = str(latest_date)[:10]

    print(f"  Pulling rows with game_date >= {cutoff_str} ...")
    new_rows = sb_get_all(
        "ncaa_historical",
        f"actual_home_score=not.is.null&game_date=gte.{cutoff_str}&select=*&order=game_date.asc"
    )
    elapsed = time.time() - t0
    print(f"  Fetched {len(new_rows)} rows in {elapsed:.0f}s")

    if not new_rows:
        print("  No new data — cache is current")
        _save_timestamp(n_before, len(df_local.columns))
        return df_local

    df_new = pd.DataFrame(new_rows)

    # Align columns (Supabase may have new columns)
    for col in df_new.columns:
        if col not in df_local.columns:
            df_local[col] = pd.NA
            print(f"  New column from Supabase: {col}")
    for col in df_local.columns:
        if col not in df_new.columns:
            df_new[col] = pd.NA

    # Merge: remove old versions of updated rows, append new ones
    # Use game_id as primary key if available, otherwise game_date + team combo
    if "game_id" in df_local.columns and "game_id" in df_new.columns:
        merge_key = "game_id"
    elif "espn_game_id" in df_local.columns and "espn_game_id" in df_new.columns:
        merge_key = "espn_game_id"
    else:
        # Fallback: remove all rows with game_date >= cutoff, replace with new data
        merge_key = None

    if merge_key:
        new_ids = set(df_new[merge_key].dropna().astype(str))
        df_keep = df_local[~df_local[merge_key].astype(str).isin(new_ids)]
        df_merged = pd.concat([df_keep, df_new], ignore_index=True)
        n_updated = n_before - len(df_keep)
        n_added = len(df_new) - n_updated
        print(f"  Updated: {n_updated} rows | Added: {n_added} new rows")
    else:
        df_old = df_local[df_local["game_date"] < cutoff_str]
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        print(f"  Replaced {n_before - len(df_old)} rows from {cutoff_str}+")

    # Sort by season + game_date
    if "season" in df_merged.columns and "game_date" in df_merged.columns:
        df_merged = df_merged.sort_values(["season", "game_date"]).reset_index(drop=True)

    # Save
    if "game_id" in df_merged.columns: df_merged["game_id"] = df_merged["game_id"].astype(str)

    # Fix mixed types before parquet save (int/str mix causes ArrowInvalid)
    for _col in ['game_id', 'home_team_id', 'away_team_id']:
        if _col in df_merged.columns:
            df_merged[_col] = df_merged[_col].astype(str)
    df_merged.to_parquet(PARQUET_PATH, index=False)
    size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    _save_timestamp(len(df_merged), len(df_merged.columns), size_mb)

    print(f"\n  ✅ Incremental sync: {n_before} → {len(df_merged)} rows ({size_mb:.1f} MB)")
    print(f"     Pulled {len(new_rows)} rows vs {n_before} full dump (saved ~{(1-len(new_rows)/max(n_before,1))*100:.0f}% egress)")

    return df_merged


def _full_dump():
    """Pull entire table from Supabase."""
    if not KEY:
        print("❌ SUPABASE_ANON_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("  FULL DUMP ncaa_historical → local parquet")
    print("=" * 60)

    t0 = time.time()
    print(f"\n  Fetching from Supabase...")
    rows = sb_get_all(
        "ncaa_historical",
        "actual_home_score=not.is.null&select=*&order=season.asc",
        timeout=300  # 5 min timeout for full dump
    )
    elapsed = time.time() - t0
    print(f"  Fetched {len(rows)} rows in {elapsed:.0f}s")

    if not rows:
        print("  ❌ No data returned")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"  Columns: {len(df.columns)}")
    print(f"  Seasons: {sorted(df['season'].dropna().unique().tolist())}")

    # Save parquet
    if "game_id" in df.columns: df["game_id"] = df["game_id"].astype(str)

    # Fix mixed types before parquet save
    for _col in ['game_id', 'home_team_id', 'away_team_id']:
        if _col in df.columns:
            df[_col] = df[_col].astype(str)
    df.to_parquet(PARQUET_PATH, index=False)
    size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    _save_timestamp(len(rows), len(df.columns), size_mb)

    print(f"  Saved: {PARQUET_PATH} ({size_mb:.1f} MB)")
    print(f"\n  ✅ Done. {len(rows)} rows × {len(df.columns)} cols → {size_mb:.1f} MB")

    return df


def _save_timestamp(rows, columns, size_mb=None):
    """Save dump metadata."""
    ts = datetime.now(timezone.utc).isoformat()
    meta = {
        "dumped_at": ts,
        "rows": rows,
        "columns": columns,
    }
    if size_mb is not None:
        meta["size_mb"] = round(size_mb, 1)
    with open(TIMESTAMP_PATH, 'w') as f:
        json.dump(meta, f, indent=2)


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
        print(f"  Rows: {meta['rows']} × {meta['columns']} cols ({meta.get('size_mb', '?')} MB)")
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
    elif "--full" in sys.argv:
        dump(full=True)
    else:
        dump()
