#!/usr/bin/env python3
"""
ncaa_push_player_features.py — Push cached player-derived features to Supabase
═══════════════════════════════════════════════════════════════════════════════
12 player-derived features with ~64,800 coverage sitting in ncaa_extract_cache.json
but never pushed to ncaa_historical. This script fixes that.

Features pushed:
  - home/away_minutes_hhi        (rotation concentration — HHI of minutes)
  - home/away_star1_pts_share    (star dependency — top scorer's share)
  - home/away_top3_pts_share     (top-3 scorer concentration)
  - home/away_bench_pts          (bench scoring raw)
  - home/away_bench_pts_share    (bench scoring share)
  - home/away_players_used       (rotation size)

Usage:
  python3 ncaa_push_player_features.py
  python3 ncaa_push_player_features.py --dry-run     # preview without pushing
  python3 ncaa_push_player_features.py --batch 200   # custom batch size
"""
import os, sys, json, time, argparse
import requests

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
CACHE_FILE = "ncaa_extract_cache.json"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY or SUPABASE_KEY env var")
    sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

# All player-derived columns to push
PLAYER_COLS = [
    "home_minutes_hhi", "away_minutes_hhi",
    "home_star1_pts_share", "away_star1_pts_share",
    "home_top3_pts_share", "away_top3_pts_share",
    "home_bench_pts", "away_bench_pts",
    "home_bench_pts_share", "away_bench_pts_share",
    "home_players_used", "away_players_used",
]


def sb_patch_batch(game_ids_and_patches, batch_size=100):
    """Patch rows one at a time (Supabase REST doesn't support batch PATCH by different keys)."""
    success = 0
    errors = 0
    for i, (game_id, patch) in enumerate(game_ids_and_patches):
        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
        try:
            r = requests.patch(url, headers=HEADERS, json=patch, timeout=15)
            if r.ok:
                success += 1
            else:
                errors += 1
                if errors <= 5:
                    print(f"    PATCH error {r.status_code} for {game_id}: {r.text[:100]}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Exception for {game_id}: {e}")

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(game_ids_and_patches)} | success:{success} errors:{errors}")
            time.sleep(0.5)  # brief pause every 500

        # Rate limit: ~50 req/sec is safe for Supabase
        if (i + 1) % 50 == 0:
            time.sleep(0.2)

    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Push player-derived features to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Preview without pushing")
    parser.add_argument("--batch", type=int, default=100, help="Batch size for progress reporting")
    args = parser.parse_args()

    print("=" * 70)
    print("  PUSH PLAYER-DERIVED FEATURES: Cache → Supabase")
    print("=" * 70)

    # Load cache
    if not os.path.exists(CACHE_FILE):
        print(f"  ERROR: {CACHE_FILE} not found")
        sys.exit(1)

    print(f"  Loading {CACHE_FILE}...")
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"  Cache: {len(cache)} games")

    # Build patches: only include games that have at least one player column
    patches = []
    col_counts = {col: 0 for col in PLAYER_COLS}

    for game_id, data in cache.items():
        if not data:
            continue

        patch = {}
        for col in PLAYER_COLS:
            val = data.get(col)
            if val is not None:
                # Round floats to 4 decimal places for clean storage
                if isinstance(val, float):
                    patch[col] = round(val, 4)
                else:
                    patch[col] = val
                col_counts[col] += 1

        if patch:
            patches.append((game_id, patch))

    print(f"\n  Games with player data: {len(patches)}")
    print(f"\n  Column coverage:")
    for col in PLAYER_COLS:
        print(f"    {col:<30} {col_counts[col]:>8,}")

    if args.dry_run:
        print(f"\n  DRY RUN — showing 3 sample patches:")
        for game_id, patch in patches[:3]:
            print(f"    {game_id}: {json.dumps(patch, indent=6)}")
        print(f"\n  Would push {len(patches)} patches. Run without --dry-run to execute.")
        return

    # Push to Supabase
    print(f"\n  Pushing {len(patches):,} patches to ncaa_historical...")
    t0 = time.time()
    success, errors = sb_patch_batch(patches)
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  PUSH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Success: {success:,}")
    print(f"  Errors:  {errors:,}")
    print(f"  Time:    {elapsed/60:.1f} minutes")
    print(f"  Rate:    {len(patches)/elapsed:.0f} patches/sec")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
