#!/usr/bin/env python3
"""
ncaa_reextract_venue_refs.py — Re-extract venue/refs from cached ESPN data
══════════════════════════════════════════════════════════════════════════════
No API calls. Reads ncaa_extract_cache.json, re-runs the FIXED venue/refs
extraction on the raw summaries, updates cache, then pushes to Supabase.

Usage:
  SUPABASE_ANON_KEY="..." python3 ncaa_reextract_venue_refs.py
"""
import os, sys, json, time, requests

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
CACHE_FILE = "ncaa_extract_cache.json"

# We need the RAW ESPN summaries, not the extracted data.
# The current cache stores extracted results, not raw JSON.
# So we need to re-fetch from ESPN — BUT we can do it from a raw cache.
#
# PLAN B: Since the cache only has extracted data (not raw summaries),
# we need to re-hit ESPN. But we can do it FAST by only fetching games
# that have PBP data (meaning ESPN returned a valid summary) and only
# extracting venue/refs.

RAW_CACHE_FILE = "ncaa_raw_summary_cache.json"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"


def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok


def fetch_summary(event_id):
    url = f"{ESPN_SUMMARY}?event={event_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok: return r.json()
            if r.status_code == 429: time.sleep(2 ** (attempt + 1)); continue
            if r.status_code == 404: return None
        except requests.exceptions.RequestException:
            time.sleep(1)
    return None


def extract_venue_refs_fixed(summary):
    """FIXED extraction — reads from gameInfo instead of header.competitions."""
    r = {}
    gi = summary.get("gameInfo", {})

    venue = gi.get("venue", {})
    if venue:
        r["venue_name"] = venue.get("fullName", "")
        r["venue_indoor"] = 1 if not venue.get("grass", False) else 0
        cap = venue.get("capacity")
        if cap:
            try: r["venue_capacity"] = int(cap)
            except: pass
        addr = venue.get("address", {})
        if addr:
            r["venue_city"] = addr.get("city", "")
            r["venue_state"] = addr.get("state", "")

    att = gi.get("attendance")
    if att:
        try: r["attendance"] = int(att)
        except: pass

    officials = gi.get("officials", [])
    for i, off in enumerate(officials[:3]):
        name = off.get("displayName", "")
        if name:
            r[f"referee_{i+1}"] = name

    return r


def main():
    print("=" * 70)
    print("  VENUE/REFS RE-EXTRACTION (fixed paths)")
    print("=" * 70)

    # Load existing cache to get game IDs that had valid ESPN responses
    if not os.path.exists(CACHE_FILE):
        print(f"  ERROR: {CACHE_FILE} not found"); sys.exit(1)

    with open(CACHE_FILE) as f:
        cache = json.load(f)

    # Only re-fetch games that had data (non-empty extracted results)
    game_ids = [gid for gid, data in cache.items() if data]
    print(f"  Games with ESPN data: {len(game_ids)}")

    # Load raw summary cache if it exists (to avoid re-fetching)
    raw_cache = {}
    if os.path.exists(RAW_CACHE_FILE):
        with open(RAW_CACHE_FILE) as f:
            raw_cache = json.load(f)
        print(f"  Raw cache: {len(raw_cache)} summaries")

    # Phase 1: Fetch raw summaries and extract venue/refs
    print(f"\n  Phase 1: Fetching + extracting venue/refs...")
    venue_count = 0
    refs_count = 0
    errors = 0
    patched_cache = {}  # venue/refs data per game_id

    to_fetch = [gid for gid in game_ids if gid not in raw_cache]
    already_cached = [gid for gid in game_ids if gid in raw_cache]
    print(f"  Already in raw cache: {len(already_cached)}")
    print(f"  Need to fetch: {len(to_fetch)}")

    # First process already-cached raw summaries
    for gid in already_cached:
        summary = raw_cache[gid]
        if not summary:
            continue
        vr = extract_venue_refs_fixed(summary)
        if vr:
            patched_cache[gid] = vr
            # Update the main extract cache too
            cache[gid].update(vr)
            if "venue_name" in vr and vr["venue_name"]:
                venue_count += 1
            if "referee_1" in vr and vr["referee_1"]:
                refs_count += 1

    print(f"  From raw cache: venue={venue_count}, refs={refs_count}")

    # Now fetch remaining
    for i, gid in enumerate(to_fetch):
        summary = fetch_summary(gid)
        if summary:
            # Save raw summary
            raw_cache[gid] = summary
            vr = extract_venue_refs_fixed(summary)
            if vr:
                patched_cache[gid] = vr
                cache[gid].update(vr)
                if "venue_name" in vr and vr["venue_name"]:
                    venue_count += 1
                if "referee_1" in vr and vr["referee_1"]:
                    refs_count += 1
        else:
            raw_cache[gid] = None
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(to_fetch)} | venue:{venue_count} refs:{refs_count} err:{errors}")
            # Save raw cache periodically
            with open(RAW_CACHE_FILE, "w") as f:
                json.dump(raw_cache, f)

        time.sleep(0.2)

    # Save caches
    with open(RAW_CACHE_FILE, "w") as f:
        json.dump(raw_cache, f)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    print(f"\n  Phase 1 complete:")
    print(f"    venue: {venue_count}")
    print(f"    refs:  {refs_count}")
    print(f"    errors: {errors}")

    # Phase 2: Push venue/refs columns to Supabase
    print(f"\n  Phase 2: Pushing venue/refs to Supabase...")
    valid_cols = {
        "venue_name", "venue_indoor", "venue_capacity", "venue_city", "venue_state",
        "attendance", "referee_1", "referee_2", "referee_3",
    }

    success = 0
    skipped = 0
    for i, (gid, vr) in enumerate(patched_cache.items()):
        patch = {k: v for k, v in vr.items() if k in valid_cols and v is not None}
        if not patch:
            skipped += 1
            continue
        if sb_patch("ncaa_historical", "game_id", gid, patch):
            success += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(patched_cache)} ({success} updated, {skipped} skipped)")
        time.sleep(0.02)

    print(f"\n{'='*70}")
    print(f"  VENUE/REFS RE-EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Venue: {venue_count} | Refs: {refs_count} | Pushed: {success}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
