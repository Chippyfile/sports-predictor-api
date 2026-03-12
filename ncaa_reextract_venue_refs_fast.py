#!/usr/bin/env python3
"""
ncaa_reextract_venue_refs_fast.py — PARALLELIZED venue/refs extraction
═══════════════════════════════════════════════════════════════════════
Memory-efficient: streams raw summaries to JSONL (one line per game)
instead of holding all 65K ESPN responses in RAM.

Resumable — reads existing JSONL cache on startup, skips already-fetched.
"""
import os, sys, json, time, gc, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
CACHE_FILE = "ncaa_extract_cache.json"
RAW_CACHE_JSONL = "ncaa_raw_summary_cache.jsonl"   # ← JSONL now, not .json
CONCURRENCY = 20
SB_BATCH = 50

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"


def fetch_summary(event_id):
    url = f"{ESPN_SUMMARY}?event={event_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return event_id, r.json()
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code == 404:
                return event_id, None
        except requests.exceptions.RequestException:
            time.sleep(1)
    return event_id, None


def extract_venue_refs_fixed(summary):
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


def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok


def load_raw_cache_ids(jsonl_path):
    """Read JSONL and return set of already-fetched game IDs + extracted data."""
    fetched_ids = set()
    extracted = {}  # gid -> venue/refs dict (small, ~200 bytes each)
    if not os.path.exists(jsonl_path):
        return fetched_ids, extracted
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                gid = rec.get("_game_id")
                if gid:
                    fetched_ids.add(gid)
                    summary = rec.get("summary")
                    if summary:
                        vr = extract_venue_refs_fixed(summary)
                        if vr:
                            extracted[gid] = vr
            except json.JSONDecodeError:
                continue
    return fetched_ids, extracted


def migrate_old_raw_cache():
    """One-time: convert old .json raw cache to .jsonl.
    Handles truncated/corrupt JSON by streaming with ijson,
    falling back to manual chunk recovery if ijson not available."""
    old_file = "ncaa_raw_summary_cache.json"
    if not os.path.exists(old_file):
        return
    print(f"  Migrating {old_file} → {RAW_CACHE_JSONL} (streaming, truncation-safe)...")

    migrated = 0
    try:
        import ijson
        # ijson streams key-value pairs from a top-level JSON object
        with open(old_file, "rb") as f, open(RAW_CACHE_JSONL, "a") as out:
            parser = ijson.kvitems(f, "")
            for gid, summary in parser:
                rec = {"_game_id": gid, "summary": summary}
                out.write(json.dumps(rec) + "\n")
                migrated += 1
                if migrated % 5000 == 0:
                    print(f"    migrated {migrated}...")
    except ImportError:
        # Fallback: use json.JSONDecoder to parse as many complete entries as possible
        print("  ijson not installed, using raw decoder fallback...")
        with open(old_file, "r") as f:
            raw = f.read()
        # Strip leading '{' if present
        if raw.startswith("{"):
            raw = raw[1:]
        decoder = json.JSONDecoder()
        pos = 0
        with open(RAW_CACHE_JSONL, "a") as out:
            while pos < len(raw):
                # Skip whitespace and commas between entries
                while pos < len(raw) and raw[pos] in " \t\n\r,":
                    pos += 1
                if pos >= len(raw) or raw[pos] == "}":
                    break
                # Expect a key string
                try:
                    key, end = decoder.raw_decode(raw, pos)
                    pos = end
                    # Skip colon and whitespace
                    while pos < len(raw) and raw[pos] in " \t\n\r:":
                        pos += 1
                    value, end = decoder.raw_decode(raw, pos)
                    pos = end
                    rec = {"_game_id": key, "summary": value}
                    out.write(json.dumps(rec) + "\n")
                    migrated += 1
                    if migrated % 5000 == 0:
                        print(f"    migrated {migrated}...")
                except (json.JSONDecodeError, ValueError):
                    print(f"  Hit corrupt data at char {pos}, stopping migration.")
                    break
        del raw
        gc.collect()
    except Exception as e:
        print(f"  Migration hit error after {migrated} records: {e}")

    if migrated > 0:
        os.rename(old_file, old_file + ".bak")
        print(f"  Migrated {migrated} records → .jsonl (original renamed to .bak)")
    else:
        print(f"  WARNING: Could not recover any records from {old_file}")


def main():
    print("=" * 70)
    print("  VENUE/REFS RE-EXTRACTION — FAST (memory-efficient)")
    print("=" * 70)

    if not os.path.exists(CACHE_FILE):
        print(f"  ERROR: {CACHE_FILE} not found"); sys.exit(1)

    # Migrate old format if present
    migrate_old_raw_cache()

    # Load extract cache (small — just venue/ref fields per game)
    with open(CACHE_FILE) as f:
        cache = json.load(f)

    game_ids = [gid for gid, data in cache.items() if data]
    print(f"  Games with ESPN data: {len(game_ids)}")

    # Load already-fetched IDs from JSONL (memory-light: only IDs + extracted fields)
    print("  Loading raw cache index...")
    fetched_ids, already_extracted = load_raw_cache_ids(RAW_CACHE_JSONL)
    print(f"  Already fetched: {len(fetched_ids)}")

    # Apply already-extracted data to cache
    venue_count = refs_count = 0
    patched_cache = {}

    for gid, vr in already_extracted.items():
        patched_cache[gid] = vr
        if gid in cache and cache[gid]:
            cache[gid].update(vr)
        if vr.get("venue_name"):
            venue_count += 1
        if vr.get("referee_1"):
            refs_count += 1

    del already_extracted  # free memory
    gc.collect()
    print(f"  From raw cache: venue={venue_count}, refs={refs_count}")

    to_fetch = [gid for gid in game_ids if gid not in fetched_ids]
    del fetched_ids
    gc.collect()
    print(f"  Need to fetch: {len(to_fetch)}")

    # ── Phase 1: Parallel fetch, stream to JSONL ──
    if to_fetch:
        print(f"\n  Fetching {len(to_fetch)} games with {CONCURRENCY} threads...")
        errors = 0
        fetched = 0
        t0 = time.time()

        # Open JSONL in append mode — each result written immediately
        jsonl_file = open(RAW_CACHE_JSONL, "a", buffering=1)  # line-buffered

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(fetch_summary, gid): gid for gid in to_fetch}

            for future in as_completed(futures):
                gid, summary = future.result()
                fetched += 1

                # Write to JSONL immediately (not held in memory)
                rec = {"_game_id": gid, "summary": summary}
                jsonl_file.write(json.dumps(rec) + "\n")

                if summary:
                    vr = extract_venue_refs_fixed(summary)
                    if vr:
                        patched_cache[gid] = vr
                        if gid in cache and cache[gid]:
                            cache[gid].update(vr)
                        if vr.get("venue_name"):
                            venue_count += 1
                        if vr.get("referee_1"):
                            refs_count += 1
                else:
                    errors += 1

                # Don't hold the summary object
                del summary, rec

                if fetched % 500 == 0:
                    elapsed = time.time() - t0
                    rate = fetched / elapsed
                    remaining = (len(to_fetch) - fetched) / rate if rate > 0 else 0
                    print(f"    {fetched}/{len(to_fetch)} | "
                          f"venue:{venue_count} refs:{refs_count} err:{errors} | "
                          f"{rate:.0f}/s | ~{remaining/60:.0f}min left")

                # Periodic GC
                if fetched % 5000 == 0:
                    gc.collect()

        jsonl_file.close()

        # Save updated extract cache
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

        elapsed = time.time() - t0
        print(f"\n  Phase 1 complete in {elapsed/60:.1f} min:")
        print(f"    venue: {venue_count} | refs: {refs_count} | errors: {errors}")
        if elapsed > 0:
            print(f"    Rate: {fetched/elapsed:.0f} games/sec")

    # ── Phase 2: Push to Supabase ──
    print(f"\n  Phase 2: Pushing {len(patched_cache)} games to Supabase...")
    valid_cols = {
        "venue_name", "venue_indoor", "venue_capacity", "venue_city", "venue_state",
        "attendance", "referee_1", "referee_2", "referee_3",
    }

    to_push = []
    for gid, vr in patched_cache.items():
        patch = {k: v for k, v in vr.items() if k in valid_cols and v is not None}
        if patch:
            to_push.append((gid, patch))

    # Free patched_cache before push phase
    del patched_cache
    gc.collect()

    print(f"  Valid patches: {len(to_push)}")

    success = errors_sb = 0
    t0 = time.time()

    def push_one(args):
        gid, patch = args
        return sb_patch("ncaa_historical", "game_id", gid, patch)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(push_one, item): item for item in to_push}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success += 1
            else:
                errors_sb += 1
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(to_push)} | success:{success} err:{errors_sb} | {rate:.0f}/s")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*70}")
    print(f"  Venue: {venue_count} | Refs: {refs_count}")
    print(f"  Supabase: {success} updated, {errors_sb} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
