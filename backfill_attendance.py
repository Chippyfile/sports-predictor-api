"""
BACKFILL ATTENDANCE from ESPN game summary.

ESPN gameInfo.attendance is available for all seasons (2015-2026, confirmed).
ncaa_historical has 0% attendance. This fills it.

After completion, crowd_shock_diff can be computed from attendance data.

Rate: ~2 req/sec with 0.5s sleep. ~60K games = ~8 hours.
Resumable: saves progress to attendance_cache.json.

Usage:
  python3 backfill_attendance.py                    # all seasons
  python3 backfill_attendance.py --season 2026      # single season
  python3 backfill_attendance.py --test              # 10 games only
  python3 backfill_attendance.py --write             # actually write to Supabase
  python3 backfill_attendance.py --status            # show cache status
"""

import sys, os, time, json, requests

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
KEY = os.environ.get("SUPABASE_ANON_KEY", "")
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
CACHE_FILE = "attendance_cache.json"
BATCH_SIZE = 50

# Parse args
TEST_MODE = "--test" in sys.argv
WRITE_MODE = "--write" in sys.argv
STATUS_MODE = "--status" in sys.argv
TARGET_SEASON = None
for i, arg in enumerate(sys.argv):
    if arg == "--season" and i + 1 < len(sys.argv):
        TARGET_SEASON = int(sys.argv[i + 1])

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def sb_headers():
    return {"apikey": KEY, "Authorization": "Bearer " + KEY}

def get_game_ids_for_season(season):
    """Get all game_ids for a season from ncaa_historical via date-range queries."""
    all_ids = []
    start_year = season - 1
    # Query month by month to avoid Supabase timeout
    months = [
        ("%d-10-01" % start_year, "%d-11-01" % start_year),
        ("%d-11-01" % start_year, "%d-12-01" % start_year),
        ("%d-12-01" % start_year, "%d-01-01" % season),
        ("%d-01-01" % season, "%d-02-01" % season),
        ("%d-02-01" % season, "%d-03-01" % season),
        ("%d-03-01" % season, "%d-04-01" % season),
        ("%d-04-01" % season, "%d-05-01" % season),
    ]
    for start, end in months:
        offset = 0
        while True:
            url = "%s/rest/v1/ncaa_historical?season=eq.%d&game_date=gte.%s&game_date=lt.%s&select=game_id&limit=1000&offset=%d" % (
                SUPABASE_URL, season, start, end, offset)
            r = requests.get(url, headers=sb_headers(), timeout=60)
            if not r.ok:
                print("    Supabase error %s-%s: %d" % (start, end, r.status_code))
                break
            data = r.json()
            ids = [row["game_id"] for row in data if row.get("game_id")]
            all_ids.extend(ids)
            if len(data) < 1000:
                break
            offset += 1000
    return list(set(all_ids))  # deduplicate

def fetch_attendance(game_id):
    """Fetch attendance and venue from ESPN game summary."""
    try:
        r = requests.get("%s?event=%s" % (ESPN_SUMMARY, game_id), timeout=10)
        if not r.ok:
            return None
        d = r.json()
        gi = d.get("gameInfo", {})
        att = gi.get("attendance", 0) or 0
        venue = gi.get("venue", {})
        venue_name = venue.get("fullName", "")
        venue_cap = venue.get("capacity", 0) or 0
        return {"attendance": int(att), "venue_name": venue_name, "venue_capacity": int(venue_cap)}
    except Exception as e:
        return None

def write_to_supabase(updates):
    """Write attendance/venue back to ncaa_historical."""
    if not updates:
        return 0
    headers = {**sb_headers(), "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    written = 0
    for i in range(0, len(updates), BATCH_SIZE):
        batch = updates[i:i + BATCH_SIZE]
        rows = [{"game_id": gid, "attendance": data["attendance"]} for gid, data in batch]
        r = requests.post(
            SUPABASE_URL + "/rest/v1/ncaa_historical",
            headers=headers, json=rows, timeout=60
        )
        if r.ok:
            written += len(rows)
        else:
            print("    Write error at %d: %d %s" % (i, r.status_code, r.text[:200]))
    return written

# ── Main ──
if __name__ == "__main__":
    cache = load_cache()
    
    if STATUS_MODE:
        print("Cache: %d games" % len(cache))
        has_att = sum(1 for v in cache.values() if v.get("attendance", 0) > 0)
        print("  With attendance: %d (%.1f%%)" % (has_att, has_att / max(len(cache), 1) * 100))
        # By season (estimate from game_id ranges)
        sys.exit(0)
    
    seasons = [TARGET_SEASON] if TARGET_SEASON else [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026]
    
    print("=" * 70)
    print("  ATTENDANCE BACKFILL: Seasons %s" % seasons)
    print("  Mode: %s" % ("TEST (10 games)" if TEST_MODE else ("WRITE to Supabase" if WRITE_MODE else "DRY RUN (cache only)")))
    print("  Cache: %d games already fetched" % len(cache))
    print("=" * 70)
    
    total_fetched = 0
    total_with_att = 0
    
    for season in seasons:
        print("\n  Season %d:" % season)
        print("    Fetching game_ids from Supabase...")
        game_ids = get_game_ids_for_season(season)
        print("    Found %d games" % len(game_ids))
        
        # Filter out already cached
        needed = [gid for gid in game_ids if str(gid) not in cache]
        print("    Already cached: %d, need to fetch: %d" % (len(game_ids) - len(needed), len(needed)))
        
        if TEST_MODE:
            needed = needed[:10]
            print("    TEST MODE: limiting to %d games" % len(needed))
        
        fetched = 0
        has_att = 0
        errors = 0
        
        for i, gid in enumerate(needed):
            result = fetch_attendance(gid)
            if result:
                cache[str(gid)] = result
                fetched += 1
                if result["attendance"] > 0:
                    has_att += 1
            else:
                cache[str(gid)] = {"attendance": 0, "venue_name": "", "venue_capacity": 0, "error": True}
                errors += 1
            
            # Progress every 100
            if (i + 1) % 100 == 0:
                save_cache(cache)
                pct = (i + 1) / len(needed) * 100
                print("    [%d/%d] %.0f%% — fetched=%d, att=%d, errors=%d" % (
                    i + 1, len(needed), pct, fetched, has_att, errors))
            
            # Rate limit
            time.sleep(0.5)
        
        # Save after each season
        save_cache(cache)
        
        total_fetched += fetched
        total_with_att += has_att
        
        print("    Done: fetched=%d, with attendance=%d, errors=%d" % (fetched, has_att, errors))
        
        # Write to Supabase if requested
        if WRITE_MODE and fetched > 0:
            season_updates = [(gid, cache[str(gid)]) for gid in game_ids
                             if str(gid) in cache and cache[str(gid)].get("attendance", 0) > 0]
            written = write_to_supabase(season_updates)
            print("    Written to Supabase: %d rows" % written)
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("  Total fetched: %d, with attendance: %d" % (total_fetched, total_with_att))
    print("  Cache total: %d games" % len(cache))
    if not WRITE_MODE:
        print("\n  To write to Supabase, run again with --write")
    print("  After writing, run: python3 retrain_and_upload.py --refresh")
    print("=" * 70)
