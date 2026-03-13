#!/usr/bin/env python3
"""
ncaa_fix_hhi_players.py — Re-extract player minutes from ESPN, compute HHI + players_used
══════════════════════════════════════════════════════════════════════════════════════════
The original extraction hardcoded HHI=0.2 and players_used=0.
This script fetches boxscore.players from ESPN, extracts actual minutes,
and computes:
  - minutes_hhi: Herfindahl index of minutes (higher = more concentrated)
  - players_used: count of players who played > 0 minutes

Saves to cache file, then pushes to Supabase.

Usage:
  python3 ncaa_fix_hhi_players.py                    # full run
  python3 ncaa_fix_hhi_players.py --limit 100        # test on 100 games
  python3 ncaa_fix_hhi_players.py --push-only        # skip fetch, push cached results
"""
import os, sys, json, time, argparse
import requests
import numpy as np

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
CACHE_FILE = "ncaa_hhi_players_cache.json"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def fetch_espn_minutes(event_id):
    """Fetch player minutes from ESPN summary boxscore."""
    url = f"{ESPN_SUMMARY}?event={event_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if not r.ok:
                return None
            data = r.json()

            result = {}
            boxscore = data.get("boxscore", {})
            players_section = boxscore.get("players", [])

            # Build team_id → home/away mapping from header
            team_ha = {}
            comp = data.get("header", {}).get("competitions", [{}])[0]
            for c in comp.get("competitors", []):
                tid = str(c.get("id", c.get("team", {}).get("id", "")))
                ha = c.get("homeAway", "")
                if tid and ha:
                    team_ha[tid] = ha

            for team_data in players_section:
                # Look up home/away from header mapping
                team_id = str(team_data.get("team", {}).get("id", ""))
                home_away = team_ha.get(team_id, "")
                if home_away not in ("home", "away"):
                    continue
                prefix = f"{home_away}_"

                # Find statistics group with MIN
                for stat_group in team_data.get("statistics", []):
                    labels = [l.upper() for l in stat_group.get("labels", [])]
                    if "MIN" not in labels:
                        continue

                    min_idx = labels.index("MIN")
                    minutes_list = []

                    for athlete in stat_group.get("athletes", []):
                        if athlete.get("didNotPlay", False):
                            continue
                        stats = athlete.get("stats", [])
                        if min_idx < len(stats):
                            try:
                                mins = int(stats[min_idx])
                                minutes_list.append(mins)
                            except (ValueError, TypeError):
                                continue

                    if minutes_list:
                        total_mins = sum(minutes_list)
                        players_used = sum(1 for m in minutes_list if m > 0)

                        # HHI = sum of squared shares
                        if total_mins > 0:
                            shares = [m / total_mins for m in minutes_list if m > 0]
                            hhi = sum(s ** 2 for s in shares)
                        else:
                            hhi = 0.0

                        result[f"{prefix}minutes_hhi"] = round(hhi, 4)
                        result[f"{prefix}players_used"] = players_used

            return result if result else None

        except requests.exceptions.RequestException:
            time.sleep(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit games to fetch (0=all)")
    parser.add_argument("--push-only", action="store_true", help="Skip fetch, push cached results")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't push")
    args = parser.parse_args()

    print("=" * 70)
    print("  FIX HHI + PLAYERS_USED — Re-extract from ESPN")
    print("=" * 70)

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Existing cache: {len(cache)} games")

    if not args.push_only:
        # Get game IDs from extract cache
        with open("ncaa_extract_cache.json") as f:
            extract_cache = json.load(f)
        all_game_ids = [gid for gid, v in extract_cache.items() if v]
        print(f"  Total games with ESPN data: {len(all_game_ids)}")

        # Skip already cached
        to_fetch = [gid for gid in all_game_ids if gid not in cache]
        if args.limit > 0:
            to_fetch = to_fetch[:args.limit]
        print(f"  Already cached: {len(cache)}")
        print(f"  To fetch: {len(to_fetch)}")

        if not to_fetch:
            print("  Nothing to fetch!")
        else:
            print(f"\n  Fetching player minutes from ESPN...")
            success = 0
            errors = 0
            t0 = time.time()

            for i, gid in enumerate(to_fetch):
                result = fetch_espn_minutes(gid)
                if result:
                    cache[gid] = result
                    success += 1
                else:
                    cache[gid] = {}
                    errors += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(to_fetch) - i - 1) / rate / 60
                    print(f"    {i+1}/{len(to_fetch)} | success:{success} errors:{errors} | {eta:.0f}min left")

                    # Save cache periodically
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cache, f)

                time.sleep(0.2)  # ~5 req/sec

            # Final cache save
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

            elapsed = time.time() - t0
            print(f"\n  Fetch complete: {success} success, {errors} errors in {elapsed/60:.1f}min")

    # Report
    valid = {gid: v for gid, v in cache.items() if v}
    print(f"\n  Valid results: {len(valid)}")

    # Sample
    for gid, v in list(valid.items())[:3]:
        print(f"    {gid}: {v}")

    # Stats
    home_hhi = [v["home_minutes_hhi"] for v in valid.values() if "home_minutes_hhi" in v]
    away_hhi = [v["away_minutes_hhi"] for v in valid.values() if "away_minutes_hhi" in v]
    home_pu = [v["home_players_used"] for v in valid.values() if "home_players_used" in v]
    if home_hhi:
        print(f"\n  HHI distribution: mean={np.mean(home_hhi):.3f}, min={min(home_hhi):.3f}, max={max(home_hhi):.3f}")
    if home_pu:
        print(f"  Players used: mean={np.mean(home_pu):.1f}, min={min(home_pu)}, max={max(home_pu)}")

    if args.dry_run:
        print(f"\n  DRY RUN — not pushing.")
        return

    # Push to Supabase
    print(f"\n  Pushing {len(valid)} patches to Supabase...")
    success = 0
    errors = 0
    t0 = time.time()

    for i, (gid, v) in enumerate(valid.items()):
        patch = {k: v2 for k, v2 in v.items() if v2 is not None}
        if not patch:
            continue

        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{gid}"
        try:
            r = requests.patch(url, headers=HEADERS, json=patch, timeout=15)
            if r.ok:
                success += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"    Error {r.status_code}: {r.text[:100]}")
        except Exception as e:
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(valid) - i - 1) / rate / 60
            print(f"    {i+1}/{len(valid)} | success:{success} errors:{errors} | {eta:.1f}min left")

        if (i + 1) % 50 == 0:
            time.sleep(0.2)

    elapsed = time.time() - t0
    print(f"\n  PUSH COMPLETE: {success:,} success, {errors} errors in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
