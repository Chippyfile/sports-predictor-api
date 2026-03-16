#!/usr/bin/env python3
"""
backfill_conferences_2026.py — Populate home_conference/away_conference for 2026 season.

Fetches conference names from ESPN team endpoint, then patches ncaa_historical.
Run this BEFORE ncaa_pit_backfill_v3.py so centrality/conf_balance compute correctly.

Usage:
    python3 backfill_conferences_2026.py [--dry-run]
"""

import os, sys, time, json, requests

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def sb_get_all(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=30)
        if not r.ok: break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data


def fetch_team_conference(team_id):
    """Get conference ID from ESPN team endpoint."""
    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}", timeout=10)
        if not r.ok:
            return ""
        data = r.json()
        team = data.get("team", data)
        groups = team.get("groups", {})
        # ESPN returns: {"id": "7", "parent": {"id": "50"}, "isConference": True}
        # group id is the conference identifier (e.g. 7 = Big Ten, 23 = SEC)
        conf_id = groups.get("id", "")
        return f"conf_{conf_id}" if conf_id else ""
    except Exception as e:
        print(f"    ESPN error for {team_id}: {e}")
        return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CONFERENCE BACKFILL — 2026 Season")
    print("=" * 60)

    # Step 1: Get unique team IDs from 2026 games
    print("  Fetching 2026 games...")
    games = sb_get_all("ncaa_historical",
                       "season=eq.2026&select=game_id,home_team_id,away_team_id,home_conference,away_conference")
    print(f"  {len(games)} games in 2026 season")

    # Collect unique team IDs
    team_ids = set()
    for g in games:
        if g.get("home_team_id"):
            team_ids.add(str(g["home_team_id"]))
        if g.get("away_team_id"):
            team_ids.add(str(g["away_team_id"]))
    print(f"  {len(team_ids)} unique teams")

    # Step 2: Fetch conference for each team from ESPN
    print(f"\n  Fetching conferences from ESPN...")
    team_conf = {}
    for i, tid in enumerate(sorted(team_ids)):
        conf = fetch_team_conference(tid)
        if conf:
            team_conf[tid] = conf
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(team_ids)} — {len(team_conf)} conferences found")
        time.sleep(0.15)  # Rate limit

    print(f"  Found conferences for {len(team_conf)}/{len(team_ids)} teams")

    # Show distribution
    from collections import Counter
    conf_counts = Counter(team_conf.values())
    print(f"\n  Top conferences:")
    for conf, count in conf_counts.most_common(15):
        print(f"    {conf}: {count} teams")

    # Save mapping for reuse
    with open("team_conferences_2026.json", "w") as f:
        json.dump(team_conf, f, indent=2)
    print(f"\n  Saved mapping to team_conferences_2026.json")

    if args.dry_run:
        print("  DRY RUN — not pushing to Supabase")
        return

    # Step 3: Patch each game with conference data
    print(f"\n  Patching {len(games)} games...")
    session = requests.Session()
    session.headers.update(HEADERS)

    success, skipped, errors = 0, 0, 0
    for i, g in enumerate(games):
        gid = g.get("game_id")
        h_tid = str(g.get("home_team_id", ""))
        a_tid = str(g.get("away_team_id", ""))
        h_conf = team_conf.get(h_tid, "")
        a_conf = team_conf.get(a_tid, "")

        if not h_conf and not a_conf:
            skipped += 1
            continue

        patch = {}
        if h_conf:
            patch["home_conference"] = h_conf
        if a_conf:
            patch["away_conference"] = a_conf

        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{gid}"
        try:
            r = session.patch(url, json=patch, timeout=15)
            if r.ok:
                success += 1
            else:
                errors += 1
        except:
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(games)} | success:{success} skip:{skipped} err:{errors}")
        time.sleep(0.02)

    session.close()
    print(f"\n{'=' * 60}")
    print(f"  CONFERENCE BACKFILL COMPLETE")
    print(f"  Updated: {success} | Skipped: {skipped} | Errors: {errors}")
    print(f"{'=' * 60}")
    print(f"  NEXT: python3 ncaa_pit_backfill_v3.py --push-season 2026")


if __name__ == "__main__":
    main()
