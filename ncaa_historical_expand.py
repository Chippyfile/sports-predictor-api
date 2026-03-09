#!/usr/bin/env python3
"""
ncaa_historical_expand.py — Expand NCAA Historical Data to 2015-2025
════════════════════════════════════════════════════════════════════
Fetches game results from ESPN scoreboard API for seasons 2015-2019,
2021-2025 (skipping 2020 COVID) and inserts into ncaa_historical.

This populates the BASIC game data (teams, scores, dates, conference,
ranks). The v3 PIT backfill then processes these games to compute
all 107 features from box scores.

Run:
  SUPABASE_ANON_KEY="..." python3 ncaa_historical_expand.py
  SUPABASE_ANON_KEY="..." python3 ncaa_historical_expand.py --season 2017
  SUPABASE_ANON_KEY="..." python3 ncaa_historical_expand.py --resume
"""
import os, sys, json, time, argparse, requests
from datetime import datetime, timedelta

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
PROGRESS_FILE = "ncaa_expand_progress.json"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def fetch_scoreboard(date_str):
    """Fetch all games for a given date (YYYYMMDD format)."""
    url = f"{ESPN_SCOREBOARD}?dates={date_str}&limit=200&groups=50"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
        except requests.exceptions.RequestException:
            time.sleep(1)
    return None


def parse_scoreboard(data):
    """Parse ESPN scoreboard into game records for ncaa_historical."""
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_team = away_team = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_team = c
            else:
                away_team = c

        if not home_team or not away_team:
            continue

        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        game_id = event.get("id", "")
        game_date_raw = event.get("date", "")

        try:
            game_date = game_date_raw[:10]
        except:
            continue

        try:
            dt = datetime.strptime(game_date, "%Y-%m-%d")
            if dt.month >= 8:
                season = dt.year + 1
            else:
                season = dt.year
        except:
            season = 0

        home_score = int(home_team.get("score", 0) or 0)
        away_score = int(away_team.get("score", 0) or 0)

        if home_score == 0 and away_score == 0:
            continue

        home_team_data = home_team.get("team", {})
        away_team_data = away_team.get("team", {})

        home_id = str(home_team_data.get("id", ""))
        away_id = str(away_team_data.get("id", ""))

        home_rank = None
        away_rank = None
        for c, setter in [(home_team, "home"), (away_team, "away")]:
            cur_rank = c.get("curatedRank", {}).get("current", 99)
            if cur_rank and cur_rank <= 25:
                if setter == "home":
                    home_rank = cur_rank
                else:
                    away_rank = cur_rank

        neutral = competition.get("neutralSite", False)
        season_type = event.get("season", {}).get("type", 2)
        is_postseason = season_type == 3

        game_record = {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "home_win": 1 if home_score > away_score else 0,
            "neutral_site": 1 if neutral else 0,
            "is_postseason": 1 if is_postseason else 0,
            "home_rank": home_rank if home_rank else None,
            "away_rank": away_rank if away_rank else None,
        }

        games.append(game_record)

    return games


def sb_upsert_batch(table, rows):
    """Upsert a batch of rows (conflict on game_id)."""
    if not rows:
        return 0

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    success = 0
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        r = requests.post(url, headers=headers, json=batch, timeout=30)
        if r.ok:
            success += len(batch)
        else:
            print(f"    Upsert error: {r.text[:200]}")
    return success


def get_existing_game_ids():
    """Get all game_ids already in ncaa_historical."""
    existing = set()
    offset, limit = 0, 1000
    while True:
        url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?select=game_id&limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=30)
        if not r.ok:
            break
        data = r.json()
        if not data:
            break
        for row in data:
            if row.get("game_id"):
                existing.add(str(row["game_id"]))
        if len(data) < limit:
            break
        offset += limit
    return existing


def generate_season_dates(season):
    """
    Generate all dates for a college basketball season.
    Season 2019 = Nov 2018 through April 2019.
    """
    start = datetime(season - 1, 11, 1)
    end = datetime(season, 4, 15)

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="Expand NCAA historical to 2015-2025")
    parser.add_argument("--season", type=int, help="Process single season (e.g. 2017)")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't insert")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAA HISTORICAL EXPANSION — 2015-2025")
    print("  Target: ~55,000 games (from ~4,185)")
    print("=" * 70)

    if args.season:
        seasons = [args.season]
    else:
        seasons = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

    progress = {}
    if args.resume and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        print(f"  Resuming from progress file")

    print(f"  Loading existing game IDs from ncaa_historical...")
    existing_ids = get_existing_game_ids()
    print(f"  Found {len(existing_ids)} existing games")

    total_inserted = 0
    total_fetched = 0
    total_skipped = 0
    total_errors = 0

    for season in seasons:
        completed_dates = set(progress.get(str(season), []))
        dates = generate_season_dates(season)
        remaining_dates = [d for d in dates if d not in completed_dates]

        print(f"\n  Season {season}: {len(dates)} dates, {len(completed_dates)} done, {len(remaining_dates)} remaining")

        season_games = []

        for di, date_str in enumerate(remaining_dates):
            data = fetch_scoreboard(date_str)

            if data:
                games = parse_scoreboard(data)
                new_games = [g for g in games if str(g["game_id"]) not in existing_ids]

                for g in new_games:
                    existing_ids.add(str(g["game_id"]))
                season_games.extend(new_games)
                total_fetched += len(games)
                total_skipped += len(games) - len(new_games)
            else:
                total_errors += 1

            if str(season) not in progress:
                progress[str(season)] = []
            progress[str(season)].append(date_str)

            if (di + 1) % 10 == 0:
                print(f"    {date_str} — {di+1}/{len(remaining_dates)} dates, "
                      f"{len(season_games)} new games this season")

            if len(season_games) >= 2000:
                if not args.dry_run:
                    inserted = sb_upsert_batch("ncaa_historical", season_games)
                    total_inserted += inserted
                    print(f"    → Inserted {inserted} games")
                season_games = []

                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress, f)

            time.sleep(0.15)

        if season_games and not args.dry_run:
            inserted = sb_upsert_batch("ncaa_historical", season_games)
            total_inserted += inserted
            print(f"    → Inserted {inserted} games")

        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)

        print(f"  Season {season} complete")

    print(f"\n{'='*70}")
    print(f"  EXPANSION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total fetched:  {total_fetched}")
    print(f"  Total new:      {total_inserted}")
    print(f"  Total skipped:  {total_skipped} (already existed)")
    print(f"  Total errors:   {total_errors}")
    print(f"  New total:      {len(existing_ids)}")
    print(f"")
    print(f"  NEXT STEPS:")
    print(f"  1. Run v3 PIT backfill on expanded dataset:")
    print(f"     python3 ncaa_pit_backfill_v3.py --fetch-only  (box scores, ~3hrs)")
    print(f"     python3 ncaa_pit_backfill_v3.py               (compute + push)")
    print(f"  2. Retrain: curl -X POST $RAILWAY_API/train/ncaa")
    print(f"  3. Backtest: curl -X POST $RAILWAY_API/backtest/ncaa")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
