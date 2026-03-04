#!/usr/bin/env python3
"""
Historical Odds Backfill — The Odds API v4
One-time pull of historical spreads, moneylines, and totals for:
  NBA, NCAA Basketball, MLB, NFL, NCAAF

Stores consensus odds into Supabase historical tables.
Cost: ~10 quota units per request (1 region × 1 market group).
We request h2h,spreads,totals in one call = 30 units (10 × 3 markets × 1 region).

Usage:
  python3 historical_odds_backfill.py --api-key YOUR_KEY --sport nba --season 2024
  python3 historical_odds_backfill.py --api-key YOUR_KEY --all
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
import requests

# ── Supabase config ──
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

ODDS_API_BASE = "https://api.the-odds-api.com/v4/historical/sports"

# ── Sport configurations ──
SPORTS = {
    "nba": {
        "key": "basketball_nba",
        "table": "nba_historical",
        "home_col": "home_team",       # stores abbreviations (ATL, BOS, etc.)
        "away_col": "away_team",
        "id_col": "id",
        "seasons": {
            2025: ("2024-10-14", "2025-06-22"),  # Preseason games start mid-Oct
            2024: ("2023-10-14", "2024-06-17"),
            2023: ("2022-10-14", "2023-06-12"),
            2022: ("2021-10-14", "2022-06-16"),
            2021: ("2020-12-20", "2021-07-20"),
        },
    },
    "ncaa": {
        "key": "basketball_ncaab",
        "table": "ncaa_historical",
        "home_col": "home_team_abbr",  # NCAA uses _abbr columns
        "away_col": "away_team_abbr",
        "name_col_home": "home_team_name",  # also has full names for fuzzy matching
        "name_col_away": "away_team_name",
        "id_col": "id",
        "seasons": {
            2026: ("2025-11-03", "2026-04-07"),
            2025: ("2024-11-04", "2025-04-07"),
            2024: ("2023-11-06", "2024-04-08"),
            2023: ("2022-11-07", "2023-04-03"),
            2022: ("2021-11-09", "2022-04-04"),
        },
    },
    "mlb": {
        "key": "baseball_mlb",
        "table": "mlb_historical",
        "home_col": "home_team",       # stores abbreviations
        "away_col": "away_team",
        "id_col": "game_pk",
        "seasons": {
            2025: ("2025-03-18", "2025-10-01"),  # Tokyo/Seoul series can start early
            2024: ("2024-03-20", "2024-09-29"),  # Seoul series Mar 20-21
            2023: ("2023-03-08", "2023-10-01"),   # WBC + London series
            2022: ("2022-04-07", "2022-10-05"),
            2021: ("2021-04-01", "2021-10-03"),
        },
    },
    "nfl": {
        "key": "americanfootball_nfl",
        "table": "nfl_historical",
        "home_col": "home_team",
        "away_col": "away_team",
        "id_col": "id",
        "seasons": {
            2025: ("2025-09-04", "2026-02-08"),
            2024: ("2024-09-05", "2025-02-09"),
            2023: ("2023-09-07", "2024-02-11"),
            2022: ("2022-09-08", "2023-02-12"),
            2021: ("2021-09-09", "2022-02-13"),
        },
    },
    "ncaaf": {
        "key": "americanfootball_ncaaf",
        "table": "ncaaf_historical",
        "home_col": "home_team",
        "away_col": "away_team",
        "id_col": "id",
        "seasons": {
            2025: ("2025-08-23", "2026-01-20"),
            2024: ("2024-08-24", "2025-01-20"),
            2023: ("2023-08-26", "2024-01-08"),
            2022: ("2022-08-27", "2023-01-09"),
            2021: ("2021-08-28", "2022-01-10"),
        },
    },
}

# ── Team name normalization (Odds API uses full names, our tables use abbreviations) ──
NBA_NAME_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

MLB_NAME_MAP = {
    # Maps Odds API full names → baseball-reference abbreviations used in mlb_historical
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCA", "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYN", "New York Yankees": "NYA", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN", "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}

NFL_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS", "Washington Football Team": "WAS",
}

SPORT_NAME_MAPS = {
    "nba": NBA_NAME_MAP,
    "mlb": MLB_NAME_MAP,
    "nfl": NFL_NAME_MAP,
    "ncaaf": {},  # NCAAF uses full names in Odds API and likely in our table too
    "ncaa": {},   # NCAA uses full names — matched via name_col_home/name_col_away
}


def extract_consensus_odds(bookmakers, home_team):
    """
    Extract consensus spread, moneyline, total from bookmakers array.
    Uses median across all bookmakers for robustness.
    """
    home_spreads, home_mls, totals = [], [], []

    for bk in bookmakers:
        for market in bk.get("markets", []):
            key = market["key"]
            outcomes = market.get("outcomes", [])

            if key == "spreads":
                for o in outcomes:
                    if o["name"] == home_team and "point" in o:
                        home_spreads.append(o["point"])

            elif key == "h2h":
                for o in outcomes:
                    if o["name"] == home_team:
                        home_mls.append(o["price"])

            elif key == "totals":
                for o in outcomes:
                    if o["name"] == "Over" and "point" in o:
                        totals.append(o["point"])

    result = {}
    if home_spreads:
        home_spreads.sort()
        result["market_spread_home"] = home_spreads[len(home_spreads) // 2]
    if home_mls:
        home_mls.sort()
        result["market_home_ml"] = home_mls[len(home_mls) // 2]
    if totals:
        totals.sort()
        result["market_ou_total"] = totals[len(totals) // 2]

    return result


def fetch_historical_odds(api_key, sport_key, date_str, retries=3):
    """
    Fetch historical odds snapshot for a given sport and date.
    Returns list of game dicts with odds.
    """
    url = f"{ODDS_API_BASE}/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "date": f"{date_str}T12:00:00Z",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"    Timeout on {date_str}, retry {attempt + 1}...")
                time.sleep(5)
                continue
            else:
                print(f"    Timeout on {date_str}, skipping after {retries} retries")
                return [], "?", "?"
        except requests.exceptions.RequestException as e:
            print(f"    Request error on {date_str}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return [], "?", "?"

        # Track quota usage from headers
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")

        if resp.status_code == 422:
            return [], remaining, used
        if resp.status_code == 429:
            print(f"    Rate limited — waiting 60s...")
            time.sleep(60)
            return fetch_historical_odds(api_key, sport_key, date_str, retries)
        if resp.status_code != 200:
            print(f"    API error {resp.status_code}: {resp.text[:200]}")
            return [], remaining, used

        data = resp.json()
        games = data.get("data", [])
        return games, remaining, used

    return [], "?", "?"


def supabase_update(table, match_filter, update_data):
    """Update a row in Supabase by match filter."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_filter}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    resp = requests.patch(url, json=update_data, headers=headers, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"    UPDATE FAILED {resp.status_code}: {match_filter} → {resp.text[:100]}")
    return resp.status_code in (200, 204)
    resp = requests.patch(url, json=update_data, headers=headers, timeout=10)
    return resp.status_code in (200, 204)


def supabase_select(table, query, limit=10000):
    """Select rows from Supabase with pagination (1000 per page)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("    WARNING: SUPABASE_URL or SUPABASE_KEY not set!")
        return []
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "count=exact",
    }
    all_rows = []
    page_size = 1000
    offset = 0
    while offset < limit:
        url = f"{SUPABASE_URL}/rest/v1/{table}?{query}&limit={page_size}&offset={offset}"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code not in (200, 206):
            print(f"    Supabase error {resp.status_code}: {resp.text[:200]}")
            break
        page = resp.json()
        if not page:
            break
        all_rows.extend(page)
        if len(page) < page_size:
            break  # Last page
        offset += page_size
    return all_rows


def backfill_sport(api_key, sport_name, season, dry_run=False):
    """
    Backfill historical odds for a sport and season.
    Steps:
      1. Generate list of dates in the season
      2. For each date, fetch odds snapshot
      3. Match games to existing rows in historical table
      4. Update rows with market_spread_home, market_ou_total, market_home_ml
    """
    cfg = SPORTS[sport_name]
    sport_key = cfg["key"]
    table = cfg["table"]
    season_dates = cfg["seasons"].get(season)

    if not season_dates:
        print(f"  No season dates configured for {sport_name} {season}")
        return 0

    start_str, end_str = season_dates
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    # For NBA/MLB: games happen most days, sample every day
    # For NFL/NCAAF: games happen on specific days, sample every day but most will be empty
    total_days = (end - start).days + 1
    print(f"\n{'='*60}")
    print(f"  {sport_name.upper()} {season}: {start_str} → {end_str} ({total_days} days)")
    print(f"  Table: {table}")
    print(f"{'='*60}")

    # Pre-fetch existing rows to build match index
    home_col = cfg["home_col"]
    away_col = cfg["away_col"]
    id_col = cfg["id_col"]
    name_col_home = cfg.get("name_col_home")  # NCAA has full names too
    name_col_away = cfg.get("name_col_away")

    select_cols = f"{id_col},{home_col},{away_col},game_date,market_spread_home,market_ou_total"
    if name_col_home:
        select_cols += f",{name_col_home},{name_col_away}"

    print(f"  Loading existing {table} rows for matching...")
    existing = supabase_select(
        table,
        f"select={select_cols}&season=eq.{season}",
        limit=10000,
    )
    print(f"  Found {len(existing)} existing rows")

    # Build match index: multiple keys per row for flexible matching
    # Key formats: (team_identifier, team_identifier, date) -> row_id
    match_index = {}
    already_has_odds = 0
    for row in existing:
        gd = str(row.get("game_date", ""))[:10]
        home = row.get(home_col, "")
        away = row.get(away_col, "")
        row_id = row.get(id_col)
        if home and away and gd and row_id:
            match_index[(home, away, gd)] = row_id
            # Also index by full name if available (NCAA)
            if name_col_home:
                hname = row.get(name_col_home, "")
                aname = row.get(name_col_away, "")
                if hname and aname:
                    match_index[(hname, aname, gd)] = row_id
            if row.get("market_spread_home") is not None:
                already_has_odds += 1

    if already_has_odds > 0:
        print(f"  {already_has_odds} rows already have odds — will skip those")

    # Build set of rows that already have odds (don't overwrite)
    skip_ids = set()
    for row in existing:
        if row.get("market_spread_home") is not None:
            skip_ids.add(row.get(id_col))

    # Instead of querying every calendar day, only query dates that have games
    # This saves ~30-40% of API calls (skips off-days, All-Star break, etc.)
    game_dates = set()
    for row in existing:
        gd = str(row.get("game_date", ""))[:10]
        if gd and start_str <= gd <= end_str:
            game_dates.add(gd)

    dates_to_query = sorted(game_dates)
    print(f"  {len(dates_to_query)} actual game dates (vs {total_days} calendar days — saving {total_days - len(dates_to_query)} empty days)")

    # Get name map for this sport
    name_map = SPORT_NAME_MAPS.get(sport_name, {})

    matched = 0
    unmatched = 0
    written = 0
    skipped_dates = 0
    api_calls = 0

    current_idx = 0
    while current_idx < len(dates_to_query):
        date_str = dates_to_query[current_idx]
        games, remaining, used = fetch_historical_odds(api_key, sport_key, date_str)
        api_calls += 1

        if not games:
            skipped_dates += 1
            current += timedelta(days=1)
            time.sleep(0.25)  # Gentle rate limiting
            continue

        day_matched = 0
        for game in games:
            home_full = game.get("home_team", "")
            away_full = game.get("away_team", "")
            commence = game.get("commence_time", "")[:10]
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            odds = extract_consensus_odds(bookmakers, home_full)
            if not odds:
                continue

            # Try to match: full name first, then abbreviation
            # Also try commence_date - 1 day to handle UTC offset
            # (e.g. 10 PM ET game on Mar 28 → commence_time 2024-03-29T02:00Z)
            home_abbr = name_map.get(home_full, home_full)
            away_abbr = name_map.get(away_full, away_full)

            commence_prev = ""
            if len(commence) == 10:
                try:
                    commence_prev = (datetime.strptime(commence, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                except ValueError:
                    pass

            row_id = (
                match_index.get((home_full, away_full, commence)) or
                match_index.get((home_abbr, away_abbr, commence)) or
                match_index.get((home_full, away_full, date_str)) or
                match_index.get((home_abbr, away_abbr, date_str)) or
                # UTC offset fallback: try previous day
                match_index.get((home_full, away_full, commence_prev)) or
                match_index.get((home_abbr, away_abbr, commence_prev))
            )

            if row_id:
                if row_id in skip_ids:
                    day_matched += 1
                    matched += 1
                    continue  # Already has odds
                if not dry_run:
                    ok = supabase_update(table, f"{id_col}=eq.{row_id}", odds)
                    if ok:
                        written += 1
                day_matched += 1
                matched += 1
            else:
                unmatched += 1

        print(f"  {date_str}: {len(games)} games, {day_matched} matched | "
              f"quota remaining: {remaining}, used: {used}")

        current_idx += 1
        time.sleep(0.3)  # Stay under rate limit

        # Safety check: stop if quota is getting low
        try:
            if int(remaining) < 100:
                print(f"\n  ⚠️ Quota low ({remaining} remaining) — stopping to preserve quota")
                break
        except ValueError:
            pass

    print(f"\n  Summary: {api_calls} API calls, {matched} matched, "
          f"{written} written, {unmatched} unmatched, {skipped_dates} empty dates")
    return written


def estimate_quota(sports, seasons):
    """Estimate total API quota needed."""
    total_days = 0
    for sport in sports:
        cfg = SPORTS[sport]
        for season in seasons:
            dates = cfg["seasons"].get(season)
            if dates:
                start = datetime.strptime(dates[0], "%Y-%m-%d")
                end = datetime.strptime(dates[1], "%Y-%m-%d")
                days = (end - start).days + 1
                total_days += days

    # Each call costs 10 quota units (1 region × h2h+spreads+totals)
    # But we only query dates with actual games, not every calendar day
    quota_cost = total_days * 10
    print(f"\nQuota Estimate (worst case — every calendar day):")
    print(f"  Total calendar days: {total_days}")
    print(f"  Cost per day: 30 units (10 × 3 markets × 1 region)")
    print(f"  Worst case cost: {quota_cost:,} units")
    print(f"  Actual cost ~60-70% of this (game days only)")
    print(f"  Mega plan (40,000/mo) covers: {'Yes' if quota_cost < 40000 else 'Tight — run in priority order'}")
    return quota_cost


def main():
    parser = argparse.ArgumentParser(description="Historical odds backfill")
    parser.add_argument("--api-key", required=True, help="The Odds API key")
    parser.add_argument("--sport", choices=list(SPORTS.keys()) + ["all"],
                        default="all", help="Sport to backfill")
    parser.add_argument("--season", type=int, help="Specific season (e.g., 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Supabase")
    parser.add_argument("--estimate", action="store_true", help="Just estimate quota cost")

    args = parser.parse_args()

    # Determine sports and seasons
    sports = list(SPORTS.keys()) if args.sport == "all" else [args.sport]
    if args.season:
        seasons = [args.season]
    else:
        seasons = [2021, 2022, 2023, 2024, 2025]

    if args.estimate:
        estimate_quota(sports, seasons)
        return

    # Load Supabase creds from env or .env file
    if not SUPABASE_URL:
        print("Set SUPABASE_URL and SUPABASE_KEY environment variables")
        print("Or create a .env file with these values")
        sys.exit(1)

    print(f"Backfilling historical odds")
    print(f"  Sports: {', '.join(sports)}")
    print(f"  Seasons: {', '.join(map(str, seasons))}")
    print(f"  Dry run: {args.dry_run}")

    estimate_quota(sports, seasons)

    total_matched = 0
    for sport in sports:
        for season in seasons:
            try:
                n = backfill_sport(args.api_key, sport, season, dry_run=args.dry_run)
                total_matched += n
            except Exception as e:
                print(f"  ERROR: {sport} {season}: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_matched} rows updated with historical odds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
