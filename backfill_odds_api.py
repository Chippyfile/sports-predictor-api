#!/usr/bin/env python3
"""
backfill_odds_api.py — Backfill opening/closing spreads from The Odds API
==========================================================================
Pulls historical odds snapshots (morning=opening, evening=closing) for each
game day from 2020-2025, matches to Supabase games by team name + date,
and pushes spread_movement, closing_spread, ml_movement features.

New Supabase columns:
  odds_api_spread_open, odds_api_spread_close, odds_api_spread_movement
  odds_api_ml_home_open, odds_api_ml_home_close
  odds_api_total_open, odds_api_total_close, odds_api_total_movement

Cost: ~2 requests per game day × ~150 days/season × 5 seasons = ~1,500 requests

Usage:
  python3 -u backfill_odds_api.py --check       # estimate cost, no API calls
  python3 -u backfill_odds_api.py --test-day     # test one day (2 requests)
  python3 -u backfill_odds_api.py                # full backfill
  python3 -u backfill_odds_api.py --season 2024  # single season
"""
import sys, os, json, time, argparse
from datetime import datetime, timedelta
import requests as req
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

API_KEY = "49c82dfaa93f161fbc28de41ea6a44c4"
ODDS_BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_ncaab/odds"

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

CACHE_FILE = "odds_api_cache.json"

# Team name mapping: Odds API name → ESPN/Supabase abbreviation
# We'll match by fuzzy team name + date instead of hardcoding
TEAM_ALIASES = {}  # built dynamically


def get_game_days():
    """Get all unique game dates from Supabase."""
    rows = sb_get("ncaa_historical", "select=game_date&order=game_date.asc")
    dates = sorted(set(r["game_date"] for r in rows if r.get("game_date")))
    return dates


def get_games_for_date(date_str):
    """Get all games for a specific date from Supabase."""
    rows = sb_get("ncaa_historical",
                   f"game_date=eq.{date_str}&select=game_id,home_team_abbr,away_team_abbr,home_team_id,away_team_id")
    return rows


def fetch_odds_snapshot(date_str, time_of_day="open"):
    """Fetch odds snapshot from The Odds API for a given date."""
    # Opening: morning of game day; Closing: late evening
    if time_of_day == "open":
        ts = f"{date_str}T14:00:00Z"  # 9 AM ET  
    else:
        ts = f"{date_str}T22:00:00Z"  # 5 PM ET (before most tips, captures closing)

    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "spreads,h2h,totals",
        "oddsFormat": "american",
        "date": ts,
        "bookmakers": "draftkings",
    }

    try:
        r = req.get(ODDS_BASE, params=params, timeout=15)
        if r.status_code == 422:
            # Date not available
            return None, r.headers.get("x-requests-remaining", "?")
        if not r.ok:
            return None, r.headers.get("x-requests-remaining", "?")

        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        games = data.get("data", [])
        return games, remaining
    except Exception as e:
        print(f"    Error fetching {date_str} {time_of_day}: {e}")
        return None, "?"


def parse_odds_game(game):
    """Parse a single game from Odds API response."""
    result = {
        "home_team": game.get("home_team", ""),
        "away_team": game.get("away_team", ""),
        "commence_time": game.get("commence_time", ""),
    }

    for bk in game.get("bookmakers", []):
        if bk["key"] != "draftkings":
            continue

        for mkt in bk.get("markets", []):
            if mkt["key"] == "spreads":
                for o in mkt["outcomes"]:
                    if o["name"] == game.get("home_team"):
                        result["spread_home"] = o.get("point", 0)
                    elif o["name"] == game.get("away_team"):
                        result["spread_away"] = o.get("point", 0)

            elif mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    if o["name"] == game.get("home_team"):
                        result["ml_home"] = o.get("price", 0)
                    elif o["name"] == game.get("away_team"):
                        result["ml_away"] = o.get("price", 0)

            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    if o["name"] == "Over":
                        result["total"] = o.get("point", 0)
                    break  # just need the line once

    return result


def normalize_team_name(name):
    """Normalize team name for matching."""
    # Remove common suffixes and clean up
    name = name.lower().strip()
    for suffix in [" bulldogs", " tigers", " bears", " eagles", " hawks",
                   " wildcats", " warriors", " knights", " panthers", " lions",
                   " cougars", " huskies", " devils", " blue devils",
                   " terriers", " colonels", " cardinals", " gaels",
                   " waves", " dons", " vikings", " vandals", " bobcats",
                   " bengals", " grizzlies", " colonials", " seahawks",
                   " paladins", " demons", " islanders", " penguins",
                   " golden", " st", " state"]:
        pass  # keep full name for matching
    return name


def match_game_to_supabase(odds_game, supabase_games, team_map):
    """Match an Odds API game to a Supabase game by team name."""
    home = odds_game["home_team"].lower()
    away = odds_game["away_team"].lower()

    for sg in supabase_games:
        sb_home = (team_map.get(sg.get("home_team_id", ""), "") or
                   sg.get("home_team_abbr", "") or "").lower()
        sb_away = (team_map.get(sg.get("away_team_id", ""), "") or
                   sg.get("away_team_abbr", "") or "").lower()

        # Try substring matching
        if ((sb_home and sb_home in home) or (home and any(w in sb_home for w in home.split()[:2]))) and \
           ((sb_away and sb_away in away) or (away and any(w in sb_away for w in away.split()[:2]))):
            return sg["game_id"]

    return None


def build_team_name_map():
    """Build mapping of team_id to full name from Odds API data."""
    # We'll build this dynamically as we encounter games
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Estimate cost only")
    parser.add_argument("--test-day", action="store_true", help="Test one day (2 requests)")
    parser.add_argument("--season", type=int, help="Backfill single season (e.g., 2024)")
    parser.add_argument("--start-date", type=str, help="Start from this date (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BACKFILL HISTORICAL ODDS (The Odds API)")
    print("=" * 70)

    # Get all game dates
    print("\n  Loading game dates from Supabase...")
    all_dates = get_game_days()
    print(f"  Total game dates: {len(all_dates)}")

    # Filter to 2020+ (Odds API coverage starts late 2020)
    all_dates = [d for d in all_dates if "2023-11-01" <= d <= "2025-10-31"]
    print(f"  Game dates 2024-2025: {len(all_dates)}")

    if args.season:
        # Season runs Nov of prior year to April
        season_start = f"{args.season - 1}-11-01"
        season_end = f"{args.season}-04-30"
        all_dates = [d for d in all_dates if season_start <= d <= season_end]
        print(f"  Season {args.season}: {len(all_dates)} game dates")

    if args.start_date:
        all_dates = [d for d in all_dates if d >= args.start_date]
        print(f"  Starting from {args.start_date}: {len(all_dates)} game dates")

    estimated_requests = len(all_dates) * 2
    print(f"  Estimated API requests: {estimated_requests}")
    print(f"  Current remaining: ~22,800")

    if args.check:
        print(f"\n  CHECK ONLY — no API calls made.")
        return

    if args.test_day:
        all_dates = all_dates[-1:]  # most recent
        print(f"  TEST MODE: 1 day only")

    # Load cache
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded cache: {len(cache)} entries")
    except:
        cache = {}

    # Process each game day
    total_matched = 0
    total_unmatched = 0
    total_requests = 0
    patches = []

    for di, date_str in enumerate(all_dates):
        cache_key = f"odds_{date_str}"

        if cache_key in cache:
            # Use cached data
            day_patches = cache[cache_key]
            patches.extend(day_patches)
            total_matched += len(day_patches)
            continue

        # Fetch opening and closing snapshots
        open_games, remaining = fetch_odds_snapshot(date_str, "open")
        if open_games is None:
            cache[cache_key] = []
            continue
        total_requests += 1

        time.sleep(0.2)  # rate limit

        close_games, remaining = fetch_odds_snapshot(date_str, "close")
        if close_games is None:
            close_games = []
        total_requests += 1

        # Parse opening lines
        open_lines = {}
        for g in open_games:
            parsed = parse_odds_game(g)
            key = f"{parsed['home_team']}|{parsed['away_team']}"
            open_lines[key] = parsed

        # Parse closing lines
        close_lines = {}
        for g in close_games:
            parsed = parse_odds_game(g)
            key = f"{parsed['home_team']}|{parsed['away_team']}"
            close_lines[key] = parsed

        # Match opening + closing for same game
        day_patches = []
        matched_keys = set(open_lines.keys()) & set(close_lines.keys())

        for key in matched_keys:
            op = open_lines[key]
            cl = close_lines[key]

            patch = {
                "home_team_odds_api": op["home_team"],
                "away_team_odds_api": op["away_team"],
                "date": date_str,
            }

            # Spread movement
            if "spread_home" in op and "spread_home" in cl:
                patch["odds_api_spread_open"] = op["spread_home"]
                patch["odds_api_spread_close"] = cl["spread_home"]
                patch["odds_api_spread_movement"] = round(cl["spread_home"] - op["spread_home"], 1)

            # Moneyline
            if "ml_home" in op and "ml_home" in cl:
                patch["odds_api_ml_home_open"] = op["ml_home"]
                patch["odds_api_ml_home_close"] = cl["ml_home"]
                if "ml_away" in op:
                    patch["odds_api_ml_away_open"] = op["ml_away"]
                if "ml_away" in cl:
                    patch["odds_api_ml_away_close"] = cl["ml_away"]

            # Total movement
            if "total" in op and "total" in cl:
                patch["odds_api_total_open"] = op["total"]
                patch["odds_api_total_close"] = cl["total"]
                patch["odds_api_total_movement"] = round(cl["total"] - op["total"], 1)

            # Sanity check: movements > 6 pts are likely mismatches
            mvmt = abs(patch.get("odds_api_spread_movement", 0))
            if mvmt > 6:
                continue  # skip likely mismatch
            if len(patch) > 3:  # more than just team names + date
                day_patches.append(patch)
                total_matched += 1

        cache[cache_key] = day_patches
        patches.extend(day_patches)

        # Save cache periodically
        if (di + 1) % 10 == 0 or di == len(all_dates) - 1:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        if (di + 1) % 20 == 0 or di == len(all_dates) - 1:
            print(f"    {di+1}/{len(all_dates)} dates | {total_matched} matched | "
                  f"{total_requests} API calls | remaining: {remaining}")

        time.sleep(0.2)

    print(f"\n  Total matched games with spread movement: {total_matched}")
    print(f"  Total API requests used: {total_requests}")

    # Show sample
    if patches:
        print(f"\n  Sample data:")
        for p in patches[:10]:
            mvmt = p.get("odds_api_spread_movement", "N/A")
            print(f"    {p['date']} {p['home_team_odds_api']}: "
                  f"spread {p.get('odds_api_spread_open','?')}→{p.get('odds_api_spread_close','?')} "
                  f"(mvmt={mvmt})")

    if args.test_day:
        print(f"\n  TEST MODE — no Supabase push.")
        return

    # ── Match to Supabase games and push ────────────────────────────
    print(f"\n  Matching to Supabase games...")

    # Load Supabase games with team names for matching
    sb_games = sb_get("ncaa_historical",
                       "select=game_id,game_date,home_team_abbr,away_team_abbr&order=game_date.asc")
    sb_by_date = {}
    for g in sb_games:
        d = g.get("game_date", "")
        if d not in sb_by_date:
            sb_by_date[d] = []
        sb_by_date[d].append(g)

    push_patches = []
    unmatched = 0

    for p in patches:
        date = p["date"]
        home = p["home_team_odds_api"].lower()
        away = p["away_team_odds_api"].lower()

        sb_day = sb_by_date.get(date, [])
        matched_id = None

        for sg in sb_day:
            sb_home = (sg.get("home_team_abbr") or "").lower()
            sb_away = (sg.get("away_team_abbr") or "").lower()

            # Fuzzy match: check if abbreviation appears in full name
            # e.g., "gonz" in "gonzaga bulldogs" or "gonzaga" contains "gonz"
            home_match = (sb_home and (sb_home in home or
                          any(word.startswith(sb_home[:4]) for word in home.split() if len(sb_home) >= 3)))
            away_match = (sb_away and (sb_away in away or
                          any(word.startswith(sb_away[:4]) for word in away.split() if len(sb_away) >= 3)))

            if home_match and away_match:
                matched_id = sg["game_id"]
                break

        if matched_id:
            push_data = {k: v for k, v in p.items()
                         if k not in ["home_team_odds_api", "away_team_odds_api", "date"]}
            push_patches.append({"game_id": matched_id, **push_data})
        else:
            unmatched += 1

    print(f"  Matched to Supabase: {len(push_patches)}")
    print(f"  Unmatched: {unmatched}")

    # Push to Supabase
    if push_patches:
        print(f"\n  Pushing {len(push_patches)} patches...")
        success, errors = 0, 0
        for i, patch in enumerate(push_patches):
            game_id = patch.pop("game_id")
            try:
                url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
                r = req.patch(url, headers=SUPABASE_HEADERS, json=patch, timeout=30)
                if r.ok:
                    success += 1
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"    Error on {game_id}: {r.text[:100]}")
            except:
                errors += 1

            if (i + 1) % 500 == 0 or i == len(push_patches) - 1:
                print(f"    {i+1}/{len(push_patches)} | success:{success} errors:{errors}")

        print(f"\n  PUSH COMPLETE: {success} success, {errors} errors")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
