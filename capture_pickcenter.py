#!/usr/bin/env python3
"""
capture_pickcenter.py — Save DraftKings open/close lines from ESPN pickcenter
==============================================================================
Runs against current season games to accumulate spread movement, moneyline,
and total line data for future training.

New Supabase columns (will be created if needed):
  dk_spread_open, dk_spread_close, dk_spread_movement
  dk_ml_home_open, dk_ml_home_close, dk_ml_away_open, dk_ml_away_close
  dk_total_open, dk_total_close, dk_total_movement

Usage:
  python3 -u capture_pickcenter.py --check      # test on 20 games, no push
  python3 -u capture_pickcenter.py               # capture + push all current season
  python3 -u capture_pickcenter.py --today        # only today's games
  python3 -u capture_pickcenter.py --backfill     # all games this season

Run daily via cron or after each game day:
  0 6 * * * cd ~/Desktop/sports-predictor-api && python3 capture_pickcenter.py >> pickcenter.log 2>&1
"""
import sys, os, json, time, argparse
import requests
import pandas as pd

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}

CACHE_FILE = "pickcenter_cache.json"


def extract_pickcenter(game_id):
    """Extract DraftKings open/close lines from ESPN pickcenter for one game."""
    url = f"{ESPN_BASE}/summary?event={game_id}"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        if not r.ok:
            return None
        data = r.json()
    except:
        return None

    pc = data.get("pickcenter", [])
    if not pc or not isinstance(pc, list):
        return None

    result = {}
    for entry in pc:
        # Point Spread
        ps = entry.get("pointSpread", {})
        if ps:
            home = ps.get("home", {})
            h_open = home.get("open", {}).get("line")
            h_close = home.get("close", {}).get("line")
            h_open_odds = home.get("open", {}).get("odds")
            h_close_odds = home.get("close", {}).get("odds")

            if h_open is not None and h_close is not None:
                try:
                    result["dk_spread_open"] = float(h_open)
                    result["dk_spread_close"] = float(h_close)
                    result["dk_spread_movement"] = round(float(h_close) - float(h_open), 1)
                except:
                    pass
            if h_open_odds:
                result["dk_spread_open_odds"] = str(h_open_odds)
            if h_close_odds:
                result["dk_spread_close_odds"] = str(h_close_odds)

        # Moneyline
        ml = entry.get("moneyline", {})
        if ml:
            for side in ["home", "away"]:
                s = ml.get(side, {})
                for timing in ["open", "close"]:
                    odds = s.get(timing, {}).get("odds")
                    if odds:
                        try:
                            result[f"dk_ml_{side}_{timing}"] = int(odds)
                        except:
                            result[f"dk_ml_{side}_{timing}"] = str(odds)

        # Total
        total = entry.get("total", {})
        if total:
            over = total.get("over", {})
            o_open = over.get("open", {}).get("line", "")
            o_close = over.get("close", {}).get("line", "")

            # Parse "o148.5" → 148.5
            try:
                o_open_val = float(str(o_open).replace("o", "").replace("u", ""))
                o_close_val = float(str(o_close).replace("o", "").replace("u", ""))
                result["dk_total_open"] = o_open_val
                result["dk_total_close"] = o_close_val
                result["dk_total_movement"] = round(o_close_val - o_open_val, 1)
            except:
                pass

        # Only need first entry (DraftKings)
        if result:
            break

    return result if result else None


def get_season_game_ids(season_start="2025-11-01"):
    """Get all game IDs for current season from Supabase."""
    rows = sb_get("ncaa_historical",
                   f"game_date=gte.{season_start}&select=game_id,game_date&order=game_date.asc")
    return pd.DataFrame(rows)


def get_today_game_ids():
    """Get today's game IDs from ESPN scoreboard."""
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    url = f"{ESPN_BASE}/scoreboard?dates={today}"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        data = r.json()
        events = data.get("events", [])
        return [e["id"] for e in events]
    except:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Test 20 games, no push")
    parser.add_argument("--today", action="store_true", help="Only today's games")
    parser.add_argument("--backfill", action="store_true", help="All current season games")
    args = parser.parse_args()

    print("=" * 70)
    print("  CAPTURE PICKCENTER DATA (DraftKings open/close lines)")
    print("=" * 70)

    # Load cache
    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded cache: {len(cache)} entries")
    except:
        cache = {}

    # Get game IDs
    if args.today:
        game_ids = get_today_game_ids()
        print(f"  Today's games: {len(game_ids)}")
    else:
        print("  Loading current season games...")
        df = get_season_game_ids()
        if len(df) == 0:
            print("  No current season games found.")
            return

        # First run — no dk columns yet
        todo = df

        # Also skip cached
        todo = todo[~todo["game_id"].astype(str).isin(cache.keys())]

        game_ids = todo["game_id"].astype(str).tolist()
        print(f"  Total season games: {len(df)}")
        print(f"  Already cached: {len(cache)}")
        print(f"  To check: {len(game_ids)}")

    if args.check:
        game_ids = game_ids[:20]
        print(f"  CHECK MODE: testing {len(game_ids)} games")

    # Extract
    patches = []
    found = 0
    no_data = 0
    t0 = time.time()

    for i, gid in enumerate(game_ids):
        gid = str(gid)

        if gid in cache:
            result = cache[gid]
        else:
            result = extract_pickcenter(gid)
            cache[gid] = result
            time.sleep(0.15)

        if result:
            patches.append({"game_id": int(gid), **result})
            found += 1
        else:
            no_data += 1

        if (i + 1) % 100 == 0 or i == len(game_ids) - 1:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            remaining = (len(game_ids) - i - 1) / max(rate, 0.01) / 60
            print(f"    {i+1}/{len(game_ids)} | found: {found} | no_data: {no_data} | ~{remaining:.0f}min left")

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    print(f"\n  Found pickcenter data: {found}/{len(game_ids)} games")

    # Show samples
    if patches:
        print(f"\n  Sample data:")
        for p in patches[:5]:
            mvmt = p.get("dk_spread_movement", "N/A")
            t_mvmt = p.get("dk_total_movement", "N/A")
            print(f"    {p['game_id']}: spread {p.get('dk_spread_open','?')}→{p.get('dk_spread_close','?')} "
                  f"(mvmt={mvmt}) | total {p.get('dk_total_open','?')}→{p.get('dk_total_close','?')} "
                  f"(mvmt={t_mvmt}) | ML home {p.get('dk_ml_home_open','?')}→{p.get('dk_ml_home_close','?')}")

    if args.check:
        print(f"\n  CHECK MODE — no data pushed.")
        return

    # Push
    if patches:
        print(f"\n  Pushing {len(patches)} pickcenter patches...")
        success, errors = 0, 0
        for i, patch in enumerate(patches):
            game_id = patch.pop("game_id")
            try:
                url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
                r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
                if r.ok:
                    success += 1
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"    Error on {game_id}: {r.text[:100]}")
            except:
                errors += 1

            if (i + 1) % 100 == 0 or i == len(patches) - 1:
                print(f"    {i+1}/{len(patches)} | success:{success} errors:{errors}")

        print(f"\n  PUSH COMPLETE: {success} success, {errors} errors")
    else:
        print("  No data to push.")

    print(f"\n  Done. Run daily to accumulate training data for spread movement features.")


if __name__ == "__main__":
    main()
