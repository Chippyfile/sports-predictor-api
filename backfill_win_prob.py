#!/usr/bin/env python3
"""
Backfill win_pct_home for all graded NCAA predictions using /predict/ncaa/full.
Replaces sigma-based probabilities with real ML isotonic-calibrated probabilities.

Usage:
    python3 backfill_win_prob.py
"""

import requests
import time
import json
from datetime import datetime

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4YWFxdHF2bHdqdnl1ZWR5YXVvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTgwNjM1NSwiZXhwIjoyMDg3MzgyMzU1fQ.m9D8hP71LjKnYT3MPxHPtYS4aSD6TX5lMA7286T_L_U"
ML_API = "https://sports-predictor-api-production.up.railway.app"

HEADERS_GET = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}
HEADERS_PATCH = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

PAGE_SIZE = 500
DELAY = 0.3  # seconds between ML API calls


def fetch_all_graded_games():
    """Fetch all graded games with team IDs and current win_pct_home."""
    games = []
    offset = 0
    print("Fetching graded games from Supabase...")
    while True:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/ncaa_predictions"
            f"?result_entered=eq.true"
            f"&select=id,game_date,home_team_id,away_team_id,neutral_site,game_id,win_pct_home,spread_home"
            f"&order=id.asc&limit={PAGE_SIZE}&offset={offset}",
            headers=HEADERS_GET, timeout=30
        )
        batch = r.json()
        if not batch:
            break
        games.extend(batch)
        print(f"  Fetched {len(games)} games...", end="\r")
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    print(f"\nTotal graded games: {len(games)}")
    return games


def get_ml_prediction(home_team_id, away_team_id, neutral_site, game_date, game_id):
    """Call /predict/ncaa/full and return ml_win_prob_home and ml_margin."""
    try:
        r = requests.post(
            f"{ML_API}/predict/ncaa/full",
            json={
                "home_team_id": str(home_team_id),
                "away_team_id": str(away_team_id),
                "neutral_site": bool(neutral_site),
                "game_date": game_date,
                "game_id": str(game_id) if game_id else None,
            },
            timeout=12
        )
        if not r.ok:
            return None, None
        data = r.json()
        if data.get("error"):
            return None, None
        return data.get("ml_win_prob_home"), data.get("ml_margin")
    except Exception as e:
        return None, None


def patch_game(game_id, win_pct_home, spread_home):
    """Patch win_pct_home and spread_home for a game."""
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/ncaa_predictions?id=eq.{game_id}",
        headers=HEADERS_PATCH,
        json={"win_pct_home": round(win_pct_home, 4), "spread_home": round(spread_home, 1)},
        timeout=15
    )
    return r.status_code in (200, 204)


def main():
    games = fetch_all_graded_games()
    total = len(games)

    updated = 0
    failed = 0
    skipped = 0

    log = []
    start = datetime.now()

    print(f"\nStarting backfill of {total} games...")
    print(f"Estimated time: {total * DELAY / 60:.1f} minutes\n")

    for i, game in enumerate(games):
        game_id = game["id"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        neutral = game.get("neutral_site", False)
        game_date = game.get("game_date")
        espn_game_id = game.get("game_id")

        if not home_id or not away_id:
            skipped += 1
            continue

        win_prob, margin = get_ml_prediction(home_id, away_id, neutral, game_date, espn_game_id)

        if win_prob is None:
            failed += 1
            log.append({"id": game_id, "status": "failed"})
        else:
            success = patch_game(game_id, win_prob, margin or game["spread_home"])
            if success:
                updated += 1
                log.append({"id": game_id, "status": "ok", "win_prob": win_prob, "margin": margin})
            else:
                failed += 1
                log.append({"id": game_id, "status": "patch_failed"})

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = (datetime.now() - start).seconds
            remaining = (total - i - 1) * DELAY
            print(f"  [{i+1}/{total}] Updated: {updated} | Failed: {failed} | Skipped: {skipped} | ETA: {remaining/60:.1f}min")

        time.sleep(DELAY)

    # Save log
    with open("backfill_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Backfill complete!")
    print(f"  Updated:  {updated}")
    print(f"  Failed:   {failed}")
    print(f"  Skipped:  {skipped}")
    print(f"  Total:    {total}")
    print(f"  Log saved to backfill_log.json")


if __name__ == "__main__":
    main()
