"""
ncaa_daily_prefetch.py — Pre-cache ESPN data for today's games.

Hits ESPN scoreboard once to get all games, then pre-fetches stats, record,
and schedule data for each team playing today. After this runs, prediction
requests are almost entirely cache hits — only the game summary call is live.

USAGE:
    # As a standalone cron job (run daily at ~8 AM ET):
    python ncaa_daily_prefetch.py

    # As a Flask endpoint (hit via Railway cron or external scheduler):
    POST /prefetch/ncaa

    # Manually from shell:
    curl -X POST https://your-railway-app.up.railway.app/prefetch/ncaa

ESPN calls per run:
    1 scoreboard call
    + 3 calls per team (stats, record, schedule)
    × ~15-30 games/day = ~90-180 calls total
    Spread over ~30 seconds with rate limiting
"""

import time
import requests
import concurrent.futures
from datetime import datetime, timezone

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"

# Rate limit: slight delay between teams to avoid ESPN throttling
DELAY_BETWEEN_TEAMS = 0.3  # seconds


def fetch_todays_games(date_str=None):
    """
    Get all NCAA games for a given date from ESPN scoreboard.

    Returns: list of {
        "game_id": str,
        "home_team_id": str,
        "away_team_id": str,
        "home_team_name": str,
        "away_team_name": str,
        "game_time": str (ISO),
        "status": str ("pre" | "in" | "post"),
    }
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    else:
        # Accept YYYY-MM-DD and convert to YYYYMMDD
        date_str = date_str.replace("-", "")

    try:
        r = requests.get(f"{ESPN_SCOREBOARD}?dates={date_str}&limit=100", timeout=15)
        if not r.ok:
            print(f"  [prefetch] Scoreboard failed: HTTP {r.status_code}")
            return []

        data = r.json()
        events = data.get("events", [])
        games = []

        for ev in events:
            game_id = ev.get("id", "")
            comp = ev.get("competitions", [{}])[0]
            status_type = comp.get("status", {}).get("type", {}).get("state", "pre")

            home_team = None
            away_team = None
            for team in comp.get("competitors", []):
                team_info = {
                    "id": str(team.get("id", "")),
                    "name": team.get("team", {}).get("displayName", ""),
                }
                if team.get("homeAway") == "home":
                    home_team = team_info
                else:
                    away_team = team_info

            if home_team and away_team:
                games.append({
                    "game_id": game_id,
                    "home_team_id": home_team["id"],
                    "away_team_id": away_team["id"],
                    "home_team_name": home_team["name"],
                    "away_team_name": away_team["name"],
                    "game_time": ev.get("date", ""),
                    "status": status_type,
                })

        print(f"  [prefetch] Found {len(games)} games for {date_str}")
        return games

    except Exception as e:
        print(f"  [prefetch] Scoreboard error: {e}")
        return []


def prefetch_team_data(team_id, game_date_str, opp_rank=200):
    """
    Pre-fetch and cache all ESPN data for one team.

    Calls (importing from ncaa_full_predict to use its cache):
        1. _fetch_espn_team_stats(team_id)
        2. _fetch_espn_team_record(team_id)
        3. _fetch_team_schedule_data(team_id, game_date_str, opp_rank)
    """
    from ncaa_full_predict import (
        _fetch_espn_team_stats,
        _fetch_espn_team_record,
        _fetch_team_schedule_data,
    )

    results = {"stats": False, "record": False, "schedule": False}

    try:
        stats = _fetch_espn_team_stats(team_id)
        results["stats"] = stats is not None
    except Exception as e:
        print(f"  [prefetch] Stats failed for {team_id}: {e}")

    try:
        record = _fetch_espn_team_record(team_id)
        results["record"] = record is not None
    except Exception as e:
        print(f"  [prefetch] Record failed for {team_id}: {e}")

    try:
        sched = _fetch_team_schedule_data(team_id, game_date_str, opp_rank)
        results["schedule"] = sched is not None
    except Exception as e:
        print(f"  [prefetch] Schedule failed for {team_id}: {e}")

    return results


def run_prefetch(date_str=None):
    """
    Main prefetch routine: get today's games, pre-cache all team data.

    Returns summary dict for logging/endpoint response.
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start = time.time()
    games = fetch_todays_games(date_str)

    if not games:
        return {
            "status": "no_games",
            "date": date_str,
            "games": 0,
            "teams_cached": 0,
            "elapsed_sec": round(time.time() - start, 1),
        }

    # Deduplicate team IDs (a team only plays once per day)
    teams_to_fetch = {}
    for g in games:
        if g["home_team_id"] not in teams_to_fetch:
            teams_to_fetch[g["home_team_id"]] = {
                "name": g["home_team_name"],
                "opp_rank": 200,  # Could be enriched from scoreboard data
            }
        if g["away_team_id"] not in teams_to_fetch:
            teams_to_fetch[g["away_team_id"]] = {
                "name": g["away_team_name"],
                "opp_rank": 200,
            }

    print(f"  [prefetch] Pre-fetching data for {len(teams_to_fetch)} teams...")

    # Fetch in small batches to avoid overwhelming ESPN
    success_count = 0
    fail_count = 0
    team_ids = list(teams_to_fetch.keys())

    # Process 4 teams at a time with delays between batches
    BATCH_SIZE = 4
    for i in range(0, len(team_ids), BATCH_SIZE):
        batch = team_ids[i:i + BATCH_SIZE]

        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(
                    prefetch_team_data, tid, date_str, teams_to_fetch[tid]["opp_rank"]
                ): tid
                for tid in batch
            }
            for future in concurrent.futures.as_completed(futures):
                tid = futures[future]
                try:
                    result = future.result()
                    if all(result.values()):
                        success_count += 1
                    else:
                        fail_count += 1
                        failed = [k for k, v in result.items() if not v]
                        print(f"  [prefetch] Partial fail for {tid} ({teams_to_fetch[tid]['name']}): {failed}")
                except Exception as e:
                    fail_count += 1
                    print(f"  [prefetch] Error for {tid}: {e}")

        # Rate limit between batches
        if i + BATCH_SIZE < len(team_ids):
            time.sleep(DELAY_BETWEEN_TEAMS * BATCH_SIZE)

    elapsed = round(time.time() - start, 1)
    summary = {
        "status": "ok",
        "date": date_str,
        "games": len(games),
        "teams_total": len(teams_to_fetch),
        "teams_cached": success_count,
        "teams_failed": fail_count,
        "elapsed_sec": elapsed,
        "games_list": [
            {
                "game_id": g["game_id"],
                "matchup": f"{g['away_team_name']} @ {g['home_team_name']}",
                "status": g["status"],
            }
            for g in games
        ],
    }

    print(f"  [prefetch] Done: {success_count}/{len(teams_to_fetch)} teams cached in {elapsed}s")
    return summary


# ═══════════════════════════════════════════════════════════════
# FLASK ENDPOINT — Add to your app.py
# ═══════════════════════════════════════════════════════════════
"""
from ncaa_daily_prefetch import run_prefetch

@app.route("/prefetch/ncaa", methods=["POST"])
def prefetch_ncaa():
    date_str = request.json.get("date") if request.json else None
    result = run_prefetch(date_str)
    return jsonify(result)
"""


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_prefetch(date)
    print(json.dumps(result, indent=2))
