"""
mlb_abs_collector.py — Nightly ABS Challenge Data Collector
============================================================
Scrapes ABS challenge data from MLB Stats API v1.1 feed/live endpoint.
Stores per-team and per-ump rolling stats to Supabase.

Run nightly after MLB grading cron (12:30 AM+ ET).

Tables:
  mlb_abs_team:  team, games, challenges_won, challenges_lost, win_rate, cpg
  mlb_abs_ump:   ump_name, ump_id, games, challenges, overturns, ovr_rate, cpg, rpg

Usage:
  python mlb_abs_collector.py                   # collect today's games
  python mlb_abs_collector.py --backfill        # backfill all 2026 games
  python mlb_abs_collector.py --date 2026-04-08 # specific date
"""

import requests
import json
import argparse
import traceback
from datetime import datetime, timedelta, timezone
from collections import defaultdict

API = "https://statsapi.mlb.com/api"

TEAM_ID_TO_ABBR = {
    108:"LAA",109:"ARI",110:"BAL",111:"BOS",112:"CHC",113:"CIN",114:"CLE",
    115:"COL",116:"DET",117:"HOU",118:"KC",119:"LAD",120:"WSH",121:"NYM",
    133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",138:"STL",139:"TB",
    140:"TEX",141:"TOR",142:"MIN",143:"PHI",144:"ATL",145:"CWS",146:"MIA",
    147:"NYY",158:"MIL",
}


def fetch_completed_games(date_str):
    """Get all completed game_pks for a date."""
    r = requests.get(f"{API}/v1/schedule", params={
        "sportId": 1, "date": date_str, "gameType": "R",
    }, timeout=15)
    if not r.ok:
        return []
    pks = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameCode") == "F":
                pks.append(g["gamePk"])
    return pks


def fetch_game_abs(game_pk):
    """Fetch ABS + ump + score data from a single game."""
    try:
        feed = requests.get(f"{API}/v1.1/game/{game_pk}/feed/live", timeout=10).json()
        gd = feed.get("gameData", {})
        ld = feed.get("liveData", {})

        # Teams
        h_id = gd.get("teams", {}).get("home", {}).get("id")
        a_id = gd.get("teams", {}).get("away", {}).get("id")
        h_abbr = TEAM_ID_TO_ABBR.get(h_id, "?")
        a_abbr = TEAM_ID_TO_ABBR.get(a_id, "?")

        # HP ump
        hp_ump = None
        hp_ump_id = None
        for o in ld.get("boxscore", {}).get("officials", []):
            if o.get("officialType") == "Home Plate":
                hp_ump = o["official"]["fullName"]
                hp_ump_id = o["official"]["id"]
                break

        # Score
        ls = ld.get("linescore", {})
        h_runs = int(ls.get("teams", {}).get("home", {}).get("runs", 0))
        a_runs = int(ls.get("teams", {}).get("away", {}).get("runs", 0))
        total = h_runs + a_runs
        margin = h_runs - a_runs

        # ABS
        abs_d = gd.get("absChallenges", {})
        h_won = int(abs_d.get("home", {}).get("usedSuccessful", 0))
        h_lost = int(abs_d.get("home", {}).get("usedFailed", 0))
        a_won = int(abs_d.get("away", {}).get("usedSuccessful", 0))
        a_lost = int(abs_d.get("away", {}).get("usedFailed", 0))

        # Game date
        game_date = gd.get("datetime", {}).get("officialDate", "")

        return {
            "game_pk": game_pk,
            "game_date": game_date,
            "home": h_abbr, "away": a_abbr,
            "h_runs": h_runs, "a_runs": a_runs,
            "total": total, "margin": margin,
            "hp_ump": hp_ump, "hp_ump_id": hp_ump_id,
            "h_abs_won": h_won, "h_abs_lost": h_lost,
            "a_abs_won": a_won, "a_abs_lost": a_lost,
            "total_challenges": h_won + h_lost + a_won + a_lost,
            "total_overturns": h_won + a_won,
        }
    except Exception as e:
        print(f"  [abs] Error fetching {game_pk}: {e}")
        return None


def collect_date_range(start_date, end_date):
    """Collect all ABS data for a date range."""
    games = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        pks = fetch_completed_games(date_str)
        for pk in pks:
            result = fetch_game_abs(pk)
            if result:
                games.append(result)
        if pks:
            print(f"  [abs] {date_str}: {len(pks)} games")
        current += timedelta(days=1)

    return games


def build_team_stats(games):
    """Aggregate per-team ABS stats."""
    teams = defaultdict(lambda: {"games": 0, "won": 0, "lost": 0})
    for g in games:
        teams[g["home"]]["games"] += 1
        teams[g["home"]]["won"] += g["h_abs_won"]
        teams[g["home"]]["lost"] += g["h_abs_lost"]
        teams[g["away"]]["games"] += 1
        teams[g["away"]]["won"] += g["a_abs_won"]
        teams[g["away"]]["lost"] += g["a_abs_lost"]

    rows = []
    for team, t in teams.items():
        total_chal = t["won"] + t["lost"]
        rows.append({
            "team": team,
            "season": 2026,
            "games": t["games"],
            "challenges_won": t["won"],
            "challenges_lost": t["lost"],
            "challenges_total": total_chal,
            "win_rate": round(t["won"] / max(total_chal, 1), 4),
            "cpg": round(total_chal / max(t["games"], 1), 2),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    return rows


def build_ump_stats(games):
    """Aggregate per-ump ABS stats."""
    umps = defaultdict(lambda: {
        "ump_id": None, "games": 0, "challenges": 0,
        "overturns": 0, "total_runs": 0,
    })
    for g in games:
        if not g["hp_ump"]:
            continue
        u = umps[g["hp_ump"]]
        u["ump_id"] = g["hp_ump_id"]
        u["games"] += 1
        u["challenges"] += g["total_challenges"]
        u["overturns"] += g["total_overturns"]
        u["total_runs"] += g["total"]

    rows = []
    for name, u in umps.items():
        rows.append({
            "ump_name": name,
            "ump_id": u["ump_id"],
            "season": 2026,
            "games": u["games"],
            "challenges": u["challenges"],
            "overturns": u["overturns"],
            "ovr_rate": round(u["overturns"] / max(u["challenges"], 1), 4),
            "cpg": round(u["challenges"] / max(u["games"], 1), 2),
            "rpg": round(u["total_runs"] / max(u["games"], 1), 2),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    return rows


def upload_to_supabase(team_rows, ump_rows):
    """Upload aggregated stats to Supabase via REST API."""
    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

        # Team ABS
        if team_rows:
            for row in team_rows:
                r = requests.post(f"{SUPABASE_URL}/rest/v1/mlb_abs_team",
                    headers=headers, json=row, timeout=10)
                if not r.ok and r.status_code != 409:
                    print(f"  [abs] Team {row['team']}: {r.status_code}")
            print(f"  [abs] Uploaded {len(team_rows)} team ABS profiles")

        # Ump ABS
        if ump_rows:
            for row in ump_rows:
                r = requests.post(f"{SUPABASE_URL}/rest/v1/mlb_abs_ump",
                    headers=headers, json=row, timeout=10)
                if not r.ok and r.status_code != 409:
                    print(f"  [abs] Ump {row['ump_name']}: {r.status_code}")
            print(f"  [abs] Uploaded {len(ump_rows)} ump ABS profiles")

    except Exception as e:
        print(f"  [abs] Supabase upload error: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Backfill all 2026 games")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--no-upload", action="store_true", help="Skip Supabase upload")
    args = parser.parse_args()

    print("=" * 70)
    print("  MLB ABS CHALLENGE DATA COLLECTOR")
    print("=" * 70)

    if args.backfill:
        start = "2026-03-25"
        end = datetime.now().strftime("%Y-%m-%d")
        print(f"  Backfill: {start} → {end}")
    elif args.date:
        start = end = args.date
        print(f"  Date: {args.date}")
    else:
        # Default: yesterday (for post-grading cron)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = end = yesterday
        print(f"  Yesterday: {yesterday}")

    games = collect_date_range(start, end)
    print(f"\n  Collected {len(games)} games")

    if not games:
        print("  No games found.")
        return

    # For backfill, rebuild full aggregates; for daily, we still rebuild full
    # because upsert replaces the row
    if args.backfill:
        all_games = games
    else:
        # For daily updates, we need ALL games to rebuild aggregates
        # Fetch full season
        print("  Fetching full season for aggregate rebuild...")
        all_games = collect_date_range("2026-03-25", datetime.now().strftime("%Y-%m-%d"))
        print(f"  Total season: {len(all_games)} games")

    team_rows = build_team_stats(all_games)
    ump_rows = build_ump_stats(all_games)

    # Print summary
    print(f"\n  Team ABS stats: {len(team_rows)} teams")
    for t in sorted(team_rows, key=lambda x: -x["win_rate"])[:5]:
        print(f"    {t['team']:>5}: {t['win_rate']:.0%} win rate, {t['cpg']:.1f} C/G ({t['games']}G)")
    print(f"  ...")
    for t in sorted(team_rows, key=lambda x: x["win_rate"])[:3]:
        print(f"    {t['team']:>5}: {t['win_rate']:.0%} win rate, {t['cpg']:.1f} C/G ({t['games']}G)")

    print(f"\n  Ump ABS stats: {len(ump_rows)} umps")
    for u in sorted(ump_rows, key=lambda x: -x["challenges"])[:5]:
        print(f"    {u['ump_name']:>25}: {u['ovr_rate']:.0%} OVR, {u['cpg']:.1f} C/G, {u['rpg']:.1f} R/G ({u['games']}G)")

    if not args.no_upload:
        upload_to_supabase(team_rows, ump_rows)
    else:
        print("\n  Skipping upload (--no-upload)")

    print("\n  Done.")


if __name__ == "__main__":
    main()
