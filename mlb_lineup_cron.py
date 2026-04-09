"""
mlb_lineup_cron.py — Pre-cache rolling lineup wOBA for all 30 MLB teams
========================================================================
Runs nightly after grading. For each team, fetches the last 10 completed
boxscores, computes mean lineup wOBA from the batting order, and upserts
to `mlb_team_lineup_rolling` in Supabase.

At serve time, mlb_ats_v9_serve.py reads from this table instead of
hitting 20+ MLB Stats API boxscore endpoints per prediction.

Supabase table schema:
  CREATE TABLE mlb_team_lineup_rolling (
    team_abbr TEXT PRIMARY KEY,
    team_id INT,
    rolling_lineup_woba FLOAT,
    games_counted INT,
    last_updated TEXT
  );
"""

import numpy as np
import requests
import time

MLB_API = "https://statsapi.mlb.com/api/v1"

TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",  119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT", 135: "SD",  136: "SEA",
    137: "SF",  138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

N_RECENT = 10  # Must match training pipeline


def _fetch_all_batter_stats(season=2026):
    """Fetch season batting stats for wOBA computation. Same formula as v9 serve."""
    url = (f"{MLB_API}/stats?stats=season&group=hitting&season={season}"
           f"&sportIds=1&limit=1000&playerPool=ALL")
    r = requests.get(url, timeout=15)
    if not r.ok:
        print(f"  [lineup_cron] Batter stats API failed: {r.status_code}")
        return {}

    splits = r.json().get("stats", [{}])[0].get("splits", [])
    lookup = {}
    for s in splits:
        pid = s.get("player", {}).get("id")
        stat = s.get("stat", {})
        pa = int(stat.get("plateAppearances", 0) or 0)
        if pid and pa >= 10:
            ab = int(stat.get("atBats", 0) or 0)
            hits = int(stat.get("hits", 0) or 0)
            doubles = int(stat.get("doubles", 0) or 0)
            triples = int(stat.get("triples", 0) or 0)
            hr = int(stat.get("homeRuns", 0) or 0)
            bb = int(stat.get("baseOnBalls", 0) or 0)
            hbp = int(stat.get("hitByPitch", 0) or 0)
            singles = hits - doubles - triples - hr
            denom = ab + bb + hbp + 0.001
            woba = (0.69*bb + 0.72*hbp + 0.89*singles + 1.27*doubles + 1.62*triples + 2.10*hr) / denom
            lookup[pid] = round(woba, 4)

    print(f"  [lineup_cron] Loaded {len(lookup)} batter wOBAs for {season}")
    return lookup


def _compute_team_rolling_woba(team_id, batter_stats):
    """Fetch last N_RECENT boxscores for a team, return mean lineup wOBA."""
    from datetime import datetime, timedelta

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    r = requests.get(f"{MLB_API}/schedule", params={
        "sportId": 1, "teamId": team_id,
        "startDate": start, "endDate": end, "gameType": "R",
    }, timeout=10)
    if not r.ok:
        return None, 0

    game_pks = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameCode") == "F":
                game_pks.append(g["gamePk"])

    if not game_pks:
        return None, 0

    lineup_wobas = []
    for gpk in game_pks[-N_RECENT:]:
        try:
            r2 = requests.get(f"{MLB_API}/game/{gpk}/boxscore", timeout=8)
            if not r2.ok:
                continue
            data = r2.json()
            home_id = data.get("teams", {}).get("home", {}).get("team", {}).get("id")
            side = "home" if home_id == team_id else "away"
            order = data.get("teams", {}).get(side, {}).get("battingOrder", [])[:9]
            if order:
                wobas = [batter_stats.get(pid, 0.315) for pid in order]
                lineup_wobas.append(np.mean(wobas))
        except Exception:
            continue

    if not lineup_wobas:
        return None, 0

    return round(float(np.mean(lineup_wobas)), 4), len(lineup_wobas)


def refresh_all_teams():
    """Refresh rolling lineup wOBA for all 30 teams. Called from nightly cron."""
    from config import SUPABASE_URL, SUPABASE_KEY

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    t0 = time.time()
    batter_stats = _fetch_all_batter_stats()
    if not batter_stats:
        print("  [lineup_cron] No batter stats — aborting")
        return {"error": "no batter stats", "teams": 0}

    updated = 0
    skipped = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    for team_id, abbr in TEAM_IDS.items():
        rolling_woba, n_games = _compute_team_rolling_woba(team_id, batter_stats)
        if rolling_woba is None:
            skipped += 1
            continue

        row = {
            "team_abbr": abbr,
            "team_id": team_id,
            "rolling_lineup_woba": rolling_woba,
            "games_counted": n_games,
            "last_updated": now,
        }

        r = requests.patch(
            f"{SUPABASE_URL}/rest/v1/mlb_team_lineup_rolling?team_abbr=eq.{abbr}",
            headers=headers, json=row, timeout=10,
        )

        if r.ok:
            # PATCH returns 200 even if no row matched (0 rows updated).
            # Check if we need to INSERT instead.
            if r.status_code == 200 and r.text.strip() in ("", "[]"):
                # Row doesn't exist yet — insert
                r2 = requests.post(
                    f"{SUPABASE_URL}/rest/v1/mlb_team_lineup_rolling",
                    headers={**headers, "Prefer": "return=minimal"},
                    json=row, timeout=10,
                )
                if r2.ok:
                    updated += 1
                else:
                    print(f"  [lineup_cron] INSERT failed for {abbr}: {r2.status_code}")
            else:
                updated += 1
        else:
            print(f"  [lineup_cron] PATCH failed for {abbr}: {r.status_code}")

    elapsed = round(time.time() - t0, 1)
    print(f"  [lineup_cron] Done: {updated} teams updated, {skipped} skipped, {elapsed}s")
    return {"teams_updated": updated, "skipped": skipped, "duration_sec": elapsed}


if __name__ == "__main__":
    refresh_all_teams()
