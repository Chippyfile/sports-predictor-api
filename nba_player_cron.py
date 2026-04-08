"""
nba_player_cron.py — Daily hoopR player impact update
======================================================
Downloads latest hoopR parquet for current season, computes rolling
BPM/VORP/impact for all players, uploads to Supabase nba_player_impact.

Called by main.py route: /cron/nba-player-update
Schedule: ~1AM EST via GitHub Actions (after all games complete)
"""

import numpy as np
import pandas as pd
import requests as _req
import io, os, time
from datetime import datetime, timezone
from collections import defaultdict

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY or ''}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

WINDOW = 10
MIN_MINUTES = 5.0
CURRENT_SEASON = 2026
HOOPR_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/player_box/parquet/player_box_{season}.parquet"

ABBR_MAP = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "WSH": "WAS",
            "UTAH": "UTA", "UTH": "UTA", "PHO": "PHX", "BKN": "BKN", "BK": "BKN"}


def compute_bpm_proxy(mins, pts, reb, ast, stl, blk, tov):
    if mins < MIN_MINUTES:
        return 0.0
    s = 36.0 / mins
    return (0.064 * pts * s + 0.116 * reb * s + 0.192 * ast * s
            + 0.225 * stl * s + 0.128 * blk * s - 0.137 * tov * s - 1.62)


def run_player_update():
    """Download hoopR data, compute rolling metrics, upload to Supabase."""
    results = {"status": "running"}
    t0 = time.time()

    # ── 1. Download current season from hoopR (~3MB, ~2 seconds) ──
    url = HOOPR_URL.format(season=CURRENT_SEASON)
    try:
        r = _req.get(url, timeout=30)
        if not r.ok or len(r.content) < 1000:
            results["error"] = f"hoopR download failed: HTTP {r.status_code}"
            return results
        box = pd.read_parquet(io.BytesIO(r.content))
        print(f"  [player_cron] Downloaded {CURRENT_SEASON}: {len(box)} rows ({len(r.content)//1024}KB)")
    except Exception as e:
        results["error"] = f"hoopR download error: {e}"
        return results

    # ── 2. Clean ──
    box["game_date"] = pd.to_datetime(box["game_date"])
    for col in ["minutes", "plus_minus", "points", "rebounds", "assists",
                 "steals", "blocks", "turnovers"]:
        box[col] = pd.to_numeric(box[col], errors="coerce").fillna(0)
    box["starter"] = box["starter"].fillna(False).astype(bool)
    box["did_not_play"] = box["did_not_play"].fillna(False).astype(bool)
    box["team_abbreviation"] = box["team_abbreviation"].map(lambda x: ABBR_MAP.get(x, x))

    played = box[(~box["did_not_play"]) & (box["minutes"] >= MIN_MINUTES)].copy()
    played = played.sort_values(["athlete_id", "game_date"])
    print(f"  [player_cron] {len(played)} player-games with minutes >= {MIN_MINUTES}")

    # ── 3. Compute rolling metrics per player ──
    records = []
    for athlete_id, pdf in played.groupby("athlete_id"):
        recent = pdf.tail(WINDOW)
        if len(recent) < 3:
            continue

        season_games = pdf[pdf["season"] == CURRENT_SEASON]
        name = recent.iloc[-1]["athlete_display_name"]
        team = recent.iloc[-1]["team_abbreviation"]
        position = recent.iloc[-1].get("athlete_position_abbreviation", "")

        avg_min = recent["minutes"].mean()
        avg_pts = recent["points"].mean()
        avg_reb = recent["rebounds"].mean()
        avg_ast = recent["assists"].mean()
        avg_stl = recent["steals"].mean()
        avg_blk = recent["blocks"].mean()
        avg_tov = recent["turnovers"].mean()
        avg_pm = recent["plus_minus"].mean()

        avg_bpm = np.mean([compute_bpm_proxy(
            r["minutes"], r["points"], r["rebounds"], r["assists"],
            r["steals"], r["blocks"], r["turnovers"]
        ) for _, r in recent.iterrows()])

        n_season = len(season_games) if len(season_games) > 0 else len(recent)
        total_min = season_games["minutes"].sum() if len(season_games) > 0 else recent["minutes"].sum()
        min_share = avg_min / 48.0
        vorp = (avg_bpm + 2.0) * min_share * n_season / 82.0
        margin_impact = avg_bpm * min_share
        bpm_weighted = avg_bpm * min(avg_min / 36.0, 1.0)

        last_5 = pdf.tail(5)
        starter_rate = last_5["starter"].mean() if len(last_5) > 0 else 0

        # True shooting
        fga = recent["field_goals_attempted"].sum() if "field_goals_attempted" in recent.columns else 0
        fta = recent["free_throws_attempted"].sum() if "free_throws_attempted" in recent.columns else 0
        total_pts = recent["points"].sum()
        tsa = fga + 0.44 * fta
        ts_pct = total_pts / (2 * tsa) if tsa > 0 else 0.5

        records.append({
            "player_name": str(name),
            "team": str(team),
            "position": str(position),
            "season": CURRENT_SEASON,
            "games": int(n_season),
            "minutes": round(float(total_min), 1),
            "mpg": round(float(avg_min), 1),
            "minutes_share": round(float(min_share), 4),
            "bpm": round(float(avg_bpm), 2),
            "obpm": 0,
            "dbpm": 0,
            "vorp": round(float(vorp), 2),
            "per": 0,
            "usage_pct": 0,
            "ws": 0,
            "ts_pct": round(float(ts_pct), 3),
            "impact_score": round(float(bpm_weighted + (vorp / max(n_season, 1)) * 10), 2),
            "margin_impact": round(float(margin_impact), 3),
            "bpm_weighted": round(float(bpm_weighted), 3),
            "source": "hoopr_rolling",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"  [player_cron] Computed metrics for {len(records)} players")

    # ── 4. Upload to Supabase ──
    # Clear existing rolling data for current season
    r = _req.delete(
        f"{SUPABASE_URL}/rest/v1/nba_player_impact?season=eq.{CURRENT_SEASON}",
        headers=SB_HEADERS, timeout=15
    )
    print(f"  [player_cron] Cleared existing data: HTTP {r.status_code}")

    # Batch insert
    batch_size = 100
    success = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        r = _req.post(
            f"{SUPABASE_URL}/rest/v1/nba_player_impact",
            headers=SB_HEADERS, json=batch, timeout=30
        )
        if r.ok:
            success += len(batch)
        else:
            print(f"  [player_cron] Batch {i // batch_size + 1} error: {r.status_code} {r.text[:200]}")
        time.sleep(0.2)

    duration = round(time.time() - t0, 1)
    results.update({
        "status": "complete",
        "players_computed": len(records),
        "players_uploaded": success,
        "box_scores_processed": len(played),
        "seasons_downloaded": [CURRENT_SEASON],
        "duration_sec": duration,
    })
    print(f"  [player_cron] ✅ {success}/{len(records)} players uploaded in {duration}s")
    return results
