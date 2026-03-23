#!/usr/bin/env python3
"""
backfill_nba_2026.py — Scrape missing 2025-26 NBA games into training data

Discovers games via ESPN scoreboard API, scrapes box scores + PBP features,
and merges into nba_summary_data.parquet and nba_training_data.parquet.

The 2025-26 NBA season runs Oct 22 2025 → Apr 13 2026 (regular season).
Current training data likely stops around early/mid January.

Pipeline:
  1. Scan scoreboard day-by-day to find game IDs
  2. Filter out games already in nba_summary_data.parquet
  3. Scrape ESPN /summary for each new game (box scores, PBP, pickcenter)
  4. Extract PBP advanced features (using scrape_nba_pbp_v2 logic)
  5. Merge into summary + training parquets
  6. Re-run enrich_nba_v3 to compute rolling features

Resumable — saves checkpoint after every 50 games.

Usage:
  python3 backfill_nba_2026.py                    # full backfill
  python3 backfill_nba_2026.py --resume            # continue from checkpoint
  python3 backfill_nba_2026.py --merge-only        # skip scrape, just merge
  python3 backfill_nba_2026.py --start 2025-12-01  # start from specific date
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────
SEASON_START = "2025-10-22"   # 2025-26 NBA regular season start
SEASON_END = "2026-03-22"     # today (adjust as needed)
SLEEP = 0.35                  # seconds between ESPN requests
CHECKPOINT_FILE = "backfill_2026_checkpoint.json"
CHECKPOINT_EVERY = 50

# ESPN team abbreviation mapping (fixes common mismatches)
ABBR_MAP = {
    "GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP",
    "UTAH": "UTA", "PHX": "PHO", "WSH": "WAS", "BKN": "BKN",
    "PHO": "PHX",  # ESPN sometimes uses PHO
}


def map_abbr(abbr):
    """Normalize ESPN team abbreviations."""
    return ABBR_MAP.get(abbr, abbr)


def fetch_scoreboard(date_str):
    """Fetch NBA scoreboard for YYYYMMDD date. Returns list of completed games."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&limit=50"
    try:
        r = requests.get(url, timeout=15)
        if not r.ok:
            return []
        data = r.json()
    except Exception:
        return []

    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue

        game_id = str(ev.get("id", ""))
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        h_abbr = map_abbr(home.get("team", {}).get("abbreviation", ""))
        a_abbr = map_abbr(away.get("team", {}).get("abbreviation", ""))
        h_score = int(home.get("score", 0) or 0)
        a_score = int(away.get("score", 0) or 0)

        if h_score == 0 and a_score == 0:
            continue

        games.append({
            "game_id": game_id,
            "game_date": datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
            "home_team": h_abbr,
            "away_team": a_abbr,
            "actual_home_score": h_score,
            "actual_away_score": a_score,
        })
    return games


def extract_summary_features(data, game_id, game_date, home_abbr, away_abbr):
    """Extract box score stats from ESPN summary response.
    Matches the nba_summary_data.parquet / nba_training_data.parquet schema."""

    row = {
        "game_id": str(game_id),
        "game_date": game_date,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "season": 2026,
    }

    # ── Box Score ──
    boxscore = data.get("boxscore", {})
    teams_box = boxscore.get("teams", [])

    for tb in teams_box:
        team_data = tb.get("team", {})
        team_abbr = map_abbr(team_data.get("abbreviation", ""))
        prefix = "home" if team_abbr == home_abbr else "away" if team_abbr == away_abbr else None
        if not prefix:
            continue

        stats_list = tb.get("statistics", [])
        stats = {}
        for s in stats_list:
            name = s.get("name", "") if isinstance(s, dict) else ""
            val = s.get("displayValue", "") if isinstance(s, dict) else ""
            if not name or "-" in str(val):  # skip compound stats like "40-95"
                continue
            try:
                stats[name] = float(val)
            except (ValueError, TypeError):
                stats[name] = 0

        # Map ESPN stat names to our column names (ESPN returns pct as integers: 42 = 42%)
        row[f"{prefix}_fgpct"] = stats.get("fieldGoalPct", 47.1) / 100
        row[f"{prefix}_threepct"] = stats.get("threePointFieldGoalPct", 36.5) / 100
        row[f"{prefix}_ftpct"] = stats.get("freeThrowPct", 78.0) / 100
        row[f"{prefix}_assists"] = stats.get("assists", 25)
        row[f"{prefix}_turnovers"] = stats.get("totalTurnovers", stats.get("turnovers", 14))
        row[f"{prefix}_steals"] = stats.get("steals", 7.5)
        row[f"{prefix}_blocks"] = stats.get("blocks", 5.0)
        row[f"{prefix}_def_reb"] = stats.get("defensiveRebounds", 34)

        total_reb = stats.get("totalRebounds", 44)
        off_reb = stats.get("offensiveRebounds", 10)
        row[f"{prefix}_orb_pct"] = round(off_reb / max(total_reb, 1), 3) if total_reb else 0.245

        # Paint/bench/fast break from boxscore
        row[f"{prefix}_paint_pts"] = stats.get("pointsInPaint", 0)
        row[f"{prefix}_fast_break_pts"] = stats.get("fastBreakPoints", 0)
        row[f"{prefix}_bench_pts"] = stats.get("benchPoints", 0)
        row[f"{prefix}_largest_lead"] = stats.get("largestLead", 0)
        row[f"{prefix}_pts_off_turnovers"] = stats.get("turnoverPoints", 0)

    # ── Scores (from header) ──
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    comp = comps[0] if comps else {}
    competitors = comp.get("competitors", [])
    for c in competitors:
        side = c.get("homeAway", "")
        prefix = "home" if side == "home" else "away" if side == "away" else None
        if not prefix:
            continue
        score = int(c.get("score", 0) or 0)
        row[f"actual_{prefix}_score"] = score

        # Linescores (quarter scores)
        for qi, ls in enumerate(c.get("linescores", [])):
            lb = f"ot{qi-3}" if qi > 3 else f"q{qi+1}"
            val = ls.get("displayValue") or ls.get("value")
            try:
                row[f"{prefix}_{lb}"] = int(val) if val else 0
            except (ValueError, TypeError):
                row[f"{prefix}_{lb}"] = 0

    # ── Pickcenter (market data) ──
    pickcenter = data.get("pickcenter", [])
    for pc in pickcenter:
        provider = pc.get("provider", {}).get("name", "").lower()
        if "consensus" in provider or not row.get("market_spread_home"):
            spread = pc.get("spread", 0) or 0
            total = pc.get("overUnder", 0) or 0
            row["market_spread_home"] = spread
            row["market_ou_total"] = total

            # ML odds
            home_odds = pc.get("homeTeamOdds", {})
            away_odds = pc.get("awayTeamOdds", {})
            row["home_ml_close"] = home_odds.get("moneyLine", 0) or 0
            row["away_ml_close"] = away_odds.get("moneyLine", 0) or 0

    # ── Win probability (pre-game baseline) ──
    wp_data = data.get("winprobability", [])
    if wp_data and len(wp_data) > 0:
        first_wp = wp_data[0]
        row["espn_pregame_wp"] = first_wp.get("homeWinPercentage", 0.5)

    # ── Standings / seeds ──
    standings = data.get("standings", {})
    for group in standings.get("groups", []):
        for entry in group.get("standings", {}).get("entries", []):
            team_obj = entry.get("team", {}); entry_team = team_obj.get("abbreviation", "") if isinstance(team_obj, dict) else ""
            entry_team = map_abbr(entry_team)
            if entry_team not in (home_abbr, away_abbr):
                # Try matching by ID
                continue
            prefix = "home" if entry_team == home_abbr else "away"
            for st in entry.get("stats", []):
                stype = st.get("type", "")
                val = st.get("value")
                if stype == "playoffSeed" and val:
                    row[f"{prefix}_seed"] = int(val)

    # ── Venue / attendance ──
    gi = data.get("gameInfo", {})
    venue = gi.get("venue", {})
    row["attendance"] = gi.get("attendance", 0) or 0
    row["venue_capacity"] = venue.get("capacity", 19000) or 19000

    return row


def extract_pbp_features(data, game_id):
    """Extract PBP-derived features (mirrors scrape_nba_pbp_v2.py logic).
    Returns dict with game_id + PBP columns."""

    result = {"game_id": str(game_id)}

    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    comp = comps[0] if comps else {}
    competitors = comp.get("competitors", [])

    home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})
    home_id = str(home_c.get("team", {}).get("id", ""))
    away_id = str(away_c.get("team", {}).get("id", ""))

    if not home_id or not away_id:
        return result

    plays = data.get("plays", [])
    if not plays:
        return result

    # Initialize counters
    stats = {home_id: defaultdict(float), away_id: defaultdict(float)}
    opp = {home_id: away_id, away_id: home_id}
    last_to_team = last_to_clock = last_to_period = None
    last_oreb_team = last_oreb_clock = last_oreb_period = None
    run_team = None
    run_pts = 0
    max_run = {home_id: 0, away_id: 0}
    prev_hs = prev_as = 0

    for play in plays:
        tid = str(play.get("team", {}).get("id", ""))
        ptype = int(play.get("type", {}).get("id", 0))
        period = play.get("period", {}).get("number", 0)
        clock_str = play.get("clock", {}).get("displayValue", "0")
        scoring = play.get("scoringPlay", False)
        score_val = play.get("scoreValue", 0)
        hs = play.get("homeScore", prev_hs)
        aws = play.get("awayScore", prev_as)

        try:
            if ":" in str(clock_str):
                p = clock_str.split(":")
                clock_sec = int(p[0]) * 60 + float(p[1])
            else:
                clock_sec = float(clock_str)
        except:
            clock_sec = 0

        if tid not in stats:
            prev_hs, prev_as = hs, aws
            continue

        # Turnovers
        if ptype in {62, 63, 72, 84, 87, 90}:
            last_to_team, last_to_clock, last_to_period = tid, clock_sec, period
            stats[tid]["turnovers_total"] += 1

        # Offensive rebounds
        if ptype == 156:
            last_oreb_team, last_oreb_clock, last_oreb_period = tid, clock_sec, period

        # Scoring
        if scoring and score_val > 0:
            # Points off turnovers
            if (last_to_team and last_to_team != tid and last_to_period == period
                    and last_to_clock is not None
                    and 0 <= (last_to_clock - clock_sec) < 15):
                stats[tid]["pts_off_turnovers"] += score_val

            # Second chance
            if (last_oreb_team == tid and last_oreb_period == period
                    and last_oreb_clock is not None
                    and 0 <= (last_oreb_clock - clock_sec) < 10):
                stats[tid]["second_chance_pts"] += score_val

            # Clutch
            pre_margin = abs(prev_hs - prev_as)
            if period >= 4 and clock_sec <= 300 and pre_margin <= 5:
                stats[tid]["clutch_pts"] += score_val

            # Runs
            if tid == run_team:
                run_pts += score_val
            else:
                if run_team and run_pts > max_run.get(run_team, 0):
                    max_run[run_team] = run_pts
                run_team, run_pts = tid, score_val

        # And-1 proxy
        if ptype == 44:  # shooting foul
            oid = opp.get(tid, "")
            if oid:
                stats[oid]["and1_proxy"] += 1

        # FT tracking
        if "Free Throw" in play.get("type", {}).get("text", ""):
            stats[tid]["ft_count"] += 1

        # Shot tracking for zone rates
        is_shot = play.get("shootingPlay", False)
        pts_att = play.get("pointsAttempted", 0)
        if is_shot and pts_att >= 2:
            stats[tid]["fga_total"] += 1
            x = play.get("coordinate", {}).get("x", -1)
            y = play.get("coordinate", {}).get("y", -1)
            if x >= 0 and y >= 0:
                dist = ((x - 25)**2 + (y - 0)**2) ** 0.5
                if dist <= 8:
                    stats[tid]["paint_fga"] += 1
                    if scoring and score_val > 0:
                        stats[tid]["paint_fgm"] += 1
                elif pts_att == 3 or dist > 22:
                    stats[tid]["three_fga"] += 1
                else:
                    stats[tid]["mid_fga"] += 1

        prev_hs, prev_as = hs, aws

    # Final run
    if run_team and run_pts > max_run.get(run_team, 0):
        max_run[run_team] = run_pts

    # Aggregate
    for prefix, tid in [("home", home_id), ("away", away_id)]:
        s = stats[tid]
        fga = max(s["fga_total"], 1)
        result[f"{prefix}_max_run"] = max_run.get(tid, 0)
        result[f"{prefix}_and_one_count"] = int(s["and1_proxy"])
        result[f"{prefix}_paint_fg_rate"] = round(s["paint_fga"] / fga, 4)
        result[f"{prefix}_midrange_fg_rate"] = round(s["mid_fga"] / fga, 4)
        result[f"{prefix}_three_fg_rate"] = round(s["three_fga"] / fga, 4)
        result[f"{prefix}_ft_trip_rate"] = round(s["ft_count"] * 0.44 / fga, 4)
        result[f"{prefix}_clutch_pts"] = int(s["clutch_pts"])

        # PBP-derived second chance / PTO (may override box score values)
        result[f"{prefix}_second_chance_pts_pbp"] = int(s["second_chance_pts"])
        result[f"{prefix}_pts_off_turnovers_pbp"] = int(s["pts_off_turnovers"])

    # ESPN pre-game win probability from PBP
    wp = data.get("winprobability", [])
    if wp:
        result["espn_pregame_wp_pbp"] = wp[0].get("homeWinPercentage", 0.5)

    return result


def discover_missing_games(start_date, end_date, existing_ids):
    """Scan ESPN scoreboards to find game IDs not in existing data."""
    print(f"\n  Discovering games from {start_date} to {end_date}...")
    all_games = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days_scanned = 0

    while current <= end:
        ds = current.strftime("%Y%m%d")
        games = fetch_scoreboard(ds)
        new_games = [g for g in games if g["game_id"] not in existing_ids]
        all_games.extend(new_games)
        days_scanned += 1

        if days_scanned % 14 == 0:
            print(f"    Scanned {days_scanned} days, found {len(all_games)} new games...", flush=True)

        time.sleep(0.15)
        current += timedelta(days=1)

    print(f"  Discovery complete: {days_scanned} days scanned, {len(all_games)} new games found")
    return all_games


def scrape_new_games(games, resume=False):
    """Scrape ESPN summary for each new game, extract features."""
    ckpt_path = CHECKPOINT_FILE

    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        done_ids = set(ckpt.get("done", []))
        summary_rows = ckpt.get("summary_rows", [])
        pbp_rows = ckpt.get("pbp_rows", [])
        print(f"  Resuming: {len(done_ids)} already done")
    else:
        done_ids = set()
        summary_rows = []
        pbp_rows = []

    remaining = [g for g in games if g["game_id"] not in done_ids]
    print(f"  Scraping {len(remaining)} games (~{len(remaining)*SLEEP/60:.0f} min)...")
    errors = 0

    for i, game in enumerate(remaining):
        gid = game["game_id"]
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"
            r = requests.get(url, timeout=15)
            if not r.ok:
                errors += 1
                continue

            data = r.json()

            # Extract summary (box score + market data)
            summary_row = extract_summary_features(
                data, gid, game["game_date"],
                game["home_team"], game["away_team"]
            )
            summary_row["actual_home_score"] = game["actual_home_score"]
            summary_row["actual_away_score"] = game["actual_away_score"]
            summary_rows.append(summary_row)

            # Extract PBP features
            pbp_row = extract_pbp_features(data, gid)
            pbp_rows.append(pbp_row)

            done_ids.add(gid)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error on {gid}: {e}")

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt = {"done": list(done_ids), "summary_rows": summary_rows, "pbp_rows": pbp_rows}
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)
            print(f"    {len(done_ids)}/{len(games)} done ({errors} errors)", flush=True)

        time.sleep(SLEEP)

    # Final save
    ckpt = {"done": list(done_ids), "summary_rows": summary_rows, "pbp_rows": pbp_rows}
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)

    print(f"\n  Scraping complete: {len(done_ids)} games, {errors} errors")
    return summary_rows, pbp_rows


def merge_into_parquets(summary_rows, pbp_rows):
    """Merge new games into nba_summary_data.parquet and nba_training_data.parquet."""

    print("\n" + "=" * 60)
    print("  MERGING INTO PARQUETS")
    print("=" * 60)

    new_summary = pd.DataFrame(summary_rows)
    new_pbp = pd.DataFrame(pbp_rows)
    print(f"  New summary rows: {len(new_summary)}")
    print(f"  New PBP rows: {len(new_pbp)}")

    if len(new_summary) == 0:
        print("  Nothing to merge!")
        return

    # ── Merge PBP columns into summary rows ──
    if len(new_pbp) > 0 and "game_id" in new_pbp.columns:
        pbp_cols = [c for c in new_pbp.columns if c != "game_id"]
        # Don't overwrite box score columns with PBP versions
        skip = [c for c in pbp_cols if c.endswith("_pbp")]
        merge_cols = [c for c in pbp_cols if c not in skip]

        # Only merge PBP cols that don't exist in summary
        existing_cols = set(new_summary.columns)
        new_pbp_cols = [c for c in merge_cols if c not in existing_cols]
        if new_pbp_cols:
            new_summary = new_summary.merge(
                new_pbp[["game_id"] + new_pbp_cols],
                on="game_id", how="left"
            )
            print(f"  Merged {len(new_pbp_cols)} PBP columns into summary")

    # ── Append to nba_summary_data.parquet ──
    summary_path = "nba_summary_data.parquet"
    if os.path.exists(summary_path):
        existing_summary = pd.read_parquet(summary_path)
        existing_ids = set(existing_summary["game_id"].astype(str))
        new_only = new_summary[~new_summary["game_id"].astype(str).isin(existing_ids)]
        if len(new_only) > 0:
            combined = pd.concat([existing_summary, new_only], ignore_index=True)
            combined.to_parquet(summary_path, index=False)
            print(f"  Summary: {len(existing_summary)} → {len(combined)} (+{len(new_only)} new)")
        else:
            print(f"  Summary: no new games to add (all {len(new_summary)} already exist)")
    else:
        new_summary.to_parquet(summary_path, index=False)
        print(f"  Created summary: {len(new_summary)} rows")

    # ── Append to nba_training_data.parquet ──
    training_path = "nba_training_data.parquet"
    if os.path.exists(training_path):
        existing_training = pd.read_parquet(training_path)

        # Deduplicate by game_date + home_team + away_team
        existing_keys = set(
            existing_training.apply(
                lambda r: f"{r.get('game_date','')}_{r.get('home_team','')}_{r.get('away_team','')}",
                axis=1
            )
        )
        new_keys = new_summary.apply(
            lambda r: f"{r.get('game_date','')}_{r.get('home_team','')}_{r.get('away_team','')}",
            axis=1
        )
        new_mask = ~new_keys.isin(existing_keys)
        new_training = new_summary[new_mask]

        if len(new_training) > 0:
            # Align columns — add missing columns as NaN
            for col in existing_training.columns:
                if col not in new_training.columns:
                    new_training[col] = np.nan
            new_training = new_training[existing_training.columns.tolist() +
                                         [c for c in new_training.columns if c not in existing_training.columns]]

            combined = pd.concat([existing_training, new_training], ignore_index=True)
            combined.to_parquet(training_path, index=False)
            print(f"  Training: {len(existing_training)} → {len(combined)} (+{len(new_training)} new)")
        else:
            print(f"  Training: no new games to add")
    else:
        new_summary.to_parquet(training_path, index=False)
        print(f"  Created training: {len(new_summary)} rows")

    # ── Also append to PBP v2 parquet ──
    pbp_path = "nba_pbp_features_v2.parquet"
    if os.path.exists(pbp_path) and len(new_pbp) > 0:
        existing_pbp = pd.read_parquet(pbp_path)
        existing_pbp_ids = set(existing_pbp["game_id"].astype(str))
        new_pbp_only = new_pbp[~new_pbp["game_id"].astype(str).isin(existing_pbp_ids)]
        if len(new_pbp_only) > 0:
            combined_pbp = pd.concat([existing_pbp, new_pbp_only], ignore_index=True)
            combined_pbp.to_parquet(pbp_path, index=False)
            print(f"  PBP v2: {len(existing_pbp)} → {len(combined_pbp)} (+{len(new_pbp_only)} new)")

    # ── Print date range coverage ──
    if os.path.exists(training_path):
        df = pd.read_parquet(training_path)
        gd = pd.to_datetime(df["game_date"])
        szn_mask = gd >= "2025-10-01"
        n_2026 = szn_mask.sum()
        latest = gd[szn_mask].max() if szn_mask.any() else "none"
        print(f"\n  2025-26 season coverage: {n_2026} games through {latest}")


def main():
    args = sys.argv[1:]

    if "--merge-only" in args:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE) as f:
                ckpt = json.load(f)
            merge_into_parquets(ckpt.get("summary_rows", []), ckpt.get("pbp_rows", []))
        else:
            print("No checkpoint file found. Run scraper first.")
        return

    # Parse start date
    start_date = SEASON_START
    if "--start" in args:
        idx = args.index("--start")
        start_date = args[idx + 1]

    end_date = SEASON_END
    if "--end" in args:
        idx = args.index("--end")
        end_date = args[idx + 1]

    resume = "--resume" in args

    print("=" * 60)
    print("  NBA 2025-26 Season Backfill")
    print(f"  Date range: {start_date} → {end_date}")
    print("=" * 60)

    # Load existing game IDs
    existing_ids = set()
    if os.path.exists("nba_summary_data.parquet"):
        existing = pd.read_parquet("nba_summary_data.parquet")
        existing_ids = set(existing["game_id"].astype(str))
        print(f"  Existing summary: {len(existing)} games")

        # Show current 2025-26 coverage
        gd = pd.to_datetime(existing["game_date"])
        szn = gd >= "2025-10-01"
        if szn.any():
            print(f"  Current 2026 season: {szn.sum()} games, latest={gd[szn].max().strftime('%Y-%m-%d')}")

    # Step 1: Discover missing games
    new_games = discover_missing_games(start_date, end_date, existing_ids)

    if not new_games:
        print("\n  No new games found! Data is up to date.")
        return

    print(f"\n  Found {len(new_games)} new games to scrape")
    print(f"  Date range: {new_games[0]['game_date']} → {new_games[-1]['game_date']}")

    # Step 2: Scrape summaries + PBP
    summary_rows, pbp_rows = scrape_new_games(new_games, resume=resume)

    # Step 3: Merge into parquets
    merge_into_parquets(summary_rows, pbp_rows)

    # Step 4: Remind about pipeline
    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("""
  The new games are now in the parquets, but rolling features
  need to be recomputed. Run this pipeline:

  1. Re-run enrichment to compute rolling stats for new games:
     python3 enrich_nba_v3.py

  2. Retrain the model with the expanded dataset:
     python3 retrain_nba.py

  3. Deploy:
     git add . && git commit -m "NBA backfill 2026" && git push
""")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total time: {time.time()-t0:.0f}s")
