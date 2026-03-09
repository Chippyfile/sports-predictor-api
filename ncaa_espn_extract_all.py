#!/usr/bin/env python3
"""
ncaa_espn_extract_all.py — Comprehensive ESPN Data Extraction
═════════════════════════════════════════════════════════════════
Single pass over ESPN /summary to extract EVERYTHING we've been
throwing away:

  1. ODDS:     Spread, O/U, moneylines, ESPN win %, predictor
  2. PBP:      Half scores, scoring runs, droughts, lead changes,
               clutch FTs, garbage time, pace by half
  3. PLAYERS:  Star dependency, bench depth, minutes HHI,
               starters, lineup continuity
  4. VENUE:    Name, indoor/outdoor, capacity, city/state
  5. META:     Attendance, officials/referees
  6. WINPROB:  Halftime win prob, volatility, max swing

Uses separate cache from PIT backfill (ncaa_extract_cache.json).
Fully resumable with --resume.

Run AFTER PIT backfill completes:
  SUPABASE_ANON_KEY="..." python3 ncaa_espn_extract_all.py
  SUPABASE_ANON_KEY="..." python3 ncaa_espn_extract_all.py --resume
"""
import os, sys, json, time, argparse, requests, math
from collections import defaultdict

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
CACHE_FILE = "ncaa_extract_cache.json"

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=30)
        if not r.ok: break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data

def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok

def fetch_summary(event_id):
    url = f"{ESPN_SUMMARY}?event={event_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.ok: return r.json()
            if r.status_code == 429: time.sleep(2 ** (attempt + 1)); continue
            if r.status_code == 404: return None
        except requests.exceptions.RequestException: time.sleep(1)
    return None


# ═══════════════════════════════════════════════════════════════
# 1. ODDS EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_odds(summary):
    r = {}
    # Top-level odds array
    odds_list = summary.get("odds", [])
    if odds_list and isinstance(odds_list, list):
        odds = odds_list[0] if isinstance(odds_list[0], dict) else {}
        if "spread" in odds:
            try: r["espn_spread"] = float(odds["spread"])
            except: pass
        if "overUnder" in odds:
            try: r["espn_over_under"] = float(odds["overUnder"])
            except: pass
        for k in ["moneylineHome", "homeMoneyLine"]:
            if k in odds:
                try: r["espn_ml_home"] = int(odds[k]); break
                except: pass
        for k in ["moneylineAway", "awayMoneyLine"]:
            if k in odds:
                try: r["espn_ml_away"] = int(odds[k]); break
                except: pass
        provider = odds.get("provider", {})
        if isinstance(provider, dict):
            r["odds_provider"] = provider.get("name", "")
        # Team-level odds for spread records
        for side, prefix in [("homeTeamOdds", "home"), ("awayTeamOdds", "away")]:
            to = odds.get(side, {})
            if isinstance(to, dict) and "winPercentage" in to:
                try: r[f"espn_home_win_pct"] = float(to["winPercentage"]) / 100.0
                except: pass

    # Pickcenter
    for pc in summary.get("pickcenter", []):
        if not isinstance(pc, dict): continue
        if "spread" in pc and "espn_spread" not in r:
            try: r["espn_spread"] = float(pc["spread"])
            except: pass
        if "overUnder" in pc and "espn_over_under" not in r:
            try: r["espn_over_under"] = float(pc["overUnder"])
            except: pass
        hto = pc.get("homeTeamOdds", {})
        if isinstance(hto, dict) and "winPercentage" in hto and "espn_home_win_pct" not in r:
            try: r["espn_home_win_pct"] = float(hto["winPercentage"]) / 100.0
            except: pass

    # Predictor
    pred = summary.get("predictor", {})
    if isinstance(pred, dict):
        ht = pred.get("homeTeam", {})
        if isinstance(ht, dict) and "gameProjection" in ht:
            try: r["espn_predictor_home_pct"] = float(ht["gameProjection"]) / 100.0
            except: pass

    return r


# ═══════════════════════════════════════════════════════════════
# 2. VENUE / ATTENDANCE / OFFICIALS
# ═══════════════════════════════════════════════════════════════
def extract_venue_meta(summary):
    r = {}
    header = summary.get("header", {})
    comps = header.get("competitions", [{}])
    if not comps: return r
    comp = comps[0]

    venue = comp.get("venue", {})
    if venue:
        r["venue_name"] = venue.get("fullName", "")
        r["venue_indoor"] = 1 if venue.get("indoor", True) else 0
        cap = venue.get("capacity")
        if cap:
            try: r["venue_capacity"] = int(cap)
            except: pass
        addr = venue.get("address", {})
        if addr:
            r["venue_city"] = addr.get("city", "")
            r["venue_state"] = addr.get("state", "")

    att = comp.get("attendance")
    if att:
        try: r["attendance"] = int(att)
        except: pass

    officials = comp.get("officials", [])
    for i, off in enumerate(officials[:3]):
        name = off.get("displayName", "")
        if name:
            r[f"referee_{i+1}"] = name

    return r


# ═══════════════════════════════════════════════════════════════
# 3. PLAY-BY-PLAY FEATURES
# ═══════════════════════════════════════════════════════════════
def extract_pbp_features(summary):
    """Extract features from play-by-play data."""
    r = {}
    plays = summary.get("plays", [])
    if not plays or len(plays) < 10:
        return r

    # Get team IDs from header
    header = summary.get("header", {})
    comps = header.get("competitions", [{}])
    if not comps: return r
    competitors = comps[0].get("competitors", [])
    home_tid = away_tid = None
    for c in competitors:
        tid = str(c.get("id", c.get("team", {}).get("id", "")))
        if c.get("homeAway") == "home": home_tid = tid
        else: away_tid = tid
    if not home_tid or not away_tid: return r

    # ── Parse plays ──
    home_score_timeline = []  # (seconds_elapsed, home_score, away_score)
    home_score = away_score = 0
    half = 1
    last_home_score_time = 0
    last_away_score_time = 0
    home_scoring_times = []
    away_scoring_times = []
    lead_changes = 0
    ties = 0
    home_lead_time = 0
    away_lead_time = 0
    tie_time = 0
    prev_leader = "tie"  # "home", "away", "tie"
    largest_lead_home = 0
    largest_lead_away = 0

    # Clutch tracking (last 4 minutes = last 240 seconds of game)
    # NCAA game = 2400 seconds (40 min)
    GAME_LENGTH = 2400
    CLUTCH_THRESHOLD = GAME_LENGTH - 240  # last 4 min
    home_clutch_ftm = home_clutch_fta = 0
    away_clutch_ftm = away_clutch_fta = 0

    # Garbage time: lead > 20 with < 5 min left
    GARBAGE_THRESHOLD = GAME_LENGTH - 300  # last 5 min
    garbage_time_start = None
    total_garbage_seconds = 0

    # Scoring run tracking
    current_run_team = None
    current_run_pts = 0
    home_runs = []
    away_runs = []

    for play in plays:
        # Parse clock and period
        clock = play.get("clock", {})
        period = play.get("period", {})
        period_num = period.get("number", 1) if isinstance(period, dict) else 1

        # Convert clock to seconds elapsed
        display_value = clock.get("displayValue", "20:00") if isinstance(clock, dict) else "20:00"
        try:
            parts = display_value.split(":")
            mins_left = int(parts[0])
            secs_left = int(parts[1]) if len(parts) > 1 else 0
            time_remaining_in_period = mins_left * 60 + secs_left
            if period_num == 1:
                seconds_elapsed = 1200 - time_remaining_in_period
            elif period_num == 2:
                seconds_elapsed = 1200 + (1200 - time_remaining_in_period)
            else:  # OT
                seconds_elapsed = 2400 + (period_num - 2) * 300 + (300 - time_remaining_in_period)
        except:
            seconds_elapsed = 0

        # Track scoring
        scoring_play = play.get("scoringPlay", False)
        if scoring_play:
            h_score = play.get("homeScore", home_score)
            a_score = play.get("awayScore", away_score)
            pts_added_home = max(0, h_score - home_score)
            pts_added_away = max(0, a_score - away_score)

            if pts_added_home > 0:
                home_scoring_times.append(seconds_elapsed)
                if current_run_team == "home":
                    current_run_pts += pts_added_home
                else:
                    if current_run_team == "away" and current_run_pts > 0:
                        away_runs.append(current_run_pts)
                    current_run_team = "home"
                    current_run_pts = pts_added_home

            if pts_added_away > 0:
                away_scoring_times.append(seconds_elapsed)
                if current_run_team == "away":
                    current_run_pts += pts_added_away
                else:
                    if current_run_team == "home" and current_run_pts > 0:
                        home_runs.append(current_run_pts)
                    current_run_team = "away"
                    current_run_pts = pts_added_away

            home_score = h_score
            away_score = a_score

            # Lead tracking
            margin = home_score - away_score
            if margin > 0:
                if prev_leader != "home":
                    if prev_leader == "away": lead_changes += 1
                    elif prev_leader == "tie": ties += 1
                prev_leader = "home"
                largest_lead_home = max(largest_lead_home, margin)
            elif margin < 0:
                if prev_leader != "away":
                    if prev_leader == "home": lead_changes += 1
                    elif prev_leader == "tie": ties += 1
                prev_leader = "away"
                largest_lead_away = max(largest_lead_away, abs(margin))
            else:
                if prev_leader != "tie": ties += 1
                prev_leader = "tie"

            # Garbage time detection
            if abs(margin) > 20 and seconds_elapsed >= GARBAGE_THRESHOLD:
                if garbage_time_start is None:
                    garbage_time_start = seconds_elapsed
            elif garbage_time_start is not None:
                total_garbage_seconds += seconds_elapsed - garbage_time_start
                garbage_time_start = None

        # Clutch FT tracking
        play_type = play.get("type", {})
        type_text = play_type.get("text", "") if isinstance(play_type, dict) else ""
        if seconds_elapsed >= CLUTCH_THRESHOLD and "Free Throw" in type_text:
            scoring = play.get("scoringPlay", False)
            # Determine which team
            team_id = str(play.get("team", {}).get("id", "")) if isinstance(play.get("team"), dict) else ""
            if team_id == home_tid:
                home_clutch_fta += 1
                if scoring: home_clutch_ftm += 1
            elif team_id == away_tid:
                away_clutch_fta += 1
                if scoring: away_clutch_ftm += 1

    # Finalize last run
    if current_run_team == "home" and current_run_pts > 0:
        home_runs.append(current_run_pts)
    elif current_run_team == "away" and current_run_pts > 0:
        away_runs.append(current_run_pts)

    if garbage_time_start is not None:
        total_garbage_seconds += GAME_LENGTH - garbage_time_start

    # ── Compute half scores ──
    # Find scores at halftime from play data
    half_scores_found = False
    for play in plays:
        period = play.get("period", {})
        pnum = period.get("number", 1) if isinstance(period, dict) else 1
        clock = play.get("clock", {})
        dv = clock.get("displayValue", "") if isinstance(clock, dict) else ""
        if pnum == 1 and dv == "0:00":
            r["home_1h_score"] = play.get("homeScore", 0)
            r["away_1h_score"] = play.get("awayScore", 0)
            half_scores_found = True
            break
    # Also check if period 2 gives us halftime via the first play
    if not half_scores_found:
        for play in plays:
            period = play.get("period", {})
            pnum = period.get("number", 1) if isinstance(period, dict) else 1
            if pnum == 2:
                r["home_1h_score"] = play.get("homeScore", 0)
                r["away_1h_score"] = play.get("awayScore", 0)
                half_scores_found = True
                break

    if half_scores_found and "home_1h_score" in r:
        r["home_2h_score"] = home_score - r["home_1h_score"]
        r["away_2h_score"] = away_score - r["away_1h_score"]
        r["home_1h_margin"] = r["home_1h_score"] - r["away_1h_score"]
        r["home_2h_margin"] = r["home_2h_score"] - r["away_2h_score"]

    # ── Scoring runs ──
    r["home_largest_run"] = max(home_runs) if home_runs else 0
    r["away_largest_run"] = max(away_runs) if away_runs else 0
    r["home_runs_8plus"] = sum(1 for run in home_runs if run >= 8)
    r["away_runs_8plus"] = sum(1 for run in away_runs if run >= 8)

    # ── Scoring droughts ──
    def compute_droughts(scoring_times, game_length=2400):
        if len(scoring_times) < 2:
            return 0, 0
        sorted_times = sorted(scoring_times)
        gaps = []
        # Gap from start
        if sorted_times[0] > 0:
            gaps.append(sorted_times[0])
        for i in range(1, len(sorted_times)):
            gaps.append(sorted_times[i] - sorted_times[i-1])
        # Gap to end
        if sorted_times[-1] < game_length:
            gaps.append(game_length - sorted_times[-1])
        drought_count = sum(1 for g in gaps if g >= 180)  # 3+ min
        longest = max(gaps) if gaps else 0
        return drought_count, longest

    h_droughts, h_longest = compute_droughts(home_scoring_times)
    a_droughts, a_longest = compute_droughts(away_scoring_times)
    r["home_drought_count"] = h_droughts
    r["away_drought_count"] = a_droughts
    r["home_longest_drought_sec"] = h_longest
    r["away_longest_drought_sec"] = a_longest

    # ── Game flow ──
    r["lead_changes"] = lead_changes
    r["ties"] = ties
    r["largest_lead_home"] = largest_lead_home
    r["largest_lead_away"] = largest_lead_away

    # Approximate time with lead (rough — based on scoring plays)
    total_plays = len([p for p in plays if p.get("scoringPlay")])
    if total_plays > 0:
        home_lead_plays = sum(1 for p in plays if p.get("scoringPlay") and
                             p.get("homeScore", 0) > p.get("awayScore", 0))
        r["home_time_with_lead_pct"] = round(home_lead_plays / total_plays, 3)
    else:
        r["home_time_with_lead_pct"] = 0.5

    # ── Clutch ──
    r["home_clutch_ftm"] = home_clutch_ftm
    r["home_clutch_fta"] = home_clutch_fta
    r["away_clutch_ftm"] = away_clutch_ftm
    r["away_clutch_fta"] = away_clutch_fta

    # ── Garbage time ──
    r["garbage_time_seconds"] = total_garbage_seconds
    r["is_garbage_time_game"] = 1 if total_garbage_seconds >= 120 else 0

    return r


# ═══════════════════════════════════════════════════════════════
# 4. PLAYER-DERIVED FEATURES
# ═══════════════════════════════════════════════════════════════
def extract_player_features(summary):
    """Extract player-level features from boxscore.players."""
    r = {}
    bs = summary.get("boxscore", {})
    players_data = bs.get("players", [])
    if not players_data:
        return r

    for team_block in players_data:
        team_info = team_block.get("team", {})
        team_id = str(team_info.get("id", ""))

        # Determine home/away from header
        header = summary.get("header", {})
        comps = header.get("competitions", [{}])
        side = None
        if comps:
            for c in comps[0].get("competitors", []):
                cid = str(c.get("id", c.get("team", {}).get("id", "")))
                if cid == team_id:
                    side = "home" if c.get("homeAway") == "home" else "away"
                    break
        if not side:
            continue

        # Parse each player's stats
        players = []
        for stat_group in team_block.get("statistics", []):
            # stat_group has 'keys' (column names) and 'athletes' (rows)
            keys = stat_group.get("keys", [])  # e.g. ["min", "fg", "3pt", ...]
            # Sometimes keys are in labels
            if not keys:
                keys = [l.lower() for l in stat_group.get("labels", [])]

            for athlete_data in stat_group.get("athletes", []):
                athlete = athlete_data.get("athlete", {})
                player_id = str(athlete.get("id", ""))
                display_name = athlete.get("displayName", athlete.get("shortName", ""))
                starter = athlete_data.get("starter", False)
                stats_values = athlete_data.get("stats", [])

                # Parse stats into dict
                ps = {"id": player_id, "name": display_name, "starter": starter}

                for idx, key in enumerate(keys):
                    if idx < len(stats_values):
                        val = stats_values[idx]
                        key_l = key.lower().strip()
                        if key_l == "min":
                            try:
                                if ":" in str(val):
                                    parts = str(val).split(":")
                                    ps["minutes"] = int(parts[0]) + int(parts[1]) / 60
                                else:
                                    ps["minutes"] = float(val)
                            except:
                                ps["minutes"] = 0
                        elif key_l in ("pts", "points"):
                            try: ps["points"] = int(val)
                            except: ps["points"] = 0
                        elif key_l == "fg":
                            try:
                                parts = str(val).split("-")
                                ps["fgm"] = int(parts[0])
                                ps["fga"] = int(parts[1]) if len(parts) > 1 else 0
                            except: pass
                        elif key_l in ("3pt", "3-pt"):
                            try:
                                parts = str(val).split("-")
                                ps["tpm"] = int(parts[0])
                                ps["tpa"] = int(parts[1]) if len(parts) > 1 else 0
                            except: pass
                        elif key_l == "ft":
                            try:
                                parts = str(val).split("-")
                                ps["ftm"] = int(parts[0])
                                ps["fta"] = int(parts[1]) if len(parts) > 1 else 0
                            except: pass
                        elif key_l in ("reb", "rebounds"):
                            try: ps["rebounds"] = int(val)
                            except: pass
                        elif key_l in ("ast", "assists"):
                            try: ps["assists"] = int(val)
                            except: pass
                        elif key_l in ("stl", "steals"):
                            try: ps["steals"] = int(val)
                            except: pass
                        elif key_l in ("blk", "blocks"):
                            try: ps["blocks"] = int(val)
                            except: pass
                        elif key_l in ("to", "turnovers"):
                            try: ps["turnovers"] = int(val)
                            except: pass

                if ps.get("minutes", 0) > 0 or ps.get("points", 0) > 0:
                    players.append(ps)

        if not players:
            continue

        # ── Compute team-level player features ──
        total_mins = sum(p.get("minutes", 0) for p in players)
        total_pts = sum(p.get("points", 0) for p in players)

        # Sort by minutes (most → least)
        players.sort(key=lambda p: p.get("minutes", 0), reverse=True)

        # Starters (top 5 by minutes, or flagged as starter)
        starters = [p for p in players if p.get("starter", False)]
        if len(starters) < 5:
            starters = players[:5]
        bench = [p for p in players if p not in starters]

        # Star1 and top3 scoring share
        pts_sorted = sorted(players, key=lambda p: p.get("points", 0), reverse=True)
        star1_pts = pts_sorted[0].get("points", 0) if pts_sorted else 0
        top3_pts = sum(p.get("points", 0) for p in pts_sorted[:3])
        r[f"{side}_star1_pts_share"] = round(star1_pts / max(total_pts, 1), 3)
        r[f"{side}_top3_pts_share"] = round(top3_pts / max(total_pts, 1), 3)

        # Minutes HHI (Herfindahl-Hirschman Index)
        if total_mins > 0:
            hhi = sum((p.get("minutes", 0) / total_mins) ** 2 for p in players)
            r[f"{side}_minutes_hhi"] = round(hhi, 4)
        else:
            r[f"{side}_minutes_hhi"] = 0.2

        # Bench scoring
        bench_pts = sum(p.get("points", 0) for p in bench)
        r[f"{side}_bench_pts"] = bench_pts
        r[f"{side}_bench_pts_share"] = round(bench_pts / max(total_pts, 1), 3)

        # Players used (with > 0 minutes)
        r[f"{side}_players_used"] = len([p for p in players if p.get("minutes", 0) > 0])

        # Starter IDs (comma-separated for lineup tracking)
        starter_ids = ",".join(p.get("id", "") for p in starters[:5])
        r[f"{side}_starter_ids"] = starter_ids

        # Starter total minutes
        starter_mins = sum(p.get("minutes", 0) for p in starters[:5])
        r[f"{side}_starter_mins"] = round(starter_mins, 1)

    return r


# ═══════════════════════════════════════════════════════════════
# 5. WIN PROBABILITY
# ═══════════════════════════════════════════════════════════════
def extract_winprob(summary):
    r = {}
    wp = summary.get("winprobability", [])
    if not wp or not isinstance(wp, list) or len(wp) < 5:
        return r

    try:
        probs = []
        for point in wp:
            hp = point.get("homeWinPercentage", 0.5)
            try: probs.append(float(hp))
            except: pass

        if len(probs) >= 5:
            mid = len(probs) // 2
            r["halftime_home_win_prob"] = round(probs[mid], 4)
            import numpy as np
            r["wp_volatility"] = round(float(np.std(probs)), 4)
            diffs = [abs(probs[i] - probs[i-1]) for i in range(1, len(probs))]
            r["wp_max_swing"] = round(max(diffs), 4) if diffs else 0.0
    except:
        pass

    return r


# ═══════════════════════════════════════════════════════════════
# MASTER EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_all(summary):
    """Extract everything from one ESPN summary response."""
    if not summary:
        return {}
    result = {}
    result.update(extract_odds(summary))
    result.update(extract_venue_meta(summary))
    result.update(extract_pbp_features(summary))
    result.update(extract_player_features(summary))
    result.update(extract_winprob(summary))
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Comprehensive ESPN extraction")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAA COMPREHENSIVE ESPN EXTRACTION")
    print("  Odds + PBP + Players + Venue + Officials + WinProb")
    print("=" * 70)

    print("\n  Loading games...")
    all_games = sb_get("ncaa_historical",
                       "actual_home_score=not.is.null&select=game_id&order=game_date.asc")
    game_ids = [g["game_id"] for g in all_games if g.get("game_id")]
    print(f"  Total games: {len(game_ids)}")

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Cache: {len(cache)} games")

    to_fetch = [gid for gid in game_ids if gid not in cache]
    if args.limit:
        to_fetch = to_fetch[:args.limit]
    print(f"  To fetch: {len(to_fetch)}")

    # ── PHASE 1: Fetch + Extract ──
    print(f"\n  Phase 1: Fetching from ESPN...")
    errors = 0
    counts = {"odds": 0, "pbp": 0, "players": 0, "venue": 0, "refs": 0}

    for i, gid in enumerate(to_fetch):
        summary = fetch_summary(gid)
        if summary:
            extracted = extract_all(summary)
            cache[gid] = extracted
            if "espn_spread" in extracted: counts["odds"] += 1
            if "home_1h_score" in extracted: counts["pbp"] += 1
            if "home_star1_pts_share" in extracted: counts["players"] += 1
            if "venue_name" in extracted: counts["venue"] += 1
            if "referee_1" in extracted: counts["refs"] += 1
        else:
            cache[gid] = {}
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(to_fetch)} | "
                  f"odds:{counts['odds']} pbp:{counts['pbp']} players:{counts['players']} "
                  f"venue:{counts['venue']} refs:{counts['refs']} | err:{errors}")
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        time.sleep(0.2)

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    print(f"\n  Phase 1 complete:")
    for k, v in counts.items():
        print(f"    {k}: {v} ({v/max(len(to_fetch),1)*100:.0f}%)")
    print(f"    errors: {errors}")

    if args.fetch_only:
        print("  --fetch-only mode, stopping")
        return

    # ── PHASE 2: Push to Supabase ──
    print(f"\n  Phase 2: Pushing to Supabase...")
    success = 0
    skipped = 0

    # Valid columns (must exist in Supabase)
    valid_cols = {
        "espn_spread", "espn_over_under", "espn_ml_home", "espn_ml_away",
        "espn_home_win_pct", "espn_predictor_home_pct", "odds_provider",
        "venue_name", "venue_indoor", "venue_capacity", "venue_city", "venue_state",
        "attendance", "referee_1", "referee_2", "referee_3",
        "home_1h_score", "away_1h_score", "home_2h_score", "away_2h_score",
        "home_1h_margin", "home_2h_margin",
        "home_largest_run", "away_largest_run",
        "home_runs_8plus", "away_runs_8plus",
        "home_drought_count", "away_drought_count",
        "home_longest_drought_sec", "away_longest_drought_sec",
        "lead_changes", "ties", "home_time_with_lead_pct",
        "largest_lead_home", "largest_lead_away",
        "home_clutch_ftm", "home_clutch_fta", "away_clutch_ftm", "away_clutch_fta",
        "garbage_time_seconds", "is_garbage_time_game",
        "halftime_home_win_prob", "wp_volatility", "wp_max_swing",
        "home_star1_pts_share", "away_star1_pts_share",
        "home_top3_pts_share", "away_top3_pts_share",
        "home_minutes_hhi", "away_minutes_hhi",
        "home_bench_pts", "away_bench_pts",
        "home_bench_pts_share", "away_bench_pts_share",
        "home_players_used", "away_players_used",
        "home_starter_ids", "away_starter_ids",
        "home_starter_mins", "away_starter_mins",
    }

    for i, gid in enumerate(game_ids):
        extracted = cache.get(gid, {})
        if not extracted:
            skipped += 1
            continue

        patch = {k: v for k, v in extracted.items() if k in valid_cols and v is not None}
        if not patch:
            skipped += 1
            continue

        if sb_patch("ncaa_historical", "game_id", gid, patch):
            success += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(game_ids)} ({success} updated, {skipped} skipped)")
        time.sleep(0.02)

    print(f"\n{'='*70}")
    print(f"  COMPREHENSIVE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Games: {len(game_ids)} | Fetched: {len(to_fetch)} | Updated: {success}")
    print(f"")
    print(f"  DATA CAPTURED:")
    print(f"    ODDS:    spread, O/U, ML home/away, ESPN win %, predictor")
    print(f"    PBP:     half scores, scoring runs, droughts, lead changes,")
    print(f"             clutch FTs, garbage time, time with lead")
    print(f"    PLAYERS: star1 share, top3 share, minutes HHI, bench pts,")
    print(f"             players used, starter IDs, starter minutes")
    print(f"    VENUE:   name, indoor, capacity, city, state, attendance")
    print(f"    REFS:    referee_1, referee_2, referee_3")
    print(f"    WINPROB: halftime prob, volatility, max swing")
    print(f"")
    print(f"  NEXT: Retrain model with ESPN odds filling has_market gaps")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
