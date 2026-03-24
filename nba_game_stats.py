"""
nba_game_stats.py — Post-game boxscore extraction + rolling average computation

Called after game grading to:
  1. Fetch ESPN summary for completed games
  2. Extract boxscore stats (bench_pts, paint_pts, fast_break, etc.)
  3. Save raw stats to nba_game_stats table
  4. Recompute rolling 10-game averages
  5. Upsert to nba_team_rolling (30 rows, one per team)

At prediction time, nba_full_predict.py reads 2 rows from nba_team_rolling.

Usage:
  from nba_game_stats import process_completed_game, process_completed_games, backfill_game_stats
  
  # After grading a single game
  process_completed_game(game_id, game_date, home_abbr, away_abbr, home_score, away_score, market_spread)
  
  # After grading a batch
  process_completed_games(graded_games_list)
  
  # Backfill historical
  backfill_game_stats(days_back=30)
"""

import numpy as np
import requests
import time
from datetime import datetime, timedelta

ESPN_ABBR_MAP = {
    "GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS",
    "WSH":"WAS","UTAH":"UTA","UTH":"UTA","PHO":"PHX","BKLYN":"BKN","BK":"BKN",
}
NBA_ESPN_IDS = {
    "ATL":1,"BOS":2,"BKN":17,"CHA":30,"CHI":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GSW":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MEM":29,"MIA":14,
    "MIL":15,"MIN":16,"NOP":3,"NYK":18,"OKC":25,"ORL":19,"PHI":20,"PHX":21,
    "POR":22,"SAC":23,"SAS":24,"TOR":28,"UTA":26,"WAS":27,
}

def _map(a): return ESPN_ABBR_MAP.get(a, a)

def _sb():
    """Get Supabase URL and headers."""
    from config import SUPABASE_URL, SUPABASE_KEY
    return (SUPABASE_URL,
            {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
             "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"})


# ═══════════════════════════════════════════════════════════
# ESPN BOXSCORE EXTRACTION
# ═══════════════════════════════════════════════════════════

def _fetch_boxscore_stats(game_id):
    """Fetch completed game boxscore from ESPN summary.
    Returns dict keyed by team_abbr with stats per team."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    try:
        r = requests.get(url, timeout=15)
        if not r.ok:
            return None
        data = r.json()
    except Exception:
        return None

    # Verify game is completed
    status = data.get("header", {}).get("competitions", [{}])[0].get("status", {})
    if isinstance(status, dict):
        state = status.get("type", {}).get("state", "")
        if state != "post":
            return None  # game not finished

    result = {}

    # ── Boxscore team stats ──
    boxscore = data.get("boxscore", {})
    for tb in boxscore.get("teams", []):
        team_obj = tb.get("team", {})
        if not isinstance(team_obj, dict):
            continue
        abbr = _map(team_obj.get("abbreviation", ""))
        if not abbr:
            continue

        stats = {}
        for s in tb.get("statistics", []):
            if isinstance(s, dict):
                name = s.get("name", "")
                dv = s.get("displayValue", "")
                try:
                    stats[name] = float(dv)
                except (ValueError, TypeError):
                    stats[name] = dv  # compound strings like "45-85" stay as strings

        # Parse compound stat strings: "made-attempted" → (made, attempted)
        def _parse_compound(key):
            v = stats.get(key, "")
            if isinstance(v, str) and "-" in v:
                parts = v.split("-")
                try: return float(parts[0]), float(parts[1])
                except: return 0, 0
            return 0, 0

        fgm, fga = _parse_compound("fieldGoalsMade-fieldGoalsAttempted")
        tpm, tpa = _parse_compound("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        ftm, fta = _parse_compound("freeThrowsMade-freeThrowsAttempted")

        # Extract the stats we need
        # NOTE: benchPoints and secondChancePoints are NOT in ESPN boxscore stats
        team_stats = {
            "team_abbr": abbr,
            "bench_pts": 0,  # ESPN doesn't provide this in team boxscore
            "paint_pts": stats.get("pointsInPaint", 0) or 0,
            "fast_break_pts": stats.get("fastBreakPoints", 0) or 0,
            "second_chance_pts": 0,  # ESPN doesn't provide this in team boxscore
            "largest_lead": stats.get("largestLead", 0) or 0,
            "oreb": stats.get("offensiveRebounds", 0) or 0,
            # New fields from ESPN boxscore
            "turnover_pts": stats.get("turnoverPoints", 0) or 0,
            "lead_changes": stats.get("leadChanges", 0) or 0,
            "lead_pct": stats.get("leadPercentage", 0) or 0,
            "fouls": stats.get("fouls", 0) or 0,
            "ft_pct": stats.get("freeThrowPct", 0) or 0,
            "fg_pct": stats.get("fieldGoalPct", 0) or 0,
            "three_pct": stats.get("threePointFieldGoalPct", 0) or 0,
            "fgm": fgm, "fga": fga,
            "tpm": tpm, "tpa": tpa,
            "ftm": ftm, "fta": fta,
        }

        # Computed rates from parsed compound stats
        if fga > 0:
            team_stats["three_fg_rate"] = round(tpm / fga, 4)
            team_stats["ft_trip_rate"] = round(fta / fga, 4)
        else:
            team_stats["three_fg_rate"] = 0.35
            team_stats["ft_trip_rate"] = 0.25

        result[abbr] = team_stats

    # ── Quarter scoring (for Q4 diff) ──
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    if comps:
        for c in comps[0].get("competitors", []):
            team_obj = c.get("team", {})
            if not isinstance(team_obj, dict):
                continue
            abbr = _map(team_obj.get("abbreviation", ""))
            if abbr not in result:
                continue
            linescores = c.get("linescores", [])
            if len(linescores) >= 4:
                try:
                    q4 = float(linescores[3].get("displayValue", 0) or 0)
                    result[abbr]["q4_scoring"] = q4
                except (ValueError, TypeError, IndexError):
                    result[abbr]["q4_scoring"] = 0
            else:
                result[abbr]["q4_scoring"] = 0

    # ── Scoring runs (max run from plays if available) ──
    plays = data.get("plays", [])
    if plays:
        # Identify team IDs
        team_id_map = {}
        for c in comps[0].get("competitors", []) if comps else []:
            t = c.get("team", {})
            if isinstance(t, dict):
                tid = str(t.get("id", ""))
                ab = _map(t.get("abbreviation", ""))
                if tid and ab:
                    team_id_map[tid] = ab

        # Track scoring runs
        current_run_team = None
        current_run_pts = 0
        max_run = {abbr: 0 for abbr in result}

        for play in plays:
            if not play.get("scoringPlay"):
                continue
            score_val = play.get("scoreValue", 0)
            if not score_val:
                continue
            team_id = str(play.get("team", {}).get("id", "")) if isinstance(play.get("team"), dict) else ""
            team_abbr = team_id_map.get(team_id, "")
            if not team_abbr:
                continue

            if team_abbr == current_run_team:
                current_run_pts += score_val
            else:
                if current_run_team and current_run_team in max_run:
                    max_run[current_run_team] = max(max_run[current_run_team], current_run_pts)
                current_run_team = team_abbr
                current_run_pts = score_val

        # Final run
        if current_run_team and current_run_team in max_run:
            max_run[current_run_team] = max(max_run[current_run_team], current_run_pts)

        for abbr, run in max_run.items():
            if abbr in result:
                result[abbr]["max_run"] = run
    else:
        for abbr in result:
            result[abbr].setdefault("max_run", 0)

    return result


# ═══════════════════════════════════════════════════════════
# SAVE TO SUPABASE
# ═══════════════════════════════════════════════════════════

def _save_game_stats(game_id, game_date, team_abbr, stats, actual_margin, market_spread=0):
    """Save raw game stats to nba_game_stats table."""
    url, headers = _sb()
    row = {
        "game_id": str(game_id),
        "game_date": game_date,
        "team_abbr": team_abbr,
        "bench_pts": stats.get("bench_pts", 0),
        "paint_pts": stats.get("paint_pts", 0),
        "fast_break_pts": stats.get("fast_break_pts", 0),
        "second_chance_pts": stats.get("second_chance_pts", 0),
        "largest_lead": stats.get("largest_lead", 0),
        "q4_scoring": stats.get("q4_scoring", 0),
        "max_run": stats.get("max_run", 0),
        "three_fg_rate": stats.get("three_fg_rate", 0),
        "ft_trip_rate": stats.get("ft_trip_rate", 0),
        "actual_margin": actual_margin,
        "oreb": stats.get("oreb", 0),
    }
    try:
        resp = requests.post(
            f"{url}/rest/v1/nba_game_stats",
            json=row, headers=headers, timeout=10
        )
        return resp.ok
    except Exception as e:
        print(f"  [nba_game_stats] save error for {team_abbr} game {game_id}: {e}")
        return False


def _recompute_rolling(team_abbr, as_of_date=None):
    """Recompute rolling 10-game averages from nba_game_stats and upsert to nba_team_rolling."""
    url, headers = _sb()
    
    # Fetch last 10 games
    date_filter = f"&game_date=lte.{as_of_date}" if as_of_date else ""
    q = (f"{url}/rest/v1/nba_game_stats"
         f"?team_abbr=eq.{team_abbr}{date_filter}"
         f"&order=game_date.desc&limit=10"
         f"&select=game_date,bench_pts,paint_pts,fast_break_pts,second_chance_pts,"
         f"largest_lead,q4_scoring,max_run,three_fg_rate,ft_trip_rate,actual_margin,oreb")
    try:
        rows = requests.get(q, headers=headers, timeout=10).json() or []
    except Exception:
        rows = []

    if not rows:
        return False

    n = len(rows)

    def avg(key):
        vals = [r.get(key, 0) or 0 for r in rows]
        return round(float(np.mean(vals)), 3) if vals else 0

    rolling = {
        "team_abbr": team_abbr,
        "updated_date": rows[0].get("game_date", as_of_date or datetime.now().strftime("%Y-%m-%d")),
        "games_counted": n,
        "roll_bench_pts": avg("bench_pts"),
        "roll_paint_pts": avg("paint_pts"),
        "roll_fast_break_pts": avg("fast_break_pts"),
        "roll_second_chance_pts": avg("second_chance_pts"),
        "roll_largest_lead": avg("largest_lead"),
        "roll_q4_scoring": avg("q4_scoring"),
        "roll_max_run": avg("max_run"),
        "roll_three_fg_rate": avg("three_fg_rate"),
        "roll_ft_trip_rate": avg("ft_trip_rate"),
        "roll_margin_avg": avg("actual_margin"),
        "roll_oreb": avg("oreb"),
    }

    # ATS margin (need market spread — compute from actual_margin)
    # For now, just use margin directly; ATS will come from nba_predictions table
    
    try:
        resp = requests.post(
            f"{url}/rest/v1/nba_team_rolling",
            json=rolling, headers=headers, timeout=10
        )
        return resp.ok
    except Exception as e:
        print(f"  [nba_team_rolling] upsert error for {team_abbr}: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def process_completed_game(game_id, game_date, home_abbr, away_abbr, 
                           home_score, away_score, market_spread=0):
    """Process a single completed game: fetch boxscore, save stats, update rolling."""
    home_abbr = _map(home_abbr)
    away_abbr = _map(away_abbr)
    
    # Fetch boxscore
    box = _fetch_boxscore_stats(game_id)
    if not box:
        return False

    home_margin = home_score - away_score
    away_margin = away_score - home_score

    saved = 0
    for abbr, margin in [(home_abbr, home_margin), (away_abbr, away_margin)]:
        stats = box.get(abbr, {})
        if not stats:
            continue
        if _save_game_stats(game_id, game_date, abbr, stats, margin, market_spread):
            saved += 1
        if _recompute_rolling(abbr, as_of_date=game_date):
            pass

    return saved == 2


def process_completed_games(games):
    """Process a batch of completed games.
    
    Args:
        games: list of dicts with keys: game_id, game_date, home_team, away_team,
               actual_home_score, actual_away_score, market_spread_home (optional)
    """
    processed = 0
    errors = 0
    for g in games:
        try:
            ok = process_completed_game(
                game_id=g.get("game_id"),
                game_date=g.get("game_date"),
                home_abbr=g.get("home_team", ""),
                away_abbr=g.get("away_team", ""),
                home_score=g.get("actual_home_score", 0) or 0,
                away_score=g.get("actual_away_score", 0) or 0,
                market_spread=g.get("market_spread_home", 0) or 0,
            )
            if ok:
                processed += 1
            else:
                errors += 1
        except Exception as e:
            print(f"  [nba_game_stats] error processing {g.get('game_id')}: {e}")
            errors += 1
        time.sleep(0.3)  # ESPN rate limit

    print(f"  [nba_game_stats] Processed {processed}/{len(games)} games ({errors} errors)")
    return processed


def backfill_game_stats(days_back=30):
    """Backfill game stats from nba_predictions for recent completed games."""
    url, headers = _sb()
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Get completed games that don't have game_stats yet
    q = (f"{url}/rest/v1/nba_predictions"
         f"?result_entered=eq.true&game_date=gte.{cutoff}"
         f"&order=game_date.asc"
         f"&select=game_id,game_date,home_team,away_team,actual_home_score,"
         f"actual_away_score,market_spread_home")
    try:
        games = requests.get(q, headers=headers, timeout=15).json() or []
    except Exception:
        print("  [backfill] Failed to fetch completed games")
        return 0

    # Check which game_ids already have stats
    existing = set()
    try:
        eq = f"{url}/rest/v1/nba_game_stats?game_date=gte.{cutoff}&select=game_id"
        existing_rows = requests.get(eq, headers=headers, timeout=10).json() or []
        existing = {r["game_id"] for r in existing_rows}
    except Exception:
        pass

    missing = [g for g in games if g.get("game_id") and g["game_id"] not in existing]
    print(f"  [backfill] {len(games)} completed games, {len(existing)} already have stats, {len(missing)} to process")

    if not missing:
        return 0

    return process_completed_games(missing)


def get_team_rolling(team_abbr):
    """Read pre-computed rolling averages for a team. Returns dict or None."""
    url, headers = _sb()
    q = f"{url}/rest/v1/nba_team_rolling?team_abbr=eq.{_map(team_abbr)}&select=*&limit=1"
    try:
        rows = requests.get(q, headers=headers, timeout=10).json() or []
        return rows[0] if rows else None
    except Exception:
        return None


def get_rolling_diffs(home_abbr, away_abbr):
    """Get rolling stat diffs for a matchup. Returns dict of feature overrides."""
    h = get_team_rolling(home_abbr)
    a = get_team_rolling(away_abbr)
    if not h or not a:
        return {}

    diffs = {}
    for stat in ["bench_pts", "paint_pts", "fast_break_pts", "second_chance_pts",
                 "largest_lead", "q4_scoring", "max_run", "three_fg_rate", "ft_trip_rate"]:
        h_val = h.get(f"roll_{stat}", 0) or 0
        a_val = a.get(f"roll_{stat}", 0) or 0
        diffs[f"roll_{stat}_diff"] = round(h_val - a_val, 3)

    # Interaction features
    h_sc = h.get("roll_second_chance_pts", 0) or 0
    a_sc = a.get("roll_second_chance_pts", 0) or 0
    h_oreb = h.get("roll_oreb", 0) or 0
    a_oreb = a.get("roll_oreb", 0) or 0
    diffs["second_chance_x_oreb"] = round((h_sc * h_oreb) - (a_sc * a_oreb), 3)

    # Max run average (not a diff — combined signal)
    diffs["roll_max_run_avg"] = round(
        ((h.get("roll_max_run", 0) or 0) + (a.get("roll_max_run", 0) or 0)) / 2, 3)

    return diffs
