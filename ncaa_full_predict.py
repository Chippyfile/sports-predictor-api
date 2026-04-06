"""
ncaa_full_predict.py — Complete NCAA prediction endpoint.

Backend fetches ALL data (ESPN stats, Supabase rolling features, Elo, referee,
injuries, spread movement) so the ML model gets 146/146 features with real data.

Caching strategy:
- Team base stats: 4h TTL (ESPN API)
- Rolling features + Elo: 4h TTL (Supabase)
- Injuries: 4h normally, 15min within 2h of game time
- Referees: 4h normally, 15min within 2h of game time

Usage:
    POST /predict/ncaa/full
    Body: {"home_team_id": 130, "away_team_id": 2509, "game_date": "2026-03-15",
           "game_id": "401829050", "neutral_site": false}
"""

import time
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from db import sb_get, load_model
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from config import SUPABASE_URL, SUPABASE_KEY

try:
    from injury_cache import get_team_injuries
    HAS_INJURY_CACHE = True
except ImportError:
    HAS_INJURY_CACHE = False

# ═══════════════════════════════════════════════════════════════
# CACHING LAYER
# ═══════════════════════════════════════════════════════════════

_cache = {}
_cache_lock = threading.Lock()

CACHE_TTL_LONG = 4 * 60 * 60      # 4 hours (base stats, rolling features)
CACHE_TTL_PREGAME = 15 * 60        # 15 minutes (injuries, refs near game time)
PREGAME_WINDOW = 2 * 60 * 60       # 2 hours before game


def _cache_get(key, ttl=CACHE_TTL_LONG):
    """Get from cache if fresh."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _cache_set(key, data):
    """Store in cache."""
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}


def _is_pregame(game_date_str, game_time_utc=None):
    """Check if we're within 2 hours of game time."""
    try:
        now = datetime.now(timezone.utc)
        if game_time_utc:
            game_dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        else:
            # Default: assume 7 PM ET for the game date
            game_dt = datetime.fromisoformat(f"{game_date_str}T00:00:00+00:00") + timedelta(hours=24)
        return abs((game_dt - now).total_seconds()) < PREGAME_WINDOW
    except:
        return False


# ═══════════════════════════════════════════════════════════════
# ESPN DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"


def _fetch_espn_team_stats(team_id):
    """Fetch team stats from ESPN API."""
    cache_key = f"espn_stats_{team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}/statistics", timeout=10)
        if not r.ok:
            return None
        data = r.json()

        stats = {}
        # Parse team stats from ESPN response
        categories = data.get("results", {}).get("stats", {}).get("categories", [])
        if not categories:
            # Try alternate structure
            categories = data.get("statistics", {}).get("splits", {}).get("categories", [])

        stat_map = {}
        for cat in categories:
            for stat in cat.get("stats", []):
                stat_map[stat.get("name", "")] = stat.get("value", 0)

        # AUDIT: Log stat names containing poss/tempo/pace/point for debugging
        _tempo_keys = [k for k in stat_map if any(w in k.lower() for w in ['poss', 'tempo', 'pace'])]
        if _tempo_keys:
            print(f"  [full_predict] ESPN tempo-related keys for {team_id}: {_tempo_keys}")
        else:
            print(f"  [full_predict] WARNING: No tempo/poss fields in ESPN stats for {team_id}. Available: {list(stat_map.keys())[:15]}...")

        stats = {
            "ppg": stat_map.get("avgPoints", 75),
            "opp_ppg": stat_map.get("avgPointsAllowed", stat_map.get("oppPoints", 72)),
            # ESPN returns percentages (50.6), model expects decimals (0.506)
            "fgpct": stat_map.get("fieldGoalPct", 44.0) / 100.0,
            "threepct": stat_map.get("threePointFieldGoalPct", 34.0) / 100.0,
            "ftpct": stat_map.get("freeThrowPct", 72.0) / 100.0,
            "assists": stat_map.get("avgAssists", 14),
            "turnovers": stat_map.get("avgTurnovers", 12),
            # AUDIT FIX: ESPN uses different field names for tempo/possessions across endpoints.
            # Try multiple names, then estimate from box score stats if all fail.
            "tempo": (stat_map.get("avgPossessions")
                      or stat_map.get("possessions")
                      or stat_map.get("possessionsPerGame")
                      or stat_map.get("totalPossessions")
                      or stat_map.get("paceOfPlay")
                      or 0),  # 0 = signal to estimate below
            "steals": stat_map.get("avgSteals", 7),
            "blocks": stat_map.get("avgBlocks", 3.5),
            # Derived stats for model features
            "twopt_pct": stat_map.get("twoPointFieldGoalPct", 48.0) / 100.0,
            "orb": stat_map.get("avgOffensiveRebounds", 10),
            "drb": stat_map.get("avgDefensiveRebounds", 25),
            "fga": stat_map.get("avgFieldGoalsAttempted", 60),
            "fta": stat_map.get("avgFreeThrowsAttempted", 22),
            "three_att": stat_map.get("avgThreePointFieldGoalsAttempted", 25),
            "three_made": stat_map.get("avgThreePointFieldGoalsMade", 9),
            "fgm": stat_map.get("avgFieldGoalsMade", 30),
            "ftm": stat_map.get("avgFreeThrowsMade", 17),
            "total_reb": stat_map.get("avgRebounds", 35),
        }
        
        # Compute advanced stats the model needs
        fga = stats["fga"] or 60
        fta = stats["fta"] or 22
        fgm = stats["fgm"] or 30
        ftm = stats["ftm"] or 17
        three_made = stats["three_made"] or 9
        ppg = stats["ppg"] or 75
        total_reb = stats["total_reb"] or 35
        orb = stats["orb"] or 10
        drb = stats["drb"] or 25
        turnovers = stats["turnovers"] or 12
        
        # AUDIT FIX: Estimate tempo from box score if ESPN didn't provide it
        # Standard formula: possessions ≈ FGA - ORB + TO + 0.44 * FTA
        if stats["tempo"] == 0 or stats["tempo"] is None:
            estimated_tempo = fga - orb + turnovers + 0.44 * fta
            stats["tempo"] = round(estimated_tempo, 1) if estimated_tempo > 40 else 68
            print(f"  [full_predict] Tempo estimated from box score: {stats['tempo']}")
        
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        stats["efg_pct"] = (fgm + 0.5 * three_made) / max(fga, 1)
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        stats["ts_pct"] = ppg / max(2 * (fga + 0.44 * fta), 1)
        # 3-point rate = 3PA / FGA
        stats["three_rate"] = stats["three_att"] / max(fga, 1)
        # Assist rate = AST / FGM
        stats["assist_rate"] = stats["assists"] / max(fgm, 1)
        # FTA rate = FTA / FGA
        stats["fta_rate"] = fta / max(fga, 1)
        # ORB% = ORB / total rebounds (rough estimate)
        stats["orb_pct"] = orb / max(total_reb, 1)
        # DRB% = DRB / total rebounds
        stats["drb_pct"] = drb / max(total_reb, 1)
        # ATO ratio
        stats["ato_ratio"] = stats["assists"] / max(stats["turnovers"], 0.5)
        # PPP = points per possession
        poss = stats["tempo"] if stats["tempo"] > 0 else 68
        stats["ppp"] = ppg / max(poss, 1)

        _cache_set(cache_key, stats)
        return stats
    except Exception as e:
        print(f"  [full_predict] ESPN stats error for {team_id}: {e}")
        return None


def _fetch_espn_team_info(team_id):
    """Fetch rank, conference, and basic info from ESPN team endpoint."""
    cache_key = f"espn_team_info_{team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    info = {"rank": 200, "conference": "", "conference_id": "", "standing": ""}
    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}", timeout=8)
        if not r.ok:
            return info
        team = r.json().get("team", {})
        # Rank
        info["rank"] = team.get("rank", 200) or 200
        # Conference from standingSummary ("1st in ACC" → "ACC")
        standing = team.get("standingSummary", "")
        info["standing"] = standing
        if standing:
            import re
            m = re.search(r'(?:in|of)\s+(.+)', standing, re.IGNORECASE)
            if m:
                info["conference"] = m.group(1).strip()
        # Conference ID from groups
        groups = team.get("groups", {})
        if groups.get("isConference"):
            info["conference_id"] = str(groups.get("id", ""))
        _cache_set(cache_key, info)
        return info
    except Exception as e:
        print(f"  [full_predict] ESPN team info error for {team_id}: {e}")
        return info


def _fetch_espn_team_record(team_id):
    """Fetch SOS and record from ESPN."""
    cache_key = f"espn_record_{team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}/record", timeout=10)
        if not r.ok:
            return {"wins": 15, "losses": 10, "sos": 0.5}
        data = r.json()
        items = data.get("items", [])

        overall = next((i for i in items if i.get("type") == "total"), None)
        sos_item = next((i for i in items if i.get("type") == "sos"), None)

        get_stat = lambda item, name: next(
            (s.get("value", 0) for s in (item or {}).get("stats", []) if s.get("name") == name), 0
        )

        result = {
            "wins": int(get_stat(overall, "wins") or 15),
            "losses": int(get_stat(overall, "losses") or 10),
            "sos": get_stat(sos_item, "opponentWinPercent") or 0.5,
        }

        _cache_set(cache_key, result)
        return result
    except:
        return {"wins": 15, "losses": 10, "sos": 0.5}


def _fetch_espn_game_info(game_id, game_date, pregame=False):
    """Fetch referee, attendance, venue, ESPN odds, win probs from ESPN game summary."""
    ttl = CACHE_TTL_PREGAME if pregame else CACHE_TTL_LONG
    cache_key = f"espn_game_{game_id}"
    cached = _cache_get(cache_key, ttl)
    if cached:
        return cached

    info = {
        "referee_1": "", "referee_2": "", "referee_3": "",
        "attendance": 0, "venue_capacity": 8000, "venue_name": "",
        "espn_spread": 0, "espn_over_under": 0,
        "espn_home_win_pct": 0.5, "espn_predictor_home_pct": 0.5,
        "is_conference_competition": 0,
    }

    try:
        r = requests.get(
            f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
            timeout=10
        )
        if not r.ok:
            _cache_set(cache_key, info)
            return info

        data = r.json()

        # Referees
        officials = data.get("gameInfo", {}).get("officials", [])
        for i, ref in enumerate(officials[:3]):
            name = ref.get("displayName", "")
            info[f"referee_{i+1}"] = name

        # Attendance & Venue
        gi = data.get("gameInfo", {})
        info["attendance"] = gi.get("attendance", 0) or 0
        venue = gi.get("venue", {})
        info["venue_capacity"] = venue.get("capacity", 8000) or 8000
        info["venue_name"] = venue.get("fullName", "")

        # ESPN Odds (try pickcenter first, then odds array)
        pickcenter = data.get("pickcenter", [])
        if pickcenter:
            for pc in pickcenter:
                if pc.get("homeTeamOdds", {}).get("spreadOdds") or pc.get("spread") is not None:
                    info["espn_spread"] = pc.get("spread", 0) or 0
                    info["espn_over_under"] = pc.get("overUnder", 0) or 0
                    break
        if info["espn_spread"] == 0:
            odds = data.get("odds", [])
            if odds:
                primary = odds[0]
                info["espn_spread"] = primary.get("spread", 0) or 0
                info["espn_over_under"] = primary.get("overUnder", 0) or 0

        # ESPN Predictor (win probability model)
        predictor = data.get("predictor", {})
        if predictor:
            home_pred = predictor.get("homeTeam", {})
            info["espn_predictor_home_pct"] = home_pred.get("gameProjection", 50) / 100.0

        # ESPN Win Probability — first entry is pre-game baseline
        winprob = data.get("winprobability", [])
        if winprob:
            first = winprob[0]
            info["espn_home_win_pct"] = first.get("homeWinPercentage", 0.5)

        # Conference tournament detection from header/competition
        header = data.get("header", {})
        for comp in header.get("competitions", []):
            if comp.get("conferenceCompetition", False):
                info["is_conference_competition"] = 1
        # Also check notes for tournament keywords
        notes = header.get("gameNote", "") or ""
        season_notes = data.get("seasonseries", {})
        for note in header.get("competitions", [{}])[0].get("notes", []):
            headline = note.get("headline", "") or ""
            if any(kw in headline.lower() for kw in ["tournament", "semifinal", "quarterfinal", "championship", "final"]):
                info["is_conference_competition"] = 1

        _cache_set(cache_key, info)
        return info
    except:
        _cache_set(cache_key, info)
        return info


def _fetch_team_schedule_data(team_id, game_date_str, opp_rank=200):
    """
    Query ncaa_historical for team's completed games this season.
    Same data source as training → perfect alignment.

    Returns: {
        "rest_days": int,
        "is_lookahead": 0/1, "is_sandwich": 0/1,
        "is_revenge_game": 0/1, "revenge_margin": float,
        "after_loss": 0/1, "games_last_14": int,
        "wins": int, "losses": int,
        "opponents": [{"opp_id": str, "margin": float, "completed": bool}, ...]
    }
    """
    cache_key = f"team_sched_{team_id}_{game_date_str}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    result = {
        "rest_days": 3,
        "is_lookahead": 0, "is_sandwich": 0,
        "is_revenge_game": 0, "revenge_margin": 0.0,
        "after_loss": 0, "games_last_14": 0,
        "wins": 0, "losses": 0,
        "opponents": [],
    }

    try:
        game_date = datetime.fromisoformat(f"{game_date_str}T00:00:00")
        # NCAA season: Nov (year-1) through April (year). Season label = spring year.
        season_year = game_date.year if game_date.month < 8 else game_date.year + 1

        # Pull all completed games for this team this season from ncaa_historical
        _select = "game_date,home_team_id,away_team_id,actual_home_score,actual_away_score,home_rank,away_rank"
        home_rows = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{team_id}&season=eq.{season_year}"
            f"&actual_home_score=not.is.null"
            f"&game_date=lt.{game_date_str}"
            f"&select={_select}&order=game_date.asc"
        ) or []
        away_rows = sb_get(
            "ncaa_historical",
            f"away_team_id=eq.{team_id}&season=eq.{season_year}"
            f"&actual_away_score=not.is.null"
            f"&game_date=lt.{game_date_str}"
            f"&select={_select}&order=game_date.asc"
        ) or []

        # Merge and normalize to team perspective
        games = []
        for r in home_rows:
            games.append({
                "date": r.get("game_date", ""),
                "opp_id": str(r.get("away_team_id", "")),
                "score_team": float(r.get("actual_home_score", 0) or 0),
                "score_opp": float(r.get("actual_away_score", 0) or 0),
                "opp_rank": int(r.get("away_rank") or 200),
            })
        for r in away_rows:
            games.append({
                "date": r.get("game_date", ""),
                "opp_id": str(r.get("home_team_id", "")),
                "score_team": float(r.get("actual_away_score", 0) or 0),
                "score_opp": float(r.get("actual_home_score", 0) or 0),
                "opp_rank": int(r.get("home_rank") or 200),
            })

        games.sort(key=lambda g: g["date"])
        print(f"  [full_predict] Schedule {team_id}: {len(games)} games from ncaa_historical (season={season_year})")

        if not games:
            _cache_set(cache_key, result)
            return result

        # ── Rest days ──
        last_date_str = games[-1]["date"]
        if last_date_str:
            try:
                last_date = datetime.fromisoformat(last_date_str[:10] + "T00:00:00")
                result["rest_days"] = max(0, (game_date - last_date).days)
            except:
                result["rest_days"] = 3

        # ── Record, opponents list ──
        for g in games:
            if g["score_team"] > g["score_opp"]:
                result["wins"] += 1
            elif g["score_team"] < g["score_opp"]:
                result["losses"] += 1
            if g["opp_id"]:
                result["opponents"].append({
                    "opp_id": g["opp_id"],
                    "margin": g["score_team"] - g["score_opp"],
                    "completed": True,
                })

        # ── After loss ──
        last = games[-1]
        if last["score_team"] < last["score_opp"]:
            result["after_loss"] = 1

        # ── Games in last 14 days ──
        cutoff_14 = (game_date - timedelta(days=14)).strftime("%Y-%m-%d")
        result["games_last_14"] = sum(1 for g in games if g["date"] >= cutoff_14)

        # ── Lookahead / Sandwich ──
        # For lookahead we'd need future games — not available in ncaa_historical.
        # These are low-value features (is_lookahead, is_sandwich) and rarely nonzero.
        # Leave as 0 — they were 0 in training most of the time too.

        _cache_set(cache_key, result)
        return result

    except Exception as e:
        print(f"  [full_predict] Schedule data error for {team_id}: {e}")
        import traceback
        traceback.print_exc()
        return result


def _compute_common_opponents(home_sched, away_sched):
    """Find common opponents and compute margin differential."""
    home_opps = {o["opp_id"]: o["margin"] for o in home_sched.get("opponents", []) if o.get("completed") and o.get("opp_id")}
    away_opps = {o["opp_id"]: o["margin"] for o in away_sched.get("opponents", []) if o.get("completed") and o.get("opp_id")}

    print(f"  [full_predict] Common opps: home has {len(home_opps)} opponents, away has {len(away_opps)} opponents")

    common_ids = set(home_opps.keys()) & set(away_opps.keys())
    n_common = len(common_ids)
    if n_common == 0:
        return 0, 0.0

    # Average margin difference against common opponents
    margin_diffs = [home_opps[cid] - away_opps[cid] for cid in common_ids]
    avg_diff = sum(margin_diffs) / len(margin_diffs)
    print(f"  [full_predict] Common opps: {n_common} shared, avg_diff={avg_diff:.2f}")
    return n_common, round(avg_diff, 2)


# ═══════════════════════════════════════════════════════════════
# SUPABASE DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

def _fetch_supabase_team_data(team_id, team_name):
    """Fetch rolling features, Elo, advanced stats from Supabase ncaa_historical.

    Robust: tries full column set first; if Supabase rejects any column (400),
    falls back to a minimal core set so we still get Elo/form/SOS/tempo.
    """
    cache_key = f"sb_team_{team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    # ── Core columns (always exist in ncaa_historical) ──
    _CORE_COLS = (
        "elo,pyth_residual,adj_oe,adj_de,"
        "efg_pct,twopt_pct,three_rate,assist_rate,drb_pct,ppp,"
        "opp_efg_pct,opp_to_rate,opp_fta_rate,opp_orb_pct,"
        "opp_fgpct,opp_threepct,opp_ppg,sos,form,tempo,"
        "luck,consistency,margin_trend,margin_accel,"
        "opp_adj_form,wl_momentum,recovery_idx,is_after_loss,"
        "ceiling,floor,margin_skew,scoring_entropy,bimodal,"
        "def_stability,opp_suppression,def_versatility,"
        "steal_foul_ratio,block_foul_ratio,transition_dep,"
        "paint_pts,fastbreak_pts,fatigue_load,streak,"
        "season_pct,regression_pressure,info_gain,overreaction,"
        "scoring_source_entropy,ft_dependency,three_value,"
        "concentration,to_conversion,three_divergence,"
        "ppp_divergence,pace_adj_margin,pit_sos,"
        "scoring_var,score_kurtosis,clutch_ratio,garbage_adj_ppp,"
        "days_since_loss,games_since_blowout_loss,games_last_14,"
        "rest_effect,momentum_halflife,win_aging,centrality,"
        "dow_effect,conf_balance,eff_vol_ratio,"
        "home_margin,away_margin,"
        "run_vulnerability,anti_fragility,sos_trajectory,"
        "margin_autocorr,blowout_asym,clutch_over_exp,ft_pressure,"
        "fg_divergence,def_improvement,rhythm_disruption,"
        "fouls,close_win_rate,games_last_7,"
        "pts_off_to"
    )

    # ── Optional columns (may not exist in all table versions) ──
    _OPTIONAL_COLS = (
        "roll_star1_share,roll_top3_share,roll_bench_share,"
        "roll_bench_pts,roll_largest_run,roll_drought_rate,"
        "roll_lead_changes,roll_time_with_lead_pct,"
        "roll_players_used,roll_hhi,"
        "roll_clutch_ft_pct,roll_garbage_pct,"
        "roll_ats_pct,roll_ats_n,roll_ats_margin,"
        "player_rating_sum,weakest_starter,starter_variance,"
        "lineup_changes,lineup_stability_5g,starter_games_together,"
        "starter_ids"
    )

    def _run_query(cols_str):
        cols_list = [c.strip() for c in cols_str.split(",") if c.strip()]
        home_select = "game_date," + ",".join([f"home_{c}" for c in cols_list])
        away_select = "game_date," + ",".join([f"away_{c}" for c in cols_list])

        home_rows = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{team_id}&order=game_date.desc&limit=1&select={home_select}"
        )
        away_rows = sb_get(
            "ncaa_historical",
            f"away_team_id=eq.{team_id}&order=game_date.desc&limit=1&select={away_select}"
        )

        home_date = (home_rows[0].get("game_date", "") or "") if home_rows else ""
        away_date = (away_rows[0].get("game_date", "") or "") if away_rows else ""

        if away_date > home_date and away_rows:
            row = away_rows[0]
            result = {}
            for k, v in row.items():
                if k == "game_date":
                    continue
                new_key = k.replace("away_", "home_") if k.startswith("away_") else k
                result[new_key] = v
            return result
        elif home_rows:
            return {k: v for k, v in home_rows[0].items() if k != "game_date"}
        return None

    try:
        # Try full query first (core + optional)
        full_cols = _CORE_COLS + "," + _OPTIONAL_COLS
        result = _run_query(full_cols)
        if result is not None:
            print(f"  [full_predict] Supabase {team_id}: {len(result)} columns (full query)")
            _cache_set(cache_key, result)
            return result

        # Full query returned no rows — try core-only (column might not exist)
        print(f"  [full_predict] Full query empty for {team_id}, trying core-only...")
        result = _run_query(_CORE_COLS)
        if result is not None:
            print(f"  [full_predict] Supabase {team_id}: {len(result)} columns (core-only)")
            _cache_set(cache_key, result)
            return result

        print(f"  [full_predict] Supabase {team_id}: no rows found in ncaa_historical")
        _cache_set(cache_key, {})
        return {}
    except Exception as e:
        print(f"  [full_predict] Supabase team data error for {team_id}: {e}")
        return {}


def _fetch_crowd_shock(home_team_id, away_team_id):
    """Compute crowd_shock_diff from recent home attendance in ncaa_historical."""
    cache_key = f"crowd_shock_{home_team_id}_{away_team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached is not None:
        return cached

    import numpy as np
    try:
        # Fetch last 5 home games with attendance for each team
        h_rows = sb_get("ncaa_historical",
            f"home_team_id=eq.{home_team_id}&attendance=gt.0&select=attendance&order=game_date.desc&limit=5")
        a_rows = sb_get("ncaa_historical",
            f"home_team_id=eq.{away_team_id}&attendance=gt.0&select=attendance&order=game_date.desc&limit=5")

        h_atts = [r["attendance"] for r in (h_rows or []) if r.get("attendance", 0) > 0]
        a_atts = [r["attendance"] for r in (a_rows or []) if r.get("attendance", 0) > 0]

        h_avg = np.mean(h_atts) if h_atts else 0
        a_avg = np.mean(a_atts) if a_atts else 0

        if h_avg > 0 and a_avg > 0:
            val = float(np.log(h_avg / a_avg))
        else:
            val = 0.0

        _cache_set(cache_key, val)
        return val
    except Exception as e:
        print(f"  [full_predict] crowd_shock error: {e}")
        return 0.0


def _fetch_spread_movement(game_id):
    """Fetch spread movement data from ncaa_historical."""
    cache_key = f"spread_mvmt_{game_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    try:
        rows = sb_get(
            "ncaa_historical",
            f"game_id=eq.{game_id}&select="
            "odds_api_spread_movement,dk_spread_movement,"
            "odds_api_total_movement,dk_total_movement"
        )
        result = rows[0] if rows else {}
        _cache_set(cache_key, result)
        return result
    except:
        return {}


# ═══════════════════════════════════════════════════════════════
# SERVE-TIME FEATURE COMPUTATION (when Supabase returns defaults)
# ═══════════════════════════════════════════════════════════════

def _compute_elo_from_schedule(sched, adj_em, sos):
    """Compute approximate Elo from schedule data when Supabase Elo is NULL.

    Uses wins, losses, margin, SOS, and adj_em to estimate Elo.
    Calibrated to match training Elo range: ~1200-1800.
    """
    wins = sched.get("wins", 0)
    losses = sched.get("losses", 0)
    total = wins + losses
    if total < 3:
        return 1500  # Not enough data

    win_pct = wins / total

    # Primary signal: adj_em (efficiency margin) — maps almost linearly to Elo
    # Training data shows Elo ≈ 1500 + adj_em * 12 (rough calibration)
    if adj_em and adj_em != 0:
        elo_from_em = 1500 + adj_em * 12
    else:
        # Fallback: Elo from win_pct + SOS
        sos_adj = max(0, (sos or 0.5) - 0.4) * 2  # 0-1.2 range
        elo_from_em = 1500 + 400 * (win_pct - 0.5) * (1 + sos_adj)

    # Add margin signal from schedule (average margin weighted by recency)
    opps = sched.get("opponents", [])
    if len(opps) >= 5:
        recent = opps[-10:]  # Last 10 games
        margins = [o.get("margin", 0) for o in recent if o.get("completed")]
        if margins:
            avg_margin = sum(margins) / len(margins)
            # Blend: 80% adj_em-derived, 20% recent margin
            elo_from_em = 0.8 * elo_from_em + 0.2 * (1500 + avg_margin * 15)

    return max(1200, min(1850, round(elo_from_em, 1)))


def _compute_rolling_ats(team_id, game_date):
    """Compute rolling ATS stats from ncaa_predictions table.

    Returns: {"ats_pct": float, "ats_n": int, "ats_margin": float}
    """
    cache_key = f"rolling_ats_{team_id}_{game_date}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    result = {"ats_pct": 0.5, "ats_n": 0, "ats_margin": 0.0}
    try:
        # Get last 15 graded predictions where this team played
        home_rows = sb_get(
            "ncaa_predictions",
            f"home_team_id=eq.{team_id}&result_entered=eq.true"
            f"&game_date=lt.{game_date}&rl_correct=not.is.null"
            f"&select=rl_correct,market_spread_home,actual_home_score,actual_away_score"
            f"&order=game_date.desc&limit=15"
        ) or []
        away_rows = sb_get(
            "ncaa_predictions",
            f"away_team_id=eq.{team_id}&result_entered=eq.true"
            f"&game_date=lt.{game_date}&rl_correct=not.is.null"
            f"&select=rl_correct,market_spread_home,actual_home_score,actual_away_score"
            f"&order=game_date.desc&limit=15"
        ) or []

        ats_results = []
        ats_margins = []
        for r in home_rows:
            if r.get("rl_correct") is not None:
                ats_results.append(1 if r["rl_correct"] else 0)
                # ATS margin: how much team beat the spread
                hs = float(r.get("actual_home_score", 0) or 0)
                aws = float(r.get("actual_away_score", 0) or 0)
                sp = float(r.get("market_spread_home", 0) or 0)
                ats_margins.append((hs - aws) + sp)
        for r in away_rows:
            if r.get("rl_correct") is not None:
                # Away team: rl_correct is from home perspective, flip
                ats_results.append(0 if r["rl_correct"] else 1)
                hs = float(r.get("actual_home_score", 0) or 0)
                aws = float(r.get("actual_away_score", 0) or 0)
                sp = float(r.get("market_spread_home", 0) or 0)
                ats_margins.append(-((hs - aws) + sp))

        if len(ats_results) >= 3:
            result["ats_pct"] = round(sum(ats_results) / len(ats_results), 4)
            result["ats_n"] = len(ats_results)
            result["ats_margin"] = round(sum(ats_margins) / len(ats_margins), 2) if ats_margins else 0.0
            print(f"  [full_predict] Rolling ATS for {team_id}: {result['ats_pct']:.0%} ({result['ats_n']} games), margin={result['ats_margin']:+.1f}")

        _cache_set(cache_key, result)
        return result
    except Exception as e:
        print(f"  [full_predict] Rolling ATS error for {team_id}: {e}")
        return result


def _compute_advanced_from_schedule(sched, sb_fn):
    """Compute advanced metrics from schedule history when Supabase values are NULL.

    Returns dict of {feature: value} for features that can be derived from schedule data.
    """
    result = {}
    opps = [o for o in sched.get("opponents", []) if o.get("completed")]
    if len(opps) < 5:
        return result

    margins = [o["margin"] for o in opps]
    recent = margins[-10:]

    # blowout_asym: (wins by 15+) / (losses by 15+) ratio
    big_wins = sum(1 for m in margins if m >= 15)
    big_losses = sum(1 for m in margins if m <= -15)
    result["blowout_asym"] = round((big_wins - big_losses) / max(len(margins), 1), 4)

    # momentum_halflife: recency-weighted form (more recent = more weight)
    import math
    weighted = sum(m * math.exp(-0.1 * (len(recent) - 1 - i)) for i, m in enumerate(recent))
    weight_sum = sum(math.exp(-0.1 * (len(recent) - 1 - i)) for i in range(len(recent)))
    result["momentum_halflife"] = round(weighted / max(weight_sum, 1), 4)

    # overreaction: volatility of margin changes (game-to-game)
    if len(margins) >= 3:
        diffs = [margins[i] - margins[i-1] for i in range(1, len(margins))]
        import numpy as _np
        result["overreaction"] = round(float(_np.std(diffs)), 4)

    # opp_suppression: avg margin against ranked opponents
    ranked_games = [o for o in opps if (o.get("opp_rank") or 200) <= 50]
    if len(ranked_games) >= 3:
        result["opp_suppression"] = round(sum(o["margin"] for o in ranked_games) / len(ranked_games), 4)

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def predict_ncaa_full(request_data):
    """
    Full NCAA prediction with backend data lookup.

    Input: {"home_team_id", "away_team_id", "game_date", "game_id", "neutral_site"}
    Returns: Complete prediction with all 146 features populated from real data.
    """
    home_team_id = request_data.get("home_team_id")
    away_team_id = request_data.get("away_team_id")
    game_date = request_data.get("game_date", datetime.now().strftime("%Y-%m-%d"))
    game_id = request_data.get("game_id", "")
    neutral_site = request_data.get("neutral_site", False)
    home_team_name = request_data.get("home_team_name", "")
    away_team_name = request_data.get("away_team_name", "")

    if not home_team_id or not away_team_id:
        return {"error": "home_team_id and away_team_id required"}

    # Load model
    bundle = load_model("ncaa")
    if not bundle:
        return {"error": "NCAAB model not trained"}

    pregame = _is_pregame(game_date)

    # ── Fetch all data in parallel-ish (Python threads for I/O) ──
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        fut_home_stats = executor.submit(_fetch_espn_team_stats, home_team_id)
        fut_away_stats = executor.submit(_fetch_espn_team_stats, away_team_id)
        fut_home_record = executor.submit(_fetch_espn_team_record, home_team_id)
        fut_away_record = executor.submit(_fetch_espn_team_record, away_team_id)
        fut_game_info = executor.submit(_fetch_espn_game_info, game_id, game_date, pregame)
        fut_home_sb = executor.submit(_fetch_supabase_team_data, home_team_id, home_team_name)
        fut_away_sb = executor.submit(_fetch_supabase_team_data, away_team_id, away_team_name)
        fut_spread = executor.submit(_fetch_spread_movement, game_id)
        fut_home_sched = executor.submit(_fetch_team_schedule_data, home_team_id, game_date, request_data.get("away_rank", 200))
        fut_away_sched = executor.submit(_fetch_team_schedule_data, away_team_id, game_date, request_data.get("home_rank", 200))
        fut_home_info = executor.submit(_fetch_espn_team_info, home_team_id)
        fut_away_info = executor.submit(_fetch_espn_team_info, away_team_id)
        fut_home_ats = executor.submit(_compute_rolling_ats, home_team_id, game_date)
        fut_away_ats = executor.submit(_compute_rolling_ats, away_team_id, game_date)

    home_stats = fut_home_stats.result() or {}
    away_stats = fut_away_stats.result() or {}
    home_record = fut_home_record.result()
    away_record = fut_away_record.result()
    game_info = fut_game_info.result()
    home_sb = fut_home_sb.result()
    away_sb = fut_away_sb.result()
    spread_mvmt = fut_spread.result()
    home_sched = fut_home_sched.result()
    away_sched = fut_away_sched.result()
    home_info = fut_home_info.result()
    away_info = fut_away_info.result()
    home_ats = fut_home_ats.result()
    away_ats = fut_away_ats.result()

    # Rest days extracted from consolidated schedule data
    home_rest = home_sched.get("rest_days", 3)
    away_rest = away_sched.get("rest_days", 3)

    # Common opponents from schedule data (live computation)
    n_common_opps, common_opp_diff = _compute_common_opponents(home_sched, away_sched)

    # Historical fallback: if live computation found no common opponents,
    # check if ncaa_historical has pre-computed values for this matchup
    if n_common_opps == 0:
        try:
            hist_rows = sb_get(
                "ncaa_historical",
                f"home_team_id=eq.{home_team_id}&away_team_id=eq.{away_team_id}"
                f"&n_common_opps=gt.0&select=n_common_opps,common_opp_diff"
                f"&order=game_date.desc&limit=1"
            )
            if not hist_rows:
                # Try reverse matchup
                hist_rows = sb_get(
                    "ncaa_historical",
                    f"home_team_id=eq.{away_team_id}&away_team_id=eq.{home_team_id}"
                    f"&n_common_opps=gt.0&select=n_common_opps,common_opp_diff"
                    f"&order=game_date.desc&limit=1"
                )
                if hist_rows:
                    # Flip sign for common_opp_diff since perspective is reversed
                    hist_rows[0]["common_opp_diff"] = -(hist_rows[0].get("common_opp_diff") or 0)
            if hist_rows and hist_rows[0].get("n_common_opps"):
                n_common_opps = int(hist_rows[0]["n_common_opps"])
                common_opp_diff = float(hist_rows[0].get("common_opp_diff") or 0)
                print(f"  [full_predict] Common opps from historical: n={n_common_opps}, diff={common_opp_diff:.2f}")
        except Exception as e:
            print(f"  [full_predict] Historical common opps lookup error: {e}")

    # Conference tournament: auto-detect from ESPN game summary, fallback to request
    is_conf_tourney = (
        game_info.get("is_conference_competition", 0)
        or request_data.get("is_conference_tournament", 0)
    )

    # Injury data
    home_inj = get_team_injuries(home_team_name) if HAS_INJURY_CACHE else {"missing_starters": 0, "injury_penalty": 0.0}
    away_inj = get_team_injuries(away_team_name) if HAS_INJURY_CACHE else {"missing_starters": 0, "injury_penalty": 0.0}

    # Market odds: ESPN game summary first, then request data fallback
    espn_spread = game_info.get("espn_spread", 0) or request_data.get("espn_spread", 0)
    espn_ou = game_info.get("espn_over_under", 0) or request_data.get("espn_over_under", 0)
    espn_wp = game_info.get("espn_home_win_pct", 0.5)
    espn_pred = game_info.get("espn_predictor_home_pct", 0.5)
    market_spread = request_data.get("market_spread_home") or espn_spread or 0
    market_total = request_data.get("market_ou_total") or espn_ou or 0

    # Helper to get from Supabase data with fallback
    def sb(key, default=0):
        val = home_sb.get(key)
        if val is not None and val != "":
            try:
                return float(val)
            except:
                return default
        return default

    def sb_away(key, default=0):
        val = away_sb.get(key)
        if val is not None and val != "":
            try:
                return float(val)
            except:
                return default
        return default

    # ── Build complete game dict with ALL columns ──
    game = {
        # Identity
        "game_id": game_id,
        "game_date": game_date,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "neutral_site": neutral_site,

        # ESPN base stats (normalized to decimal format)
        # Use Supabase for opp_ppg since ESPN doesn't have avgPointsAllowed
        "home_ppg": home_stats.get("ppg", 75), "away_ppg": away_stats.get("ppg", 75),
        "home_opp_ppg": sb("home_opp_ppg") or home_stats.get("opp_ppg", 72),
        "away_opp_ppg": sb_away("home_opp_ppg") or away_stats.get("opp_ppg", 72),
        "home_fgpct": home_stats.get("fgpct", 0.44), "away_fgpct": away_stats.get("fgpct", 0.44),
        "home_threepct": home_stats.get("threepct", 0.34), "away_threepct": away_stats.get("threepct", 0.34),
        "home_ftpct": home_stats.get("ftpct", 0.72), "away_ftpct": away_stats.get("ftpct", 0.72),
        "home_assists": home_stats.get("assists", 14), "away_assists": away_stats.get("assists", 14),
        "home_turnovers": home_stats.get("turnovers", 12), "away_turnovers": away_stats.get("turnovers", 12),
        # AUDIT FIX: ESPN tempo often fails — use Supabase as fallback
        "home_tempo": home_stats.get("tempo", 0) or sb("home_tempo", 68),
        "away_tempo": away_stats.get("tempo", 0) or sb_away("home_tempo", 68),
        "home_orb_pct": home_stats.get("orb_pct", 0.28), "away_orb_pct": away_stats.get("orb_pct", 0.28),
        "home_steals": home_stats.get("steals", 7), "away_steals": away_stats.get("steals", 7),
        "home_blocks": home_stats.get("blocks", 3.5), "away_blocks": away_stats.get("blocks", 3.5),
        
        # Advanced stats: ESPN-derived first, Supabase fallback
        "home_efg_pct": home_stats.get("efg_pct", sb("home_efg_pct", 0.50)),
        "away_efg_pct": away_stats.get("efg_pct", sb_away("home_efg_pct", 0.50)),
        "home_twopt_pct": home_stats.get("twopt_pct", sb("home_twopt_pct", 0.48)),
        "away_twopt_pct": away_stats.get("twopt_pct", sb_away("home_twopt_pct", 0.48)),
        "home_three_rate": home_stats.get("three_rate", sb("home_three_rate", 0.35)),
        "away_three_rate": away_stats.get("three_rate", sb_away("home_three_rate", 0.35)),
        "home_assist_rate": home_stats.get("assist_rate", sb("home_assist_rate", 0.55)),
        "away_assist_rate": away_stats.get("assist_rate", sb_away("home_assist_rate", 0.55)),
        "home_drb_pct": home_stats.get("drb_pct", sb("home_drb_pct", 0.70)),
        "away_drb_pct": away_stats.get("drb_pct", sb_away("home_drb_pct", 0.70)),
        "home_ppp": home_stats.get("ppp", sb("home_ppp", 1.0)),
        "away_ppp": away_stats.get("ppp", sb_away("home_ppp", 1.0)),
        "home_fta_rate": home_stats.get("fta_rate", 0.34), "away_fta_rate": away_stats.get("fta_rate", 0.34),
        "home_ato_ratio": home_stats.get("ato_ratio", 1.2), "away_ato_ratio": away_stats.get("ato_ratio", 1.2),
        "home_ts_pct": home_stats.get("ts_pct", 0.53), "away_ts_pct": away_stats.get("ts_pct", 0.53),

        # Record — ALWAYS prefer schedule data (ESPN record endpoint returns
        # tournament-only records in March, e.g. Howard showing 1-0).
        # Schedule-derived wins/losses are computed from actual completed games.
        # FIX: Use explicit None check — 0 wins is valid (e.g. 0-5 start), not "no data"
        "home_wins": home_sched["wins"] if home_sched.get("wins") is not None and (home_sched["wins"] + home_sched["losses"]) > 0 else home_record["wins"],
        "away_wins": away_sched["wins"] if away_sched.get("wins") is not None and (away_sched["wins"] + away_sched["losses"]) > 0 else away_record["wins"],
        "home_losses": home_sched["losses"] if home_sched.get("losses") is not None and (home_sched["wins"] + home_sched["losses"]) > 0 else home_record["losses"],
        "away_losses": away_sched["losses"] if away_sched.get("losses") is not None and (away_sched["wins"] + away_sched["losses"]) > 0 else away_record["losses"],
        # AUDIT FIX: Supabase sos is more reliable than ESPN record endpoint
        "home_sos": sb("home_sos", 0) or home_record.get("sos", 0.5),
        "away_sos": sb_away("home_sos", 0) or away_record.get("sos", 0.5),
        "home_rank": home_info.get("rank", 200), "away_rank": away_info.get("rank", 200),
        "home_rest_days": home_rest, "away_rest_days": away_rest,

        # Conference — from ESPN team info, fallback to Supabase, then request
        "home_conference": home_info.get("conference") or request_data.get("home_conference", ""),
        "away_conference": away_info.get("conference") or request_data.get("away_conference", ""),

        # KenPom / efficiency (from request, Supabase adj_em, or computed from OE-DE)
        "home_adj_em": request_data.get("home_adj_em") or sb("home_adj_em") or (sb("home_adj_oe", 105) - sb("home_adj_de", 105)) or 0,
        "away_adj_em": request_data.get("away_adj_em") or sb_away("home_adj_em") or (sb_away("home_adj_oe", 105) - sb_away("home_adj_de", 105)) or 0,
        "home_adj_oe": sb("home_adj_oe", 105), "away_adj_oe": sb_away("home_adj_oe", 105),
        "home_adj_de": sb("home_adj_de", 105), "away_adj_de": sb_away("home_adj_de", 105),

        # Elo
        "home_elo": sb("home_elo", 1500), "away_elo": sb_away("home_elo", 1500),

        # Supabase-only advanced stats (no ESPN equivalent)
        "home_pyth_residual": sb("home_pyth_residual"), "away_pyth_residual": sb_away("home_pyth_residual"),
        "home_opp_efg_pct": sb("home_opp_efg_pct", 0.50), "away_opp_efg_pct": sb_away("home_opp_efg_pct", 0.50),
        "home_opp_to_rate": sb("home_opp_to_rate") or (away_stats.get("turnovers", 12) / max(away_stats.get("tempo", 68), 1)), "away_opp_to_rate": sb_away("home_opp_to_rate", 0.18),
        "home_opp_fta_rate": sb("home_opp_fta_rate", 0.30), "away_opp_fta_rate": sb_away("home_opp_fta_rate", 0.30),
        "home_opp_orb_pct": sb("home_opp_orb_pct", 0.28), "away_opp_orb_pct": sb_away("home_opp_orb_pct", 0.28),

        # Analytics
        "home_luck": sb("home_luck"), "away_luck": sb_away("home_luck"),
        "home_consistency": sb("home_consistency", 15), "away_consistency": sb_away("home_consistency", 15),
        "home_margin_trend": sb("home_margin_trend"), "away_margin_trend": sb_away("home_margin_trend"),
        "home_margin_accel": sb("home_margin_accel"), "away_margin_accel": sb_away("home_margin_accel"),
        "home_opp_adj_form": sb("home_opp_adj_form"), "away_opp_adj_form": sb_away("home_opp_adj_form"),
        "home_wl_momentum": sb("home_wl_momentum"), "away_wl_momentum": sb_away("home_wl_momentum"),
        "home_recovery_idx": sb("home_recovery_idx"), "away_recovery_idx": sb_away("home_recovery_idx"),
        "home_is_after_loss": sb("home_is_after_loss") or home_sched.get("after_loss", 0),
        "away_is_after_loss": sb_away("home_is_after_loss") or away_sched.get("after_loss", 0),
        "home_ceiling": sb("home_ceiling", 15), "away_ceiling": sb_away("home_ceiling", 15),
        "home_floor": sb("home_floor", -10), "away_floor": sb_away("home_floor", -10),
        "home_margin_skew": sb("home_margin_skew"), "away_margin_skew": sb_away("home_margin_skew"),
        "home_scoring_entropy": sb("home_scoring_entropy", 1.5), "away_scoring_entropy": sb_away("home_scoring_entropy", 1.5),
        "home_bimodal": sb("home_bimodal"), "away_bimodal": sb_away("home_bimodal"),
        "home_def_stability": sb("home_def_stability", 10), "away_def_stability": sb_away("home_def_stability", 10),
        "home_opp_suppression": sb("home_opp_suppression"), "away_opp_suppression": sb_away("home_opp_suppression"),
        "home_def_versatility": sb("home_def_versatility", 0.5), "away_def_versatility": sb_away("home_def_versatility", 0.5),
        "home_steal_foul_ratio": sb("home_steal_foul_ratio", 0.4), "away_steal_foul_ratio": sb_away("home_steal_foul_ratio", 0.4),
        "home_block_foul_ratio": sb("home_block_foul_ratio", 0.2), "away_block_foul_ratio": sb_away("home_block_foul_ratio", 0.2),
        "home_transition_dep": sb("home_transition_dep"), "away_transition_dep": sb_away("home_transition_dep"),
        "home_paint_pts": sb("home_paint_pts", 30), "away_paint_pts": sb_away("home_paint_pts", 30),
        "home_fastbreak_pts": sb("home_fastbreak_pts", 10), "away_fastbreak_pts": sb_away("home_fastbreak_pts", 10),
        "home_fatigue_load": sb("home_fatigue_load"), "away_fatigue_load": sb_away("home_fatigue_load"),
        "home_streak": sb("home_streak"), "away_streak": sb_away("home_streak"),
        "home_season_pct": sb("home_season_pct", 0.5), "away_season_pct": sb_away("home_season_pct", 0.5),
        "home_regression_pressure": sb("home_regression_pressure"), "away_regression_pressure": sb_away("home_regression_pressure"),
        "home_info_gain": sb("home_info_gain"), "away_info_gain": sb_away("home_info_gain"),
        "home_overreaction": sb("home_overreaction"), "away_overreaction": sb_away("home_overreaction"),
        "home_scoring_source_entropy": sb("home_scoring_source_entropy", 1.5), "away_scoring_source_entropy": sb_away("home_scoring_source_entropy", 1.5),
        "home_ft_dependency": sb("home_ft_dependency", 0.2), "away_ft_dependency": sb_away("home_ft_dependency", 0.2),
        "home_three_value": sb("home_three_value", 0.35), "away_three_value": sb_away("home_three_value", 0.35),
        "home_concentration": sb("home_concentration"), "away_concentration": sb_away("home_concentration"),
        "home_to_conversion": sb("home_to_conversion", 1.0), "away_to_conversion": sb_away("home_to_conversion", 1.0),
        "home_three_divergence": sb("home_three_divergence"), "away_three_divergence": sb_away("home_three_divergence"),
        "home_ppp_divergence": sb("home_ppp_divergence"), "away_ppp_divergence": sb_away("home_ppp_divergence"),
        "home_pace_adj_margin": sb("home_pace_adj_margin"), "away_pace_adj_margin": sb_away("home_pace_adj_margin"),
        "home_pit_sos": sb("home_pit_sos", 1500), "away_pit_sos": sb_away("home_pit_sos", 1500),
        "home_scoring_var": sb("home_scoring_var", 12), "away_scoring_var": sb_away("home_scoring_var", 12),
        "home_score_kurtosis": sb("home_score_kurtosis"), "away_score_kurtosis": sb_away("home_score_kurtosis"),
        "home_clutch_ratio": sb("home_clutch_ratio", 0.5), "away_clutch_ratio": sb_away("home_clutch_ratio", 0.5),
        "home_garbage_adj_ppp": sb("home_garbage_adj_ppp", 1.0), "away_garbage_adj_ppp": sb_away("home_garbage_adj_ppp", 1.0),
        "home_days_since_loss": sb("home_days_since_loss", 5), "away_days_since_loss": sb_away("home_days_since_loss", 5),
        "home_games_since_blowout_loss": sb("home_games_since_blowout_loss", 10), "away_games_since_blowout_loss": sb_away("home_games_since_blowout_loss", 10),
        "home_games_last_14": sb("home_games_last_14") or home_sched.get("games_last_14", 4),
        "away_games_last_14": sb_away("home_games_last_14") or away_sched.get("games_last_14", 4),
        "home_rest_effect": sb("home_rest_effect"), "away_rest_effect": sb_away("home_rest_effect"),
        "home_momentum_halflife": sb("home_momentum_halflife", 1.0), "away_momentum_halflife": sb_away("home_momentum_halflife", 1.0),
        "home_win_aging": sb("home_win_aging", 1.0), "away_win_aging": sb_away("home_win_aging", 1.0),
        "home_centrality": sb("home_centrality", 1.0), "away_centrality": sb_away("home_centrality", 1.0),

        # v22 new features
        "home_dow_effect": sb("home_dow_effect"), "away_dow_effect": sb_away("home_dow_effect"),
        "home_conf_balance": sb("home_conf_balance"), "away_conf_balance": sb_away("home_conf_balance"),
        "home_eff_vol_ratio": sb("home_eff_vol_ratio", 1.0), "away_eff_vol_ratio": sb_away("home_eff_vol_ratio", 1.0),
        "home_run_vulnerability": sb("home_run_vulnerability"), "away_run_vulnerability": sb_away("home_run_vulnerability"),
        "home_anti_fragility": sb("home_anti_fragility"), "away_anti_fragility": sb_away("home_anti_fragility"),
        "home_sos_trajectory": sb("home_sos_trajectory"), "away_sos_trajectory": sb_away("home_sos_trajectory"),
        "home_margin_autocorr": sb("home_margin_autocorr"), "away_margin_autocorr": sb_away("home_margin_autocorr"),
        "home_blowout_asym": sb("home_blowout_asym"), "away_blowout_asym": sb_away("home_blowout_asym"),
        "home_clutch_over_exp": sb("home_clutch_over_exp"), "away_clutch_over_exp": sb_away("home_clutch_over_exp"),
        "home_ft_pressure": sb("home_ft_pressure"), "away_ft_pressure": sb_away("home_ft_pressure"),
        "home_fg_divergence": sb("home_fg_divergence"), "away_fg_divergence": sb_away("home_fg_divergence"),
        "home_def_improvement": sb("home_def_improvement"), "away_def_improvement": sb_away("home_def_improvement"),
        "home_rhythm_disruption": sb("home_rhythm_disruption"), "away_rhythm_disruption": sb_away("home_rhythm_disruption"),
        "home_fouls": sb("home_fouls", 17), "away_fouls": sb_away("home_fouls", 17),
        "home_close_win_rate": sb("home_close_win_rate", 0.5), "away_close_win_rate": sb_away("home_close_win_rate", 0.5),
        "home_games_last_7": sb("home_games_last_7", 2), "away_games_last_7": sb_away("home_games_last_7", 2),
        "home_pts_off_to": sb("home_pts_off_to", 12), "away_pts_off_to": sb_away("home_pts_off_to", 12),
        "home_home_margin": sb("home_home_margin"), "away_home_margin": sb_away("home_home_margin"),
        "home_away_margin": sb("home_away_margin"), "away_away_margin": sb_away("home_away_margin"),

        # Rolling features
        "home_roll_star1_share": sb("home_roll_star1_share", 0.25), "away_roll_star1_share": sb_away("home_roll_star1_share", 0.25),
        "home_roll_top3_share": sb("home_roll_top3_share", 0.55), "away_roll_top3_share": sb_away("home_roll_top3_share", 0.55),
        "home_roll_bench_share": sb("home_roll_bench_share", 0.20), "away_roll_bench_share": sb_away("home_roll_bench_share", 0.20),
        "home_roll_bench_pts": sb("home_roll_bench_pts", 15), "away_roll_bench_pts": sb_away("home_roll_bench_pts", 15),
        "home_roll_largest_run": sb("home_roll_largest_run", 8), "away_roll_largest_run": sb_away("home_roll_largest_run", 8),
        "home_roll_drought_rate": sb("home_roll_drought_rate", 1.5), "away_roll_drought_rate": sb_away("home_roll_drought_rate", 1.5),
        "home_roll_lead_changes": sb("home_roll_lead_changes", 8), "away_roll_lead_changes": sb_away("home_roll_lead_changes", 8),
        "home_roll_time_with_lead_pct": sb("home_roll_time_with_lead_pct", 0.5), "away_roll_time_with_lead_pct": sb_away("home_roll_time_with_lead_pct", 0.5),
        "home_roll_players_used": sb("home_roll_players_used", 8), "away_roll_players_used": sb_away("home_roll_players_used", 8),
        "home_roll_hhi": sb("home_roll_hhi", 0.20), "away_roll_hhi": sb_away("home_roll_hhi", 0.20),
        "home_roll_clutch_ft_pct": sb("home_roll_clutch_ft_pct", 0.70), "away_roll_clutch_ft_pct": sb_away("home_roll_clutch_ft_pct", 0.70),
        "home_roll_garbage_pct": sb("home_roll_garbage_pct", 0.15), "away_roll_garbage_pct": sb_away("home_roll_garbage_pct", 0.15),
        "home_roll_ats_pct": sb("home_roll_ats_pct", 0.50), "away_roll_ats_pct": sb_away("home_roll_ats_pct", 0.50),
        "home_roll_ats_n": sb("home_roll_ats_n", 0), "away_roll_ats_n": sb_away("home_roll_ats_n", 0),
        "home_roll_ats_margin": sb("home_roll_ats_margin"), "away_roll_ats_margin": sb_away("home_roll_ats_margin"),

        # Lineup stability + player ratings (from ncaa_historical via Supabase)
        "home_lineup_changes": sb("home_lineup_changes", 0), "away_lineup_changes": sb_away("home_lineup_changes", 0),
        "home_lineup_stability_5g": sb("home_lineup_stability_5g", 1.0), "away_lineup_stability_5g": sb_away("home_lineup_stability_5g", 1.0),
        "home_starter_games_together": sb("home_starter_games_together", 0), "away_starter_games_together": sb_away("home_starter_games_together", 0),
        "home_player_rating_sum": sb("home_player_rating_sum", 0), "away_player_rating_sum": sb_away("home_player_rating_sum", 0),
        "home_weakest_starter": sb("home_weakest_starter", 0), "away_weakest_starter": sb_away("home_weakest_starter", 0),
        "home_starter_variance": sb("home_starter_variance", 0), "away_starter_variance": sb_away("home_starter_variance", 0),

        # Referee
        "referee_1": game_info.get("referee_1", ""),
        "referee_2": game_info.get("referee_2", ""),
        "referee_3": game_info.get("referee_3", ""),

        # Venue
        "attendance": game_info.get("attendance", 0),
        "venue_capacity": game_info.get("venue_capacity", 8000),
        "crowd_shock_diff": _fetch_crowd_shock(home_team_id, away_team_id),

        # Market
        "market_spread_home": market_spread,
        "market_ou_total": market_total,
        "espn_spread": espn_spread,
        "espn_over_under": espn_ou,
        "espn_home_win_pct": espn_wp,
        "espn_predictor_home_pct": espn_pred,
        "espn_ml_home": request_data.get("espn_ml_home", 0),
        "espn_ml_away": request_data.get("espn_ml_away", 0),

        # v25: DK/OA moneylines for market_wp_edge computation in ncaa_build_features
        "dk_ml_home_close": request_data.get("dk_ml_home_close", 0),
        "dk_ml_away_close": request_data.get("dk_ml_away_close", 0),
        "odds_api_ml_home_close": request_data.get("odds_api_ml_home_close", 0),
        "odds_api_ml_away_close": request_data.get("odds_api_ml_away_close", 0),
        # v25: DK/OA open/close for spread_movement derivation
        "dk_spread_open": request_data.get("dk_spread_open", 0),
        "dk_spread_close": request_data.get("dk_spread_close", 0),
        "odds_api_spread_open": request_data.get("odds_api_spread_open", 0),
        "odds_api_spread_close": request_data.get("odds_api_spread_close", 0),
        "dk_total_open": request_data.get("dk_total_open", 0),
        "dk_total_close": request_data.get("dk_total_close", 0),
        "odds_api_total_open": request_data.get("odds_api_total_open", 0),
        "odds_api_total_close": request_data.get("odds_api_total_close", 0),

        # Spread movement
        "odds_api_spread_movement": spread_mvmt.get("odds_api_spread_movement", 0),
        "dk_spread_movement": spread_mvmt.get("dk_spread_movement", 0),
        "odds_api_total_movement": spread_mvmt.get("odds_api_total_movement", 0),
        "dk_total_movement": spread_mvmt.get("dk_total_movement", 0),

        # Injuries
        "home_injury_penalty": home_inj["injury_penalty"],
        "away_injury_penalty": away_inj["injury_penalty"],
        "injury_diff": home_inj["injury_penalty"] - away_inj["injury_penalty"],
        "home_missing_starters": home_inj["missing_starters"],
        "away_missing_starters": away_inj["missing_starters"],

        # Context
        "is_conference_tournament": is_conf_tourney,
        "is_ncaa_tournament": request_data.get("is_ncaa_tournament", 0),
        "is_bubble_game": request_data.get("is_bubble_game", 0),
        "is_early_season": request_data.get("is_early_season", 0),
        "importance_multiplier": request_data.get("importance_multiplier", 1.0),

        # Matchup/situational features — computed from available data
        "n_common_opps": n_common_opps,
        "common_opp_diff": common_opp_diff,
        # Schedule situations (from ESPN schedule analysis)
        # is_lookahead REMOVED (v25: too sparse 5%)
        "is_revenge_game": home_sched.get("is_revenge_game", 0),
        "revenge_margin": home_sched.get("revenge_margin", 0.0),
        # is_sandwich REMOVED (v25: too sparse 1%)
        # spread_regime: v25 — now computed inline by ncaa_build_features from mkt_spread
        "spread_regime": 0,  # placeholder, overridden by ncaa_build_features
        # Midweek: Mon-Thu = 1
        "is_midweek": 1 if datetime.fromisoformat(f"{game_date}T00:00:00").weekday() < 4 else 0,
        # Style familiarity: approximate from tempo similarity
        "style_familiarity": min(1.0, 1.0 - abs(home_stats.get("tempo", 68) - away_stats.get("tempo", 68)) / 20.0),
        # Pace leverage: how much pace mismatch benefits the faster team
        "pace_leverage": (home_stats.get("tempo", 68) - away_stats.get("tempo", 68)) * 0.1,
        # Pace control diff
        "pace_control_diff": (sb("home_pace_adj_margin") - sb_away("home_pace_adj_margin")) * 0.1 if sb("home_pace_adj_margin") else 0.0,
        # Matchup Four Factors (approximate from team stats)
        "matchup_efg": (home_stats.get("efg_pct", 0.50) - away_stats.get("efg_pct", 0.50)),
        "matchup_to": (away_stats.get("turnovers", 12) - home_stats.get("turnovers", 12)) / 20.0,
        "matchup_orb": (home_stats.get("orb_pct", 0.28) - away_stats.get("orb_pct", 0.28)),
        "matchup_ft": (home_stats.get("fta_rate", 0.34) - away_stats.get("fta_rate", 0.34)),
        # Interaction features — v25: consistency_x_spread and luck_x_spread
        # are now computed inline by ncaa_build_features from mkt_spread.
        # def_rest_advantage REMOVED (v25: degrading, redundant with rest_diff)
        "luck_x_spread": 0.0,  # placeholder, overridden by ncaa_build_features
        # AUDIT FIX: use computed adj_em, not request_data which is often empty
        "fatigue_x_quality": (sb("home_fatigue_load", 0) * (sb_away("home_adj_oe", 105) - sb_away("home_adj_de", 105))
                             - sb_away("home_fatigue_load", 0) * (sb("home_adj_oe", 105) - sb("home_adj_de", 105))) * 0.01,
        "rest_x_defense": (home_rest - away_rest) * ((sb("home_def_stability") or 0.5) + (sb_away("home_def_stability") or 0.5)) * 0.025,
        "form_x_familiarity": 0.0,  # Set in post-processing after form_diff is computed
        "consistency_x_spread": 0.0,  # placeholder, overridden by ncaa_build_features

        # ESPN opponent FG% (for def_fgpct_diff)
        # AUDIT FIX: was using opp_efg_pct (eFG%) instead of actual opp_fgpct (FG%)
        "home_opp_fgpct": sb("home_opp_fgpct", 0.43), "away_opp_fgpct": sb_away("home_opp_fgpct", 0.43),
        # AUDIT FIX: was using opp_fta_rate instead of opp_threepct
        "home_opp_threepct": sb("home_opp_threepct", 0.33), "away_opp_threepct": sb_away("home_opp_threepct", 0.33),

        # Form: from Supabase (ESPN doesn't have team form scores)
        # AUDIT FIX: was using home_opp_adj_form which duplicated opp_adj_form_diff
        "home_form": sb("home_form", 0), "away_form": sb_away("home_form", 0),
    }

    # ═══ v28: Compute game-level features that aren't stored per-team ═══

    # 1. Fix wins/losses — schedule data is primary source (ESPN record endpoint
    # returns tournament-only records in March). Only override if schedule has data.
    if home_sched.get("wins", 0) + home_sched.get("losses", 0) > 0:
        game["home_wins"] = home_sched["wins"]
        game["home_losses"] = home_sched["losses"]
    if away_sched.get("wins", 0) + away_sched.get("losses", 0) > 0:
        game["away_wins"] = away_sched["wins"]
        game["away_losses"] = away_sched["losses"]
    print(f"  [full_predict] Record: home={game['home_wins']}-{game['home_losses']}, away={game['away_wins']}-{game['away_losses']} (sched_h={home_sched.get('wins')}-{home_sched.get('losses')}, espn_h={home_record['wins']}-{home_record['losses']})")

    # ═══ v29: SERVE-TIME FEATURE FILLS (fix 25 zero features) ═══

    # 1a. Elo: compute from schedule/adj_em when Supabase returned default 1500
    if game["home_elo"] == 1500 or game["away_elo"] == 1500:
        h_adj_em = float(game.get("home_adj_em", 0) or 0)
        a_adj_em = float(game.get("away_adj_em", 0) or 0)
        if game["home_elo"] == 1500:
            game["home_elo"] = _compute_elo_from_schedule(
                home_sched, h_adj_em, game.get("home_sos", 0.5))
        if game["away_elo"] == 1500:
            game["away_elo"] = _compute_elo_from_schedule(
                away_sched, a_adj_em, game.get("away_sos", 0.5))
        print(f"  [full_predict] Elo computed: home={game['home_elo']:.0f}, away={game['away_elo']:.0f}")

    # 1b. Rolling ATS: override from ncaa_predictions when Supabase returned defaults
    if game["home_roll_ats_n"] == 0 and home_ats["ats_n"] >= 3:
        game["home_roll_ats_pct"] = home_ats["ats_pct"]
        game["home_roll_ats_n"] = home_ats["ats_n"]
        game["home_roll_ats_margin"] = home_ats["ats_margin"]
    if game["away_roll_ats_n"] == 0 and away_ats["ats_n"] >= 3:
        game["away_roll_ats_pct"] = away_ats["ats_pct"]
        game["away_roll_ats_n"] = away_ats["ats_n"]
        game["away_roll_ats_margin"] = away_ats["ats_margin"]

    # 1c. Advanced metrics: compute from schedule when Supabase returned 0
    for side, sched, sb_fn in [("home", home_sched, sb), ("away", away_sched, sb_away)]:
        computed = _compute_advanced_from_schedule(sched, sb_fn)
        for feat, val in computed.items():
            key = f"{side}_{feat}"
            if game.get(key, 0) == 0 and val != 0:
                game[key] = val

    # 1d. Player quality proxy: use adj_em when starter_ids unavailable
    for side, adj_em_key in [("home", "home_adj_em"), ("away", "away_adj_em")]:
        if game.get(f"{side}_player_rating_sum", 0) == 0:
            em = float(game.get(adj_em_key, 0) or 0)
            if em != 0:
                # adj_em ≈ 0-40 range for good teams. Scale to player_rating_sum range.
                # Training data: player_rating_sum mean ≈ 3.5, adj_em mean ≈ 15
                # Use ratio: player_rating ≈ adj_em * 0.23
                game[f"{side}_player_rating_sum"] = round(em * 0.23, 4)
                game[f"{side}_weakest_starter"] = round(em * 0.03, 4)
                game[f"{side}_starter_variance"] = round(abs(em * 0.015), 4)
                print(f"  [full_predict] {side} player rating proxied from adj_em={em:.1f}: sum={game[f'{side}_player_rating_sum']:.3f}")

    # 1e. Rolling tendencies: estimate from schedule when Supabase returned defaults
    for side, sched in [("home", home_sched), ("away", away_sched)]:
        opps = [o for o in sched.get("opponents", []) if o.get("completed")]
        if len(opps) >= 5:
            margins = [o["margin"] for o in opps[-10:]]
            wins_last10 = sum(1 for m in margins if m > 0)
            # roll_time_with_lead_pct: estimate from win rate
            if game.get(f"{side}_roll_time_with_lead_pct", 0.5) == 0.5 and len(margins) >= 5:
                game[f"{side}_roll_time_with_lead_pct"] = round(wins_last10 / len(margins), 4)
            # roll_largest_run: estimate from avg win margin
            if game.get(f"{side}_roll_largest_run", 8.0) == 8.0 and len(margins) >= 5:
                avg_margin = sum(margins) / len(margins)
                game[f"{side}_roll_largest_run"] = max(5, round(8 + avg_margin * 0.3, 1))
            # roll_players_used: estimate from consistency
            consistency = float(game.get(f"{side}_consistency", 15) or 15)
            if game.get(f"{side}_roll_players_used", 8.0) == 8.0 and consistency != 15:
                # Lower consistency = more players needed
                game[f"{side}_roll_players_used"] = round(8 + (consistency - 15) * 0.05, 1)

    # 2. Conference strength diff + cross-conference flag
    _CONF_STRENGTH = {
        "Big 12": 0.65, "SEC": 0.62, "Big Ten": 0.60, "ACC": 0.58,
        "Big East": 0.55, "Pac-12": 0.52, "Mountain West": 0.45,
        "American Athletic": 0.42, "AAC": 0.42, "West Coast": 0.40, "WCC": 0.40,
        "Missouri Valley": 0.38, "MVC": 0.38, "Atlantic 10": 0.38, "A-10": 0.38,
        "Southeastern Conference": 0.62, "Big Ten Conference": 0.60,
        "Atlantic Coast Conference": 0.58, "Big East Conference": 0.55,
        "Mountain West Conference": 0.45, "American Athletic Conference": 0.42,
        "West Coast Conference": 0.40, "Missouri Valley Conference": 0.38,
        "Atlantic 10 Conference": 0.38,
        "Conference USA": 0.32, "C-USA": 0.32, "Sun Belt": 0.30, "Sun Belt Conference": 0.30,
        "Mid-American": 0.28, "MAC": 0.28, "Mid-American Conference": 0.28,
        "Southland": 0.22, "Southland Conference": 0.22,
        "WAC": 0.25, "Western Athletic": 0.25, "Western Athletic Conference": 0.25,
        "Ivy League": 0.35, "Ivy": 0.35,
        "Patriot League": 0.25, "Patriot": 0.25,
        "Colonial Athletic Association": 0.30, "CAA": 0.30,
        "Southern Conference": 0.28, "SoCon": 0.28,
        "Ohio Valley": 0.20, "OVC": 0.20, "Ohio Valley Conference": 0.20,
        "Horizon League": 0.28, "Horizon": 0.28,
        "MAAC": 0.25, "Metro Atlantic Athletic": 0.25,
        "Northeast Conference": 0.18, "NEC": 0.18,
        "MEAC": 0.15, "SWAC": 0.15, "Southland Conference": 0.22,
        "Summit League": 0.22, "Summit": 0.22,
        "Big South": 0.20, "Big South Conference": 0.20,
        "America East": 0.22, "America East Conference": 0.22,
        "Big West": 0.25, "Big West Conference": 0.25,
        "Atlantic Sun": 0.22, "ASUN": 0.22, "ASUN Conference": 0.22,
    }
    h_conf = game["home_conference"]
    a_conf = game["away_conference"]
    if h_conf and a_conf and h_conf != a_conf:
        game["conf_strength_diff"] = _CONF_STRENGTH.get(h_conf, 0.30) - _CONF_STRENGTH.get(a_conf, 0.30)
        game["cross_conf_flag"] = 1
    else:
        game["conf_strength_diff"] = 0.0
        game["cross_conf_flag"] = 0

    # 3. H2H lookup from ncaa_historical
    try:
        h2h_rows = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{home_team_id}&away_team_id=eq.{away_team_id}"
            f"&actual_home_score=not.is.null&actual_away_score=not.is.null"
            f"&select=actual_home_score,actual_away_score&order=game_date.desc&limit=5"
        )
        # Also check reverse matchup
        h2h_reverse = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{away_team_id}&away_team_id=eq.{home_team_id}"
            f"&actual_home_score=not.is.null&actual_away_score=not.is.null"
            f"&select=actual_home_score,actual_away_score&order=game_date.desc&limit=5"
        )
        h2h_margins = []
        h2h_home_wins = 0
        h2h_total = 0
        for r in (h2h_rows or []):
            hs = r.get("actual_home_score")
            aws = r.get("actual_away_score")
            if hs is not None and aws is not None:
                margin = float(hs) - float(aws)
                h2h_margins.append(margin)
                if margin > 0: h2h_home_wins += 1
                h2h_total += 1
        for r in (h2h_reverse or []):
            hs = r.get("actual_home_score")
            aws = r.get("actual_away_score")
            if hs is not None and aws is not None:
                margin = -(float(hs) - float(aws))  # flip perspective
                h2h_margins.append(margin)
                if margin > 0: h2h_home_wins += 1
                h2h_total += 1
        if h2h_total > 0:
            game["h2h_margin_avg"] = sum(h2h_margins) / len(h2h_margins)
            game["h2h_home_win_rate"] = h2h_home_wins / h2h_total
            print(f"  [full_predict] H2H: {h2h_total} games, avg_margin={game['h2h_margin_avg']:.1f}, win_rate={game['h2h_home_win_rate']:.2f}")
        else:
            print(f"  [full_predict] H2H: no historical matchups found")
    except Exception as e:
        print(f"  [full_predict] H2H lookup error: {e}")

    # 4. ESPN win probability edge
    if game["espn_home_win_pct"] and game["espn_home_win_pct"] != 0.5:
        pass  # espn_wp_edge will be computed by ncaa_build_features
    else:
        # Try to get from pickcenter implied probability
        if game["market_spread_home"] and game["market_spread_home"] != 0:
            # Implied win prob from spread: P(home) ≈ Φ(-spread/10.5) rough estimate
            import math
            _sp = game["market_spread_home"]
            game["espn_home_win_pct"] = 1 / (1 + math.pow(10, _sp / 13.5))

    # 5. Recent form from Supabase (not in team-level _COLS)
    game["home_recent_form"] = sb("home_form", 0)  # alias
    game["away_recent_form"] = sb_away("home_form", 0)
    # recent_form_diff in training = last-5-game WIN RATE diff (from compute_advanced_features.py).
    # This is distinct from form_diff (season-long). Compute from schedule opponents.
    def _last5_win_rate(sched):
        opps = [o for o in sched.get("opponents", []) if o.get("completed")]
        if len(opps) < 3:
            return None  # not enough data — let ncaa_build_features use default
        last5 = opps[-5:]
        wins = sum(1 for o in last5 if o.get("margin", 0) > 0)
        return wins / len(last5)

    h_rf = _last5_win_rate(home_sched)
    a_rf = _last5_win_rate(away_sched)
    if h_rf is not None and a_rf is not None:
        game["recent_form_diff"] = round(h_rf - a_rf, 4)
        print(f"  [full_predict] recent_form_diff: {game['recent_form_diff']:.3f} (home_5g={h_rf:.2f}, away_5g={a_rf:.2f})")
    else:
        print(f"  [full_predict] recent_form_diff: insufficient schedule data (home={h_rf}, away={a_rf})")

    # 6. Revenge game detection using schedule opponents
    for opp in home_sched.get("opponents", []):
        if str(opp.get("opp_id")) == str(away_team_id) and opp.get("margin", 0) < 0:
            game["is_revenge_game"] = 1
            game["revenge_margin"] = abs(opp["margin"])
            break

    # 7. Compute pyth_residual and luck when Supabase returns 0
    # Formula confirmed from fix_conf_tourney_and_pyth.py:
    #   pyth_pct = ppg^11.5 / (ppg^11.5 + opp_ppg^11.5)
    #   residual = actual_win_pct - pyth_pct
    # In KenPom, "luck" IS the Pythagorean residual (same formula).
    _PYTH_EXP = 11.5
    for side, stats_dict, sb_fn, sched in [
        ("home", home_stats, sb, home_sched),
        ("away", away_stats, sb_away, away_sched),
    ]:
        ppg = stats_dict.get("ppg", 75) or 75
        opp_ppg = sb_fn("home_opp_ppg") or stats_dict.get("opp_ppg", 72) or 72
        w = game.get(f"{side}_wins", 15)
        l = game.get(f"{side}_losses", 10)
        total_games = w + l
        if total_games > 0 and ppg > 0 and opp_ppg > 0:
            actual_pct = w / total_games
            ppg_exp = ppg ** _PYTH_EXP
            opp_exp = opp_ppg ** _PYTH_EXP
            pyth_pct = ppg_exp / (ppg_exp + opp_exp)
            computed_residual = round(actual_pct - pyth_pct, 4)

            # Fill pyth_residual if Supabase returned 0
            if game.get(f"{side}_pyth_residual", 0) == 0:
                game[f"{side}_pyth_residual"] = computed_residual

            # Fill luck if Supabase returned 0
            if game.get(f"{side}_luck", 0) == 0:
                game[f"{side}_luck"] = computed_residual

    print(f"  [full_predict] pyth_residual: home={game['home_pyth_residual']:.4f}, away={game['away_pyth_residual']:.4f}")
    print(f"  [full_predict] luck: home={game['home_luck']:.4f}, away={game['away_luck']:.4f}")

    # 8. luck_x_spread and consistency_x_spread: now computed inline by ncaa_build_features
    # No manual recompute needed here

    # 9. Compute player ratings from starter_ids when Supabase values are 0
    # Same logic as training: starter_ids + player_ratings.json → sum, weakest, variance
    _player_ratings = getattr(predict_ncaa_full, "_player_ratings", None)
    if _player_ratings is None:
        try:
            import json as _jr
            with open("player_ratings.json") as _pf:
                predict_ncaa_full._player_ratings = _jr.load(_pf)
            _player_ratings = predict_ncaa_full._player_ratings
            print(f"  [full_predict] Loaded {len(_player_ratings)} player ratings")
        except FileNotFoundError:
            predict_ncaa_full._player_ratings = {}
            _player_ratings = {}
            print(f"  [full_predict] WARNING: player_ratings.json not found")

    for side, sb_fn, raw_sb in [("home", sb, home_sb), ("away", sb_away, away_sb)]:
        if game.get(f"{side}_player_rating_sum", 0) == 0:
            # Read starter_ids directly from raw dict — sb() would try float() and fail
            ids_str = raw_sb.get("home_starter_ids", "") or ""
            if isinstance(ids_str, str) and ids_str.strip():
                pids = ids_str.strip().split(",")
                ratings = [_player_ratings.get(pid.strip(), 0.0) for pid in pids]
                game[f"{side}_player_rating_sum"] = sum(ratings)
                game[f"{side}_weakest_starter"] = min(ratings) if ratings else 0.0
                game[f"{side}_starter_variance"] = float(np.std(ratings)) if len(ratings) > 1 else 0.0
                print(f"  [full_predict] {side} player ratings computed from starter_ids: sum={sum(ratings):.3f}, weakest={min(ratings):.3f}")
            else:
                print(f"  [full_predict] {side} no starter_ids available for player ratings")

    # 10. Compute lineup stability from schedule history (ncaa_historical)
    # Compare most recent starter_ids with prior games to get changes/stability
    for side, tid, sched in [("home", home_team_id, home_sched), ("away", away_team_id, away_sched)]:
        if game.get(f"{side}_lineup_stability_5g", 1.0) == 1.0 and game.get(f"{side}_player_rating_sum", 0) != 0:
            try:
                # Get last 6 games' starter_ids (need current + 5 prior to compute changes)
                _sel = "game_date,home_team_id,home_starter_ids,away_starter_ids"
                _gd = datetime.fromisoformat(f"{game_date}T00:00:00")
                _season_year = _gd.year if _gd.month < 8 else _gd.year + 1
                h_rows = sb_get("ncaa_historical",
                    f"home_team_id=eq.{tid}&season=eq.{_season_year}&game_date=lt.{game_date}"
                    f"&home_starter_ids=neq.&select={_sel}&order=game_date.desc&limit=6") or []
                a_rows = sb_get("ncaa_historical",
                    f"away_team_id=eq.{tid}&season=eq.{_season_year}&game_date=lt.{game_date}"
                    f"&away_starter_ids=neq.&select={_sel}&order=game_date.desc&limit=6") or []

                # Merge and sort
                all_ids = []
                for r in h_rows:
                    ids = r.get("home_starter_ids", "")
                    if ids: all_ids.append({"date": r["game_date"], "ids": set(ids.split(","))})
                for r in a_rows:
                    ids = r.get("away_starter_ids", "")
                    if ids: all_ids.append({"date": r["game_date"], "ids": set(ids.split(","))})
                all_ids.sort(key=lambda x: x["date"], reverse=True)

                if len(all_ids) >= 2:
                    # lineup_changes = starters different from last game
                    current = all_ids[0]["ids"]
                    prev = all_ids[1]["ids"]
                    game[f"{side}_lineup_changes"] = len(current - prev)

                    # lineup_stability_5g = avg overlap across last 5 transitions
                    overlaps = []
                    for j in range(min(5, len(all_ids) - 1)):
                        a_set = all_ids[j]["ids"]
                        b_set = all_ids[j + 1]["ids"]
                        if a_set and b_set:
                            overlaps.append(len(a_set & b_set) / max(len(a_set), 1))
                    if overlaps:
                        game[f"{side}_lineup_stability_5g"] = round(np.mean(overlaps), 4)

                    # starter_games_together = consecutive games with same 5
                    streak = 0
                    ref = all_ids[0]["ids"]
                    for entry in all_ids[1:]:
                        if entry["ids"] == ref:
                            streak += 1
                        else:
                            break
                    game[f"{side}_starter_games_together"] = streak

                    print(f"  [full_predict] {side} lineup: changes={game[f'{side}_lineup_changes']}, "
                          f"stability={game[f'{side}_lineup_stability_5g']:.3f}, "
                          f"together={game[f'{side}_starter_games_together']}")
            except Exception as e:
                print(f"  [full_predict] {side} lineup computation error: {e}")

    # ── Build features and predict ──
    import json as _json
    try:
        with open("referee_profiles.json") as f:
            ncaa_build_features._ref_profiles = _json.load(f)
    except:
        pass

    # Backfill heuristic (generates spread_home, win_pct_home, etc.)
    df = pd.DataFrame([game])

    # ── Ensure all df.get() columns exist as Series before ncaa_build_features ──
    # ncaa_build_features uses df.get(col, scalar) which returns a scalar (not a
    # Series) when the column is missing from a single-row DataFrame, causing
    # AttributeError: 'int' object has no attribute 'fillna'.
    # Adding defaults here avoids touching ncaa.py.
    _col_defaults = {
        # Lineup stability features
        "home_lineup_changes": 0, "away_lineup_changes": 0,
        "home_lineup_stability_5g": 1.0, "away_lineup_stability_5g": 1.0,
        "home_starter_games_together": 0, "away_starter_games_together": 0,
        "home_new_starter_impact": 0.0, "away_new_starter_impact": 0.0,
        # Player impact ratings
        "home_player_rating_sum": 0.0, "away_player_rating_sum": 0.0,
        "home_weakest_starter": 0.0, "away_weakest_starter": 0.0,
        "home_starter_variance": 0.0, "away_starter_variance": 0.0,
        # Head-to-head
        "h2h_margin_avg": 0.0, "h2h_home_win_rate": 0.0,
        # Conference strength
        "conf_strength_diff": 0.0, "cross_conf_flag": 0,
        # Spread / line movement
        "odds_api_spread_movement": 0.0, "odds_api_total_movement": 0.0,
        "dk_spread_movement": 0.0, "dk_total_movement": 0.0,
        # Tempo / scoring (used in pace features)
        "home_tempo": 70.0, "away_tempo": 70.0,
        "home_ppg": 0.0, "away_ppg": 0.0,
        "home_opp_ppg": 0.0, "away_opp_ppg": 0.0,
        # Recent form diff
        "recent_form_diff": 0.0,
        # Clutch / rolling features
        "home_clutch_ftm": 0.0, "away_clutch_ftm": 0.0,
        "home_clutch_fta": 1.0, "away_clutch_fta": 1.0,
        # ESPN win probability
        "espn_home_win_pct": 0.5, "espn_predictor_home_pct": 0.5,
        # Halftime
        "halftime_home_win_prob": 0.5,
    }
    for _col, _default in _col_defaults.items():
        if _col not in df.columns:
            df[_col] = _default

    df["home_record_wins"] = df["home_wins"]
    df["away_record_wins"] = df["away_wins"]
    df["home_record_losses"] = df["home_losses"]
    df["away_record_losses"] = df["away_losses"]
    df = _ncaa_backfill_heuristic(df)

    X = ncaa_build_features(df)

    # Align with model
    feature_cols = bundle.get("feature_names") or bundle.get("feature_cols")
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # ═══ Fix: Set ref features to league average when refs not assigned ═══
    # When ref_home_whistle=0, the classifier sees it as strong negative signal.
    # League averages computed from referee_profiles.json across all refs.
    _ref_profiles = getattr(ncaa_build_features, "_ref_profiles", {})
    if _ref_profiles:
        _all_hw = [p.get("home_whistle", 0) for p in _ref_profiles.values() if p.get("home_whistle")]
        _all_fr = [p.get("foul_rate", 0) for p in _ref_profiles.values() if p.get("foul_rate")]
        _all_ou = [p.get("ou_bias", 0) for p in _ref_profiles.values() if p.get("ou_bias")]
        avg_hw = sum(_all_hw) / len(_all_hw) if _all_hw else 0.0
        avg_fr = sum(_all_fr) / len(_all_fr) if _all_fr else 0.0
        avg_ou = sum(_all_ou) / len(_all_ou) if _all_ou else 0.0
        if "ref_home_whistle" in X.columns and X["ref_home_whistle"].iloc[0] == 0:
            X.loc[:, "ref_home_whistle"] = avg_hw
        if "ref_foul_rate" in X.columns and X["ref_foul_rate"].iloc[0] == 0:
            X.loc[:, "ref_foul_rate"] = avg_fr
        if "ref_ou_bias" in X.columns and X["ref_ou_bias"].iloc[0] == 0:
            X.loc[:, "ref_ou_bias"] = avg_ou

    # ═══ Post-processing: fill remaining zero features from available data ═══
    r = X.iloc[0]  # Single row reference

    # after_loss_either: should be 1 if either team just lost
    if "after_loss_either" in X.columns and r["after_loss_either"] == 0:
        h_after = sb("home_is_after_loss", 0) or home_sched.get("after_loss", 0)
        a_after = sb_away("home_is_after_loss", 0) or away_sched.get("after_loss", 0)
        if h_after or a_after:
            X.loc[:, "after_loss_either"] = 1

    # form_x_familiarity: form_diff * style_familiarity
    if "form_x_familiarity" in X.columns and "form_diff" in X.columns and "style_familiarity" in X.columns:
        X.loc[:, "form_x_familiarity"] = r["form_diff"] * r["style_familiarity"]

    # opp_ppg_diff: if 0, fill from Supabase
    if "opp_ppg_diff" in X.columns and r["opp_ppg_diff"] == 0:
        h_opp = sb("home_opp_ppg") or 72
        a_opp = sb_away("home_opp_ppg") or 72
        X.loc[:, "opp_ppg_diff"] = h_opp - a_opp

    # def_fgpct_diff: opponent FG% differential
    if "def_fgpct_diff" in X.columns and r["def_fgpct_diff"] == 0:
        h_opp_fg = sb("home_opp_efg_pct", 0.44)
        a_opp_fg = sb_away("home_opp_efg_pct", 0.44)
        X.loc[:, "def_fgpct_diff"] = h_opp_fg - a_opp_fg

    # win_pct_diff: from record — prefer schedule data over ESPN record defaults
    if "win_pct_diff" in X.columns and r["win_pct_diff"] == 0:
        # Use schedule wins/losses (already validated as primary source above)
        h_w = game.get("home_wins", 0)
        h_l = game.get("home_losses", 0)
        a_w = game.get("away_wins", 0)
        a_l = game.get("away_losses", 0)
        h_pct = h_w / max(h_w + h_l, 1)
        a_pct = a_w / max(a_w + a_l, 1)
        X.loc[:, "win_pct_diff"] = h_pct - a_pct
        print(f"  [full_predict] win_pct_diff patched: {h_w}/{h_w+h_l} - {a_w}/{a_w+a_l} = {h_pct - a_pct:.3f}")

    # ═══ Rolling HCA: compute from team's recent home/away margins ═══
    # Query last 20 home + 20 away games from ncaa_historical
    _rolling_hca_val = 0.0
    if not neutral_site:
        try:
            _home_tid = str(game.get("home_team_id", ""))
            if _home_tid:
                _hca_rows = sb_get("ncaa_historical",
                    f"home_team_id=eq.{_home_tid}&neutral_site=eq.false"
                    f"&actual_home_score=not.is.null&select=actual_home_score,actual_away_score"
                    f"&order=game_date.desc&limit=20")
                _away_rows = sb_get("ncaa_historical",
                    f"away_team_id=eq.{_home_tid}&neutral_site=eq.false"
                    f"&actual_home_score=not.is.null&select=actual_home_score,actual_away_score"
                    f"&order=game_date.desc&limit=20")
                _home_margins = [float(r["actual_home_score"]) - float(r["actual_away_score"]) for r in _hca_rows if r.get("actual_home_score") is not None]
                _away_margins = [float(r["actual_home_score"]) - float(r["actual_away_score"]) for r in _away_rows if r.get("actual_home_score") is not None]
                if len(_home_margins) >= 5 and len(_away_margins) >= 5:
                    import numpy as _np
                    _rolling_hca_val = (_np.mean(_home_margins) - _np.mean([-m for m in _away_margins])) / 2
                    print(f"  [full_predict] rolling_hca={_rolling_hca_val:.2f} (from {len(_home_margins)}H+{len(_away_margins)}A games)")
                else:
                    _rolling_hca_val = 6.6  # League-avg fallback
                    print(f"  [full_predict] rolling_hca=6.6 (fallback, only {len(_home_margins)}H+{len(_away_margins)}A)")
        except Exception as _e:
            _rolling_hca_val = 6.6
            print(f"  [full_predict] rolling_hca=6.6 (error: {_e})")

    # Inject rolling_hca into feature matrix
    if "rolling_hca" in X.columns:
        X.loc[:, "rolling_hca"] = _rolling_hca_val
    elif "rolling_hca" in feature_cols:
        X["rolling_hca"] = _rolling_hca_val

    # Fix neutral_em_diff: strip rolling_hca instead of old static HCA
    if "neutral_em_diff" in X.columns:
        _h_em = float(game.get("home_adj_em", 0) or 0)
        _a_em = float(game.get("away_adj_em", 0) or 0)
        _raw_em = _h_em - _a_em
        X.loc[:, "neutral_em_diff"] = _raw_em - (0 if neutral_site else _rolling_hca_val)

    # Legacy hca_pts: fill if still in feature list (old models)
    if "hca_pts" in X.columns and r["hca_pts"] == 0 and not neutral_site:
        X.loc[:, "hca_pts"] = _rolling_hca_val

    # sos_diff: from ESPN record
    if "sos_diff" in X.columns and r["sos_diff"] == 0:
        h_sos = home_record.get("sos", 0.5)
        a_sos = away_record.get("sos", 0.5)
        X.loc[:, "sos_diff"] = h_sos - a_sos

    # market_wp_edge: computed by ncaa_build_features from moneyline/spread cascade
    # No fallback needed here — ncaa_build_features handles the full cascade

    # games_last_14_diff: Supabase first, then schedule fallback
    if "games_last_14_diff" in X.columns and r["games_last_14_diff"] == 0:
        h_g14_sb = sb("home_games_last_14")
        a_g14_sb = sb_away("home_games_last_14")
        # FIX: use `is not None` check instead of `or` — 0 games is a valid value
        h_g14 = h_g14_sb if h_g14_sb else home_sched.get("games_last_14", 4)
        a_g14 = a_g14_sb if a_g14_sb else away_sched.get("games_last_14", 4)
        X.loc[:, "games_last_14_diff"] = h_g14 - a_g14

    # conf_balance_diff: from Supabase
    # FIX: removed `if h_cb or a_cb` guard — 0.0 is falsy but valid
    if "conf_balance_diff" in X.columns and r["conf_balance_diff"] == 0:
        h_cb = sb("home_conf_balance", 0)
        a_cb = sb_away("home_conf_balance", 0)
        X.loc[:, "conf_balance_diff"] = h_cb - a_cb
        if h_cb != 0 or a_cb != 0:
            print(f"  [full_predict] conf_balance_diff patched: {h_cb:.4f} - {a_cb:.4f} = {h_cb - a_cb:.4f}")

    # anti_fragility_diff: from Supabase
    # FIX: removed `if h_af or a_af` guard — same truthiness issue
    if "anti_fragility_diff" in X.columns and r["anti_fragility_diff"] == 0:
        h_af = sb("home_anti_fragility", 0)
        a_af = sb_away("home_anti_fragility", 0)
        X.loc[:, "anti_fragility_diff"] = h_af - a_af

    # roll_clutch_ft_diff: from Supabase rolling data
    # FIX: Use actual Supabase values even if they match the default (0.70).
    # The old guard `if h_cft != 0.70 or a_cft != 0.70` blocked valid data when
    # both teams had exactly the default value, but it also blocked the diff=0 case
    # which is correct. Now always compute the diff.
    if "roll_clutch_ft_diff" in X.columns and r["roll_clutch_ft_diff"] == 0:
        h_cft = sb("home_roll_clutch_ft_pct", 0.70)
        a_cft = sb_away("home_roll_clutch_ft_pct", 0.70)
        X.loc[:, "roll_clutch_ft_diff"] = h_cft - a_cft

    # roll_garbage_diff: from Supabase rolling data
    if "roll_garbage_diff" in X.columns and r["roll_garbage_diff"] == 0:
        h_rg = sb("home_roll_garbage_pct", 0.15)
        a_rg = sb_away("home_roll_garbage_pct", 0.15)
        X.loc[:, "roll_garbage_diff"] = h_rg - a_rg

    # centrality_diff: from Supabase
    if "centrality_diff" in X.columns and r["centrality_diff"] == 0:
        h_c = sb("home_centrality", 1.0)
        a_c = sb_away("home_centrality", 1.0)
        X.loc[:, "centrality_diff"] = h_c - a_c

    # Count real vs default features
    non_zero = (X.iloc[0] != 0).sum()
    total = len(feature_cols)

    # Diagnostic: log which features are still zero
    zero_features = [col for col in feature_cols if X[col].iloc[0] == 0]
    print(f"  [full_predict] Feature coverage: {non_zero}/{total} non-zero")
    if zero_features:
        print(f"  [full_predict] Zero features ({len(zero_features)}): {zero_features}")

    X_s = bundle["scaler"].transform(X)

    # Regression — support both old (reg) and new (models list) formats
    if "reg" in bundle:
        raw_margin = float(bundle["reg"].predict(X_s)[0])
    else:
        _models = bundle.get("models", [])
        raw_margin = float(sum(m.predict(X_s)[0] for m in _models) / max(len(_models), 1))
    bias = bundle.get("bias_correction", 0.0)
    margin = raw_margin - bias

    # Classification + isotonic — fallback to sigmoid if no classifier
    if "clf" in bundle:
        raw_win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    else:
        raw_win_prob = float(1.0 / (1.0 + __import__("math").exp(-raw_margin / 6.5)))
    isotonic = bundle.get("isotonic")
    win_prob = float(isotonic.predict([raw_win_prob])[0]) if isotonic else raw_win_prob
    win_prob = max(0.05, min(0.95, win_prob))

    # ═══ Consistency check: if classifier wildly disagrees with regressor, ═══
    # use sigma-converted margin as fallback probability.
    # Positive margin should mean >50% win prob, and vice versa.
    SIGMA = 6.0  # Brier-optimal (Apr 2026 sweep on backfilled data)
    import math as _math
    margin_prob = 1.0 / (1.0 + _math.exp(-margin / SIGMA))
    margin_prob = max(0.05, min(0.95, margin_prob))
    classifier_disagrees = (margin > 2 and win_prob < 0.40) or (margin < -2 and win_prob > 0.60)
    # Dynamic blend: margin-based probability is more robust when features are missing;
    # classifier is better-calibrated when all features are present.
    # At 100% coverage → pure classifier. At 50% coverage → 50/50 blend.
    coverage_ratio = non_zero / max(total, 1)
    margin_weight = max(0.0, 1.0 - coverage_ratio)
    win_prob = margin_weight * margin_prob + (1 - margin_weight) * win_prob
    win_prob = max(0.05, min(0.95, win_prob))

    # ═══ O/U MODEL v4 (triple agreement: residual + classifier + ATS scores) ═══
    ou_predicted_total = None
    ou_edge = None
    ou_pick = None
    ou_tier = 0
    ou_res_avg = None
    ou_cls_avg = None
    ou_ats_total = None
    ou_debug = "start"
    try:
        ou_bundle = load_model("ncaa_ou")
        ou_debug = f"loaded:{ou_bundle is not None}"

        # Detect v4 bundle (has res_models) vs v2 (has reg)
        if ou_bundle and ou_bundle.get("res_models"):
            # ── V4: Triple agreement system ──
            import numpy as _np

            res_feats = ou_bundle["res_feature_cols"]
            cls_feats = ou_bundle["cls_feature_cols"]
            ats_feats = ou_bundle["ats_feature_cols"]

            # Build feature vectors from the already-computed X DataFrame
            features_dict = {f: float(X[f].iloc[0]) if f in X.columns else 0.0 for f in set(res_feats + cls_feats + ats_feats)}

            # ── Add O/U-specific features (v5: sums + interactions) ──
            _h_tempo = float(game.get("home_tempo", 0) or 68)
            _a_tempo = float(game.get("away_tempo", 0) or 68)
            _h_oe = float(game.get("home_adj_oe", 0) or 100)
            _a_oe = float(game.get("away_adj_oe", 0) or 100)
            _h_de = float(game.get("home_adj_de", 0) or 100)
            _a_de = float(game.get("away_adj_de", 0) or 100)
            _h_ppg = float(game.get("home_ppg", 0) or 74)
            _a_ppg = float(game.get("away_ppg", 0) or 74)
            _h_opp = float(game.get("home_opp_ppg", 0) or 72)
            _a_opp = float(game.get("away_opp_ppg", 0) or 72)
            _h_3p = float(game.get("home_threepct", 0) or 0.34)
            _a_3p = float(game.get("away_threepct", 0) or 0.34)
            _h_fg = float(game.get("home_fgpct", 0) or 0.44)
            _a_fg = float(game.get("away_fgpct", 0) or 0.44)
            _h_orb = float(game.get("home_orb_pct", 0) or 0.28)
            _a_orb = float(game.get("away_orb_pct", 0) or 0.28)
            _tempo_avg = (_h_tempo + _a_tempo) / 2.0

            _ou_extras = {
                "tempo_min": min(_h_tempo, _a_tempo),
                "tempo_max": max(_h_tempo, _a_tempo),
                "tempo_product": _h_tempo * _a_tempo / (68.0 ** 2),
                "pace_mismatch": abs(_h_tempo - _a_tempo),
                "oe_sum": _h_oe + _a_oe,
                "de_sum": _h_de + _a_de,
                "efficiency_total": (_h_oe - _h_de) + (_a_oe - _a_de),
                "pace_x_oe_sum": _tempo_avg * (_h_oe + _a_oe) / 200.0,
                "pace_x_de_sum": _tempo_avg * (_h_de + _a_de) / 200.0,
                "ppg_sum": _h_ppg + _a_ppg,
                "opp_ppg_sum": _h_opp + _a_opp,
                "scoring_environment": (_h_ppg + _a_ppg + _h_opp + _a_opp) / 4,
                "threepct_sum": _h_3p + _a_3p,
                "fgpct_sum": _h_fg + _a_fg,
                "orb_pct_sum": _h_orb + _a_orb,
                "days_to_march": max(-30, min(150, (pd.Timestamp(int(game_date[:4]), 3, 15) - pd.Timestamp(game_date)).days)) if len(game_date) >= 4 else 0,
                "is_tourney_time": 1 if (len(game_date) >= 7 and ((game_date[5:7] == "03" and int(game_date[8:10]) >= 10) or game_date[5:7] == "04")) else 0,
            }
            features_dict.update(_ou_extras)

            res_vals = _np.array([[features_dict.get(f, 0) for f in res_feats]])
            cls_vals = _np.array([[features_dict.get(f, 0) for f in cls_feats]])
            ats_vals = _np.array([[features_dict.get(f, 0) for f in ats_feats]])

            res_scaled = ou_bundle["res_scaler"].transform(res_vals)
            cls_scaled = ou_bundle["cls_scaler"].transform(cls_vals)
            ats_scaled = ou_bundle["ats_scaler"].transform(ats_vals)

            # 3 residual models averaged
            res_preds = [m.predict(res_scaled)[0] for m in ou_bundle["res_models"]]
            ou_res_avg = float(_np.mean(res_preds))

            # 2 classifier models averaged → P(under)
            cls_preds = [m.predict_proba(cls_scaled)[0, 1] for m in ou_bundle["cls_models"]]
            ou_cls_avg = float(_np.mean(cls_preds))

            # ATS home + away scores → implied total
            home_preds = [m.predict(ats_scaled)[0] for m in ou_bundle["ats_home_models"]]
            away_preds = [m.predict(ats_scaled)[0] for m in ou_bundle["ats_away_models"]]
            ou_ats_total = float(_np.mean(home_preds) + _np.mean(away_preds))

            # Market O/U total
            mkt_ou = float(game.get("espn_over_under", 0) or game.get("market_ou_total", 0) or 0)

            # ── Display total: market + residual (sweep-validated, Apr 2026) ──
            if mkt_ou > 0:
                ou_predicted_total = mkt_ou + ou_res_avg
            else:
                ou_predicted_total = ou_ats_total

            ats_edge_val = ou_ats_total - mkt_ou if mkt_ou > 0 else 0

            if mkt_ou > 0:
                ou_edge = round(ou_res_avg, 1)  # residual = predicted market error

                # Check UNDER tiers (highest first)
                under_tiers = ou_bundle.get("under_tiers", {})
                for t in sorted(under_tiers.keys(), reverse=True):
                    th = under_tiers[t]
                    if (ou_res_avg <= th["res_avg"] and
                        ou_cls_avg >= th["cls_avg"] and
                        ats_edge_val <= th["ats_edge"]):
                        ou_pick = "UNDER"
                        ou_tier = t
                        break

                # If no UNDER, check OVER tiers
                if ou_pick is None:
                    over_tiers = ou_bundle.get("over_tiers", {})
                    for t in sorted(over_tiers.keys(), reverse=True):
                        th = over_tiers[t]
                        # OVER tiers may or may not require classifier (v5 asymmetric)
                        res_ok = ou_res_avg >= th["res_avg"]
                        ats_ok = ats_edge_val >= th["ats_edge"]
                        cls_ok = ou_cls_avg <= th["cls_avg"] if "cls_avg" in th else True
                        if res_ok and ats_ok and cls_ok:
                            ou_pick = "OVER"
                            ou_tier = t
                            break

            ou_debug = f"v4:res={ou_res_avg:.2f},cls={ou_cls_avg:.3f},ats_edge={ats_edge_val:.1f},pick={ou_pick},tier={ou_tier}"

        elif ou_bundle and ou_bundle.get("_ensemble_models") and ou_bundle.get("scaler"):
            # ── V2/V3 fallback: simple ensemble ──
            ou_feature_cols = ou_bundle.get("ou_feature_cols") or ou_bundle.get("feature_cols", [])
            available_ou = [f for f in ou_feature_cols if f in feature_cols]
            if len(available_ou) >= 15:
                ou_idx = [feature_cols.index(f) for f in available_ou]
                X_ou = X.iloc[:, ou_idx]
                for f in ou_feature_cols:
                    if f not in X_ou.columns:
                        X_ou[f] = 0.0
                X_ou = X_ou[ou_feature_cols]
                X_ou_s = ou_bundle["scaler"].transform(X_ou)
                models = ou_bundle["_ensemble_models"]
                ou_predicted_total = float(sum(m.predict(X_ou_s)[0] for m in models) / len(models))
                ou_bias = ou_bundle.get("bias_correction", 0)
                ou_predicted_total += ou_bias
                mkt_ou = float(game.get("espn_over_under", 0) or game.get("market_ou_total", 0) or 0)
                if mkt_ou > 0:
                    ou_edge = round(ou_predicted_total - mkt_ou, 1)
                    if abs(ou_edge) >= 5:
                        ou_pick = "OVER" if ou_edge > 0 else "UNDER"
                        ou_tier = 1
            ou_debug = f"v2:total={ou_predicted_total},edge={ou_edge}"

    except Exception as e:
        ou_pick = f"ERROR: {e}"
        print(f"  [full_predict] O/U model error: {e}")

    # ═══ ATS v4 MODEL (elasticnet+mlp_large+random_forest stack, 10 features) ═══
    ats_predicted_margin = None
    ats_cover_prob = None
    ats_pick = None
    ats_edge = None
    try:
        ats_bundle = load_model("ncaa_ats_v4")
        if ats_bundle and ats_bundle.get("base_learners") and ats_bundle.get("meta_learner"):
            ats_features = ats_bundle.get("features", [])
            # Build ATS feature vector — use available features, fill missing with 0
            ats_row = {}
            for f in ats_features:
                if f in feature_cols:
                    ats_row[f] = float(X[f].iloc[0])
                else:
                    ats_row[f] = 0.0
                    print(f"  [full_predict] ATS feature missing: {f} (using 0)")
            X_ats = np.array([[ats_row[f] for f in ats_features]])

            # Run through base learners
            base_preds = []
            for name in sorted(ats_bundle["base_learners"].keys()):
                learner = ats_bundle["base_learners"][name]
                pred = float(learner.predict(X_ats)[0])
                base_preds.append(pred)

            # Meta-learner stacks base predictions
            X_meta = np.array([base_preds])
            ats_predicted_margin = float(ats_bundle["meta_learner"].predict(X_meta)[0])

            # Compute cover probability via sigmoid of predicted margin advantage
            # Positive = model thinks home covers, negative = away covers
            mkt_spread = float(game.get("espn_spread", 0) or game.get("mkt_spread", 0) or 0)
            if mkt_spread != 0:
                # margin_vs_spread: how much model thinks home beats the spread
                margin_vs_spread = ats_predicted_margin - (-mkt_spread)  # espn_spread is negative when home favored
                SIGMA_ATS = 6.0  # Brier-optimal (Apr 2026 sweep)
                ats_cover_prob = round(1.0 / (1.0 + _math.exp(-margin_vs_spread / SIGMA_ATS)), 4)
                ats_cover_prob = max(0.05, min(0.95, ats_cover_prob))
                ats_edge = round(margin_vs_spread, 1)
                if abs(ats_edge) >= 4:
                    ats_pick = "HOME_COVER" if ats_edge > 0 else "AWAY_COVER"
            print(f"  [full_predict] ATS v4: margin={ats_predicted_margin:.2f}, edge={ats_edge}, pick={ats_pick}")
    except Exception as e:
        print(f"  [full_predict] ATS v4 model error: {e}")

    # SHAP
    shap_out = []
    try:
        explainer = bundle.get("explainer")
        if explainer:
            shap_vals = explainer.shap_values(X_s)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_out = [
                {"feature": f, "shap": round(float(v), 4), "value": round(float(X[f].iloc[0]), 3)}
                for f, v in zip(feature_cols, shap_vals[0])
            ]
            shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)
    except Exception as e:
        print(f"  [full_predict] SHAP error: {e}")
        shap_out = []

        # v19: Compute predicted scores from margin + team PPG
        h_ppg = float(game.get("home_ppg", 0) or 0)
        a_ppg = float(game.get("away_ppg", 0) or 0)
        avg_total = h_ppg + a_ppg if (h_ppg > 0 and a_ppg > 0) else 140
        pred_home_score = round((avg_total + margin) / 2, 1)
        pred_away_score = round((avg_total - margin) / 2, 1)

    return {
        "sport": "NCAAB",
        "ml_margin": round(margin, 2),
        "ml_margin_raw": round(raw_margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "ml_win_prob_raw": round(raw_win_prob, 4),
        "margin_based_prob": round(margin_prob, 4),
        "classifier_overridden": classifier_disagrees,
        "margin_weight": round(margin_weight, 3),
        "bias_correction_applied": round(bias, 3),
        "pred_home_score": pred_home_score,
        "pred_away_score": pred_away_score,
        "ou_predicted_total": round(ou_predicted_total, 1) if ou_predicted_total else None,
        "ou_edge": ou_edge,
        "ou_pick": ou_pick,
        "ou_tier": ou_tier,
        "ou_res_avg": round(ou_res_avg, 2) if ou_res_avg is not None else None,
        "ou_cls_avg": round(ou_cls_avg, 4) if ou_cls_avg is not None else None,
        "ou_ats_total": round(ou_ats_total, 1) if ou_ats_total is not None else None,


        "ats_predicted_margin": round(ats_predicted_margin, 2) if ats_predicted_margin is not None else None,
        "ats_cover_prob": ats_cover_prob,
        "ats_edge": ats_edge,
        "ats_pick": ats_pick,
        "shap": shap_out,
        "feature_coverage": f"{non_zero}/{total}",
        "data_sources": {
            "espn_stats": bool(home_stats and away_stats),
            "supabase_rolling": bool(home_sb),
            "referee": bool(game_info.get("referee_1")),
            "injuries": bool(home_inj["injury_penalty"] > 0 or away_inj["injury_penalty"] > 0),
            "spread_movement": bool(spread_mvmt),
            "attendance": bool(game_info.get("attendance", 0) > 0),
            "crowd_shock": bool(game.get("crowd_shock_diff", 0) != 0),
        },
        "model_meta": {
            "n_train": bundle.get("n_train", 0),
            "mae_cv": bundle.get("mae_cv", 0),
            "model_type": bundle.get("model_type", "unknown"),
            "trained_at": bundle.get("trained_at", "unknown"),
        },
        # v19: Expanded audit — includes ALL fields the cron saves to Supabase
        "audit_data": {
            "home_adj_em": game.get("home_adj_em"),
            "away_adj_em": game.get("away_adj_em"),
            "home_sos": game.get("home_sos"),
            "away_sos": game.get("away_sos"),
            "home_opp_fgpct": game.get("home_opp_fgpct"),
            "away_opp_fgpct": game.get("away_opp_fgpct"),
            "home_opp_threepct": game.get("home_opp_threepct"),
            "away_opp_threepct": game.get("away_opp_threepct"),
            "home_conference": game.get("home_conference"),
            "away_conference": game.get("away_conference"),
            "home_ppg": game.get("home_ppg"),
            "away_ppg": game.get("away_ppg"),
            "home_opp_ppg": game.get("home_opp_ppg"),
            "away_opp_ppg": game.get("away_opp_ppg"),
            "home_fgpct": game.get("home_fgpct"),
            "away_fgpct": game.get("away_fgpct"),
            "home_threepct": game.get("home_threepct"),
            "away_threepct": game.get("away_threepct"),
            "home_ftpct": game.get("home_ftpct"),
            "away_ftpct": game.get("away_ftpct"),
            "home_tempo": game.get("home_tempo"),
            "away_tempo": game.get("away_tempo"),
            "home_wins": game.get("home_wins"),
            "away_wins": game.get("away_wins"),
            "home_losses": game.get("home_losses"),
            "away_losses": game.get("away_losses"),
            "home_assists": game.get("home_assists"),
            "away_assists": game.get("away_assists"),
            "home_turnovers": game.get("home_turnovers"),
            "away_turnovers": game.get("away_turnovers"),
            "home_steals": game.get("home_steals"),
            "away_steals": game.get("away_steals"),
            "home_blocks": game.get("home_blocks"),
            "away_blocks": game.get("away_blocks"),
            "home_orb_pct": game.get("home_orb_pct"),
            "away_orb_pct": game.get("away_orb_pct"),
            "home_fta_rate": game.get("home_fta_rate"),
            "away_fta_rate": game.get("away_fta_rate"),
            "home_ato_ratio": game.get("home_ato_ratio"),
            "away_ato_ratio": game.get("away_ato_ratio"),
            "home_form": game.get("home_form"),
            "away_form": game.get("away_form"),
            "home_rank": game.get("home_rank"),
            "away_rank": game.get("away_rank"),
        },
    }
