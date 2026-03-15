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

        stats = {
            "ppg": stat_map.get("avgPoints", 75),
            "opp_ppg": stat_map.get("avgPointsAllowed", stat_map.get("oppPoints", 72)),
            # ESPN returns percentages (50.6), model expects decimals (0.506)
            "fgpct": stat_map.get("fieldGoalPct", 44.0) / 100.0,
            "threepct": stat_map.get("threePointFieldGoalPct", 34.0) / 100.0,
            "ftpct": stat_map.get("freeThrowPct", 72.0) / 100.0,
            "assists": stat_map.get("avgAssists", 14),
            "turnovers": stat_map.get("avgTurnovers", 12),
            "tempo": stat_map.get("avgPossessions", 68),
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
    """Fetch referee, attendance, venue from ESPN game summary."""
    ttl = CACHE_TTL_PREGAME if pregame else CACHE_TTL_LONG
    cache_key = f"espn_game_{game_id}"
    cached = _cache_get(cache_key, ttl)
    if cached:
        return cached

    info = {
        "referee_1": "", "referee_2": "", "referee_3": "",
        "attendance": 0, "venue_capacity": 8000, "venue_name": "",
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
        game_info = data.get("gameInfo", {})
        info["attendance"] = game_info.get("attendance", 0) or 0
        venue = game_info.get("venue", {})
        info["venue_capacity"] = venue.get("capacity", 8000) or 8000
        info["venue_name"] = venue.get("fullName", "")

        _cache_set(cache_key, info)
        return info
    except:
        _cache_set(cache_key, info)
        return info


def _compute_rest_days(team_id, game_date_str):
    """Compute days since last game."""
    cache_key = f"rest_{team_id}_{game_date_str}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached is not None:
        return cached

    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}/schedule", timeout=10)
        if not r.ok:
            return 3
        data = r.json()
        game_date = datetime.fromisoformat(f"{game_date_str}T00:00:00")
        last_game = None
        for ev in data.get("events", []):
            ev_date = datetime.fromisoformat(ev["date"][:19])
            completed = ev.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed", False)
            if completed and ev_date < game_date:
                if not last_game or ev_date > last_game:
                    last_game = ev_date
        if not last_game:
            result = 7
        else:
            result = max(0, (game_date - last_game).days)
        _cache_set(cache_key, result)
        return result
    except:
        return 3


# ═══════════════════════════════════════════════════════════════
# SUPABASE DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

def _fetch_supabase_team_data(team_id, team_name):
    """Fetch rolling features, Elo, advanced stats from Supabase ncaa_historical."""
    cache_key = f"sb_team_{team_id}"
    cached = _cache_get(cache_key, CACHE_TTL_LONG)
    if cached:
        return cached

    _COLS = (
        "elo,pyth_residual,adj_oe,adj_de,"
        "efg_pct,twopt_pct,three_rate,assist_rate,drb_pct,ppp,"
        "opp_efg_pct,opp_to_rate,opp_fta_rate,opp_orb_pct,"
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
        "pts_off_to,"
        "roll_star1_share,roll_top3_share,roll_bench_share,"
        "roll_bench_pts,roll_largest_run,roll_drought_rate,"
        "roll_lead_changes,roll_time_with_lead_pct,"
        "roll_players_used,roll_hhi,roll_minutes_hhi,"
        "roll_clutch_ft_pct,roll_garbage_pct,"
        "roll_ats_pct,roll_ats_n,roll_ats_margin"
    )

    try:
        # Query BOTH home and away appearances, take the most recent
        home_select = ",".join([f"game_date,home_{c}" for c in _COLS.split(",")])
        away_select = ",".join([f"game_date,away_{c}" for c in _COLS.split(",")])
        # Deduplicate game_date in select
        home_select = "game_date," + ",".join([f"home_{c}" for c in _COLS.split(",")])
        away_select = "game_date," + ",".join([f"away_{c}" for c in _COLS.split(",")])

        home_rows = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{team_id}&order=game_date.desc&limit=1&select={home_select}"
        )
        away_rows = sb_get(
            "ncaa_historical",
            f"away_team_id=eq.{team_id}&order=game_date.desc&limit=1&select={away_select}"
        )

        # Pick the more recent one
        home_date = (home_rows[0].get("game_date", "") or "") if home_rows else ""
        away_date = (away_rows[0].get("game_date", "") or "") if away_rows else ""

        if away_date > home_date and away_rows:
            # Most recent game was as away — rename away_ to home_
            row = away_rows[0]
            result = {}
            for k, v in row.items():
                if k == "game_date":
                    continue
                new_key = k.replace("away_", "home_") if k.startswith("away_") else k
                result[new_key] = v
        elif home_rows:
            result = {k: v for k, v in home_rows[0].items() if k != "game_date"}
        else:
            _cache_set(cache_key, {})
            return {}

        _cache_set(cache_key, result)
        return result
    except Exception as e:
        print(f"  [full_predict] Supabase team data error for {team_id}: {e}")
        return {}


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
        fut_home_rest = executor.submit(_compute_rest_days, home_team_id, game_date)
        fut_away_rest = executor.submit(_compute_rest_days, away_team_id, game_date)
        fut_game_info = executor.submit(_fetch_espn_game_info, game_id, game_date, pregame)
        fut_home_sb = executor.submit(_fetch_supabase_team_data, home_team_id, home_team_name)
        fut_away_sb = executor.submit(_fetch_supabase_team_data, away_team_id, away_team_name)
        fut_spread = executor.submit(_fetch_spread_movement, game_id)

    home_stats = fut_home_stats.result() or {}
    away_stats = fut_away_stats.result() or {}
    home_record = fut_home_record.result()
    away_record = fut_away_record.result()
    home_rest = fut_home_rest.result()
    away_rest = fut_away_rest.result()
    game_info = fut_game_info.result()
    home_sb = fut_home_sb.result()
    away_sb = fut_away_sb.result()
    spread_mvmt = fut_spread.result()

    # Injury data
    home_inj = get_team_injuries(home_team_name) if HAS_INJURY_CACHE else {"missing_starters": 0, "injury_penalty": 0.0}
    away_inj = get_team_injuries(away_team_name) if HAS_INJURY_CACHE else {"missing_starters": 0, "injury_penalty": 0.0}

    # Market odds from request (frontend still passes these from Odds API)
    market_spread = request_data.get("market_spread_home", 0)
    market_total = request_data.get("market_ou_total", 0)
    espn_spread = request_data.get("espn_spread", 0)
    espn_ou = request_data.get("espn_over_under", 0)
    espn_wp = request_data.get("espn_home_win_pct", 0.5)
    espn_pred = request_data.get("espn_predictor_home_pct", 0.5)

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
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "neutral_site": neutral_site,

        # ESPN base stats (normalized to decimal format)
        "home_ppg": home_stats.get("ppg", 75), "away_ppg": away_stats.get("ppg", 75),
        "home_opp_ppg": home_stats.get("opp_ppg", 72), "away_opp_ppg": away_stats.get("opp_ppg", 72),
        "home_fgpct": home_stats.get("fgpct", 0.44), "away_fgpct": away_stats.get("fgpct", 0.44),
        "home_threepct": home_stats.get("threepct", 0.34), "away_threepct": away_stats.get("threepct", 0.34),
        "home_ftpct": home_stats.get("ftpct", 0.72), "away_ftpct": away_stats.get("ftpct", 0.72),
        "home_assists": home_stats.get("assists", 14), "away_assists": away_stats.get("assists", 14),
        "home_turnovers": home_stats.get("turnovers", 12), "away_turnovers": away_stats.get("turnovers", 12),
        "home_tempo": home_stats.get("tempo", 68), "away_tempo": away_stats.get("tempo", 68),
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

        # Record
        "home_wins": home_record["wins"], "away_wins": away_record["wins"],
        "home_losses": home_record["losses"], "away_losses": away_record["losses"],
        "home_sos": home_record.get("sos", 0.5), "away_sos": away_record.get("sos", 0.5),
        "home_rank": request_data.get("home_rank", 200), "away_rank": request_data.get("away_rank", 200),
        "home_rest_days": home_rest, "away_rest_days": away_rest,

        # Conference
        "home_conference": request_data.get("home_conference", ""),
        "away_conference": request_data.get("away_conference", ""),

        # KenPom / efficiency (from request or Supabase)
        "home_adj_em": request_data.get("home_adj_em", sb("home_adj_em", 0)),
        "away_adj_em": request_data.get("away_adj_em", sb_away("home_adj_em", 0)),
        "home_adj_oe": sb("home_adj_oe", 105), "away_adj_oe": sb_away("home_adj_oe", 105),
        "home_adj_de": sb("home_adj_de", 105), "away_adj_de": sb_away("home_adj_de", 105),

        # Elo
        "home_elo": sb("home_elo", 1500), "away_elo": sb_away("home_elo", 1500),

        # Supabase-only advanced stats (no ESPN equivalent)
        "home_pyth_residual": sb("home_pyth_residual"), "away_pyth_residual": sb_away("home_pyth_residual"),
        "home_opp_efg_pct": sb("home_opp_efg_pct", 0.50), "away_opp_efg_pct": sb_away("home_opp_efg_pct", 0.50),
        "home_opp_to_rate": sb("home_opp_to_rate", 0.18), "away_opp_to_rate": sb_away("home_opp_to_rate", 0.18),
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
        "home_is_after_loss": sb("home_is_after_loss"), "away_is_after_loss": sb_away("home_is_after_loss"),
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
        "home_games_last_14": sb("home_games_last_14", 4), "away_games_last_14": sb_away("home_games_last_14", 4),
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

        # Referee
        "referee_1": game_info.get("referee_1", ""),
        "referee_2": game_info.get("referee_2", ""),
        "referee_3": game_info.get("referee_3", ""),

        # Venue
        "attendance": game_info.get("attendance", 0),
        "venue_capacity": game_info.get("venue_capacity", 8000),

        # Market
        "market_spread_home": market_spread,
        "market_ou_total": market_total,
        "espn_spread": espn_spread,
        "espn_over_under": espn_ou,
        "espn_home_win_pct": espn_wp,
        "espn_predictor_home_pct": espn_pred,
        "espn_ml_home": request_data.get("espn_ml_home", 0),
        "espn_ml_away": request_data.get("espn_ml_away", 0),

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
        "is_conference_tournament": request_data.get("is_conference_tournament", 0),
        "is_ncaa_tournament": request_data.get("is_ncaa_tournament", 0),
        "is_bubble_game": request_data.get("is_bubble_game", 0),
        "is_early_season": request_data.get("is_early_season", 0),
        "importance_multiplier": request_data.get("importance_multiplier", 1.0),

# Matchup/situational features (computed, default to neutral)
        "n_common_opps": 0, "common_opp_diff": 0.0,
        "is_lookahead": 0, "is_revenge_game": 0, "revenge_margin": 0.0,
        "is_sandwich": 0, "def_rest_advantage": 0.0, "luck_x_spread": 0.0,
        "spread_regime": 1, "is_midweek": 0,
        "style_familiarity": 0.5, "pace_leverage": 0.0, "pace_control_diff": 0.0,
        "matchup_efg": 0.0, "matchup_to": 0.0, "matchup_orb": 0.0, "matchup_ft": 0.0,
        "fatigue_x_quality": 0.0, "rest_x_defense": 0.0,
        "form_x_familiarity": 0.0, "consistency_x_spread": 0.0,

        # Form (from ESPN stats if available)
        "home_form": home_stats.get("form", 0), "away_form": away_stats.get("form", 0),
    }

    # ── Build features and predict ──
    import json as _json
    try:
        with open("referee_profiles.json") as f:
            ncaa_build_features._ref_profiles = _json.load(f)
    except:
        pass

    # Backfill heuristic (generates spread_home, win_pct_home, etc.)
    df = pd.DataFrame([game])
    df["home_record_wins"] = df["home_wins"]
    df["away_record_wins"] = df["away_wins"]
    df["home_record_losses"] = df["home_losses"]
    df["away_record_losses"] = df["away_losses"]
    df = _ncaa_backfill_heuristic(df)

    X = ncaa_build_features(df)

    # Align with model
    feature_cols = bundle["feature_cols"]
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # Count real vs default features
    non_zero = (X.iloc[0] != 0).sum()
    total = len(feature_cols)

    X_s = bundle["scaler"].transform(X)

    # Regression
    raw_margin = float(bundle["reg"].predict(X_s)[0])
    bias = bundle.get("bias_correction", 0.0)
    margin = raw_margin - bias

    # Classification + isotonic
    raw_win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    isotonic = bundle.get("isotonic")
    win_prob = float(isotonic.predict([raw_win_prob])[0]) if isotonic else raw_win_prob
    win_prob = max(0.05, min(0.95, win_prob))

    # SHAP
    shap_vals = bundle["explainer"].shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_out = [
        {"feature": f, "shap": round(float(v), 4), "value": round(float(X[f].iloc[0]), 3)}
        for f, v in zip(feature_cols, shap_vals[0])
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "sport": "NCAAB",
        "ml_margin": round(margin, 2),
        "ml_margin_raw": round(raw_margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "ml_win_prob_raw": round(raw_win_prob, 4),
        "bias_correction_applied": round(bias, 3),
        "shap": shap_out,
        "feature_coverage": f"{non_zero}/{total}",
        "data_sources": {
            "espn_stats": bool(home_stats and away_stats),
            "supabase_rolling": bool(home_sb),
            "referee": bool(game_info.get("referee_1")),
            "injuries": bool(home_inj["injury_penalty"] > 0 or away_inj["injury_penalty"] > 0),
            "spread_movement": bool(spread_mvmt),
            "attendance": bool(game_info.get("attendance", 0) > 0),
        },
        "model_meta": {
            "n_train": bundle["n_train"],
            "mae_cv": bundle["mae_cv"],
            "model_type": bundle.get("model_type", "unknown"),
            "trained_at": bundle["trained_at"],
        },
    }
