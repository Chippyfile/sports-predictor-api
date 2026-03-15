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
            "ppg": stat_map.get("avgPoints", 0),
            "opp_ppg": stat_map.get("avgPointsAllowed", stat_map.get("oppPoints", 0)),
            "fgpct": stat_map.get("fieldGoalPct", 0.44),
            "threepct": stat_map.get("threePointFieldGoalPct", 0.34),
            "ftpct": stat_map.get("freeThrowPct", 0.72),
            "assists": stat_map.get("avgAssists", 14),
            "turnovers": stat_map.get("avgTurnovers", 12),
            "tempo": stat_map.get("avgPossessions", 68),
            "orb_pct": stat_map.get("offReboundPct", 0.28),
            "steals": stat_map.get("avgSteals", 7),
            "blocks": stat_map.get("avgBlocks", 3.5),
        }

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

    # Get most recent game for this team from ncaa_historical
    try:
        # Try by team_id first, then team_name
        rows = sb_get(
            "ncaa_historical",
            f"home_team_id=eq.{team_id}&order=game_date.desc&limit=1&select="
            "home_elo,home_pyth_residual,home_adj_oe,home_adj_de,"
            "home_efg_pct,home_twopt_pct,home_three_rate,home_assist_rate,home_drb_pct,home_ppp,"
            "home_opp_efg_pct,home_opp_to_rate,home_opp_fta_rate,home_opp_orb_pct,"
            "home_luck,home_consistency,home_margin_trend,home_margin_accel,"
            "home_opp_adj_form,home_wl_momentum,home_recovery_idx,home_is_after_loss,"
            "home_ceiling,home_floor,home_margin_skew,home_scoring_entropy,home_bimodal,"
            "home_def_stability,home_opp_suppression,home_def_versatility,"
            "home_steal_foul_ratio,home_block_foul_ratio,home_transition_dep,"
            "home_paint_pts,home_fastbreak_pts,home_fatigue_load,home_streak,"
            "home_season_pct,home_regression_pressure,home_info_gain,home_overreaction,"
            "home_scoring_source_entropy,home_ft_dependency,home_three_value,"
            "home_concentration,home_to_conversion,home_three_divergence,"
            "home_ppp_divergence,home_pace_adj_margin,home_pit_sos,"
            "home_scoring_var,home_score_kurtosis,home_clutch_ratio,home_garbage_adj_ppp,"
            "home_days_since_loss,home_games_since_blowout_loss,home_games_last_14,"
            "home_rest_effect,home_momentum_halflife,home_win_aging,home_centrality,"
            "home_dow_effect,home_conf_balance,home_eff_vol_ratio,"
            "home_home_margin,home_away_margin,"
            "home_run_vulnerability,home_anti_fragility,home_sos_trajectory,"
            "home_margin_autocorr,home_blowout_asym,home_clutch_over_exp,home_ft_pressure,"
            "home_fg_divergence,home_def_improvement,home_rhythm_disruption,"
            "home_fouls,home_close_win_rate,home_games_last_7,"
            "home_pts_off_to,"
            "home_roll_star1_share,home_roll_top3_share,home_roll_bench_share,"
            "home_roll_bench_pts,home_roll_largest_run,home_roll_drought_rate,"
            "home_roll_lead_changes,home_roll_time_with_lead_pct,"
            "home_roll_players_used,home_roll_hhi,home_roll_minutes_hhi,"
            "home_roll_clutch_ft_pct,home_roll_garbage_pct,"
            "home_roll_ats_pct,home_roll_ats_n,home_roll_ats_margin"
        )

        if not rows:
            # Try as away team
            rows = sb_get(
                "ncaa_historical",
                f"away_team_id=eq.{team_id}&order=game_date.desc&limit=1&select="
                "away_elo,away_pyth_residual,away_adj_oe,away_adj_de,"
                "away_efg_pct,away_twopt_pct,away_three_rate,away_assist_rate,away_drb_pct,away_ppp,"
                "away_opp_efg_pct,away_opp_to_rate,away_opp_fta_rate,away_opp_orb_pct,"
                "away_luck,away_consistency,away_margin_trend,away_margin_accel,"
                "away_opp_adj_form,away_wl_momentum,away_recovery_idx,away_is_after_loss,"
                "away_ceiling,away_floor,away_margin_skew,away_scoring_entropy,away_bimodal,"
                "away_def_stability,away_opp_suppression,away_def_versatility,"
                "away_steal_foul_ratio,away_block_foul_ratio,away_transition_dep,"
                "away_paint_pts,away_fastbreak_pts,away_fatigue_load,away_streak,"
                "away_season_pct,away_regression_pressure,away_info_gain,away_overreaction,"
                "away_scoring_source_entropy,away_ft_dependency,away_three_value,"
                "away_concentration,away_to_conversion,away_three_divergence,"
                "away_ppp_divergence,away_pace_adj_margin,away_pit_sos,"
                "away_scoring_var,away_score_kurtosis,away_clutch_ratio,away_garbage_adj_ppp,"
                "away_days_since_loss,away_games_since_blowout_loss,away_games_last_14,"
                "away_rest_effect,away_momentum_halflife,away_win_aging,away_centrality,"
                "away_dow_effect,away_conf_balance,away_eff_vol_ratio,"
                "away_home_margin,away_away_margin,"
                "away_run_vulnerability,away_anti_fragility,away_sos_trajectory,"
                "away_margin_autocorr,away_blowout_asym,away_clutch_over_exp,away_ft_pressure,"
                "away_fg_divergence,away_def_improvement,away_rhythm_disruption,"
                "away_fouls,away_close_win_rate,away_games_last_7,"
                "away_pts_off_to,"
                "away_roll_star1_share,away_roll_top3_share,away_roll_bench_share,"
                "away_roll_bench_pts,away_roll_largest_run,away_roll_drought_rate,"
                "away_roll_lead_changes,away_roll_time_with_lead_pct,"
                "away_roll_players_used,away_roll_hhi,away_roll_minutes_hhi,"
                "away_roll_clutch_ft_pct,away_roll_garbage_pct,"
                "away_roll_ats_pct,away_roll_ats_n,away_roll_ats_margin"
            )
            if rows:
                # Rename away_ to home_ for consistency
                row = rows[0]
                result = {}
                for k, v in row.items():
                    new_key = k.replace("away_", "home_") if k.startswith("away_") else k
                    result[new_key] = v
                _cache_set(cache_key, result)
                return result
            _cache_set(cache_key, {})
            return {}

        result = rows[0]
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

        # ESPN base stats
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

        # Advanced stats from Supabase
        "home_pyth_residual": sb("home_pyth_residual"), "away_pyth_residual": sb_away("home_pyth_residual"),
        "home_efg_pct": sb("home_efg_pct", 0.50), "away_efg_pct": sb_away("home_efg_pct", 0.50),
        "home_twopt_pct": sb("home_twopt_pct", 0.48), "away_twopt_pct": sb_away("home_twopt_pct", 0.48),
        "home_three_rate": sb("home_three_rate", 0.35), "away_three_rate": sb_away("home_three_rate", 0.35),
        "home_assist_rate": sb("home_assist_rate", 0.55), "away_assist_rate": sb_away("home_assist_rate", 0.55),
        "home_drb_pct": sb("home_drb_pct", 0.70), "away_drb_pct": sb_away("home_drb_pct", 0.70),
        "home_ppp": sb("home_ppp", 1.0), "away_ppp": sb_away("home_ppp", 1.0),
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
