"""
mlb_full_predict.py — Server-side MLB prediction with full data fetching
═════════════════════════════════════════════════════════════════════════
Fetches all data server-side from MLB Stats API + Supabase.
No frontend dependency — can be called by cron or API.

Route: POST /predict/mlb/full
Input: {"game_id": 747653} or {"home_team":"ATL","away_team":"OAK","game_date":"2026-04-01"}

Data sources:
  - MLB Stats API: schedule, starters, team stats, standings
  - Supabase: mlb_team_rolling, mlb_ump_profiles
  - The Odds API: market spread, total, moneylines (via /api/odds proxy)
"""

import requests
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# v9 ATS model (CatBoost×0.8 + Lasso×0.2 with lineup features)
try:
    from mlb_ats_v9_serve import predict_mlb_ats_v9, compute_lineup_features, _fetch_batter_season_stats, fetch_pregame_lineups, fetch_ump_home_win_pct
    _HAS_V9 = True
except ImportError:
    _HAS_V9 = False
    print("  [mlb_full] mlb_ats_v9_serve not available — v9 ATS disabled")

MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 10

# ── Team abbreviation mapping (MLB API uses full names, we need abbrs) ──
_TEAM_ID_TO_ABBR = {
    108:"LAA",109:"ARI",110:"BAL",111:"BOS",112:"CHC",113:"CIN",114:"CLE",
    115:"COL",116:"DET",117:"HOU",118:"KC",119:"LAD",120:"WSH",121:"NYM",
    133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",138:"STL",139:"TB",
    140:"TEX",141:"TOR",142:"MIN",143:"PHI",144:"ATL",145:"CWS",146:"MIA",
    147:"NYY",158:"MIL",
}
_ABBR_TO_TEAM_ID = {v: k for k, v in _TEAM_ID_TO_ABBR.items()}

# ── Static park factors (matches frontend PARK_FACTORS) ──
_PARK_FACTORS = {
    108:1.02, 109:1.03, 110:0.95, 111:1.04, 112:1.04, 113:1.00, 114:0.97,
    115:1.16, 116:0.98, 117:0.99, 118:1.01, 119:1.00, 120:1.01, 121:1.03,
    133:0.99, 134:0.96, 135:0.95, 136:0.94, 137:0.96, 138:0.98, 139:0.96,
    140:1.01, 141:1.01, 142:0.98, 143:1.05, 144:1.00, 145:1.06, 146:0.97,
    147:1.03, 158:1.02,
}


def _mlb_get(endpoint, params=None, timeout=REQUEST_TIMEOUT):
    """Fetch from MLB Stats API."""
    try:
        url = f"{MLB_API}/{endpoint}"
        r = requests.get(url, params=params, timeout=timeout)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"  [mlb_full] API error {endpoint}: {e}")
    return None


def _fetch_schedule_game(game_id=None, home_team=None, away_team=None, game_date=None):
    """Fetch a specific game from the MLB schedule. Returns game dict or None."""
    if game_id:
        data = _mlb_get(f"schedule", {
            "sportId": 1, "gamePk": game_id,
            "hydrate": "probablePitcher,teams,venue,linescore,officials",
        })
    elif game_date:
        data = _mlb_get("schedule", {
            "sportId": 1, "date": game_date,
            "hydrate": "probablePitcher,teams,venue,linescore,officials",
        })
    else:
        return None

    if not data:
        return None

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append(g)

    if game_id:
        return games[0] if games else None

    # Match by team abbreviations
    if home_team and away_team:
        ht = home_team.upper().strip()
        at = away_team.upper().strip()
        for g in games:
            h_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
            a_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
            if _TEAM_ID_TO_ABBR.get(h_id) == ht and _TEAM_ID_TO_ABBR.get(a_id) == at:
                return g
    return games[0] if games else None


def _fetch_team_stats(team_id, season=None):
    """Fetch team hitting + pitching stats from MLB Stats API."""
    if not season:
        season = datetime.utcnow().year

    result = {"hitting": {}, "pitching": {}}

    # Hitting
    data = _mlb_get(f"teams/{team_id}/stats", {
        "stats": "season", "season": season, "group": "hitting",
    })
    if data:
        for split in data.get("stats", []):
            for s in split.get("splits", []):
                st = s.get("stat", {})
                result["hitting"] = {
                    "avg": float(st.get("avg", ".250").replace(".", "0.", 1) if isinstance(st.get("avg"), str) else st.get("avg", 0.250)),
                    "obp": float(st.get("obp", 0.320)),
                    "slg": float(st.get("slg", 0.400)),
                    "ops": float(st.get("ops", 0.720)),
                    "runs_per_game": float(st.get("runs", 0)) / max(1, float(st.get("gamesPlayed", 1))),
                    "games": int(st.get("gamesPlayed", 0)),
                }
                break

    # Pitching
    data = _mlb_get(f"teams/{team_id}/stats", {
        "stats": "season", "season": season, "group": "pitching",
    })
    if data:
        for split in data.get("stats", []):
            for s in split.get("splits", []):
                st = s.get("stat", {})
                ip = float(st.get("inningsPitched", "0").replace(".", "") or 0)
                # Convert innings pitched string (e.g., "45.1" means 45 1/3)
                try:
                    ip_parts = str(st.get("inningsPitched", "0")).split(".")
                    ip = int(ip_parts[0]) + (int(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
                except:
                    ip = 0
                result["pitching"] = {
                    "era": float(st.get("era", 4.25)),
                    "whip": float(st.get("whip", 1.30)),
                    "k9": float(st.get("strikeoutsPer9Inn", 8.5)),
                    "bb9": float(st.get("walksPer9Inn", 3.2)),
                    "ip": ip,
                    "games": int(st.get("gamesPlayed", 0)),
                    "hr": int(st.get("homeRuns", 0)),
                    "k": int(st.get("strikeOuts", 0)),
                    "bb": int(st.get("baseOnBalls", 0)),
                }
                break

    return result


def _fetch_pitcher_stats(pitcher_id, season=None):
    """Fetch individual pitcher season stats."""
    if not pitcher_id:
        return None
    if not season:
        season = datetime.utcnow().year

    data = _mlb_get(f"people/{pitcher_id}/stats", {
        "stats": "season", "season": season, "group": "pitching",
    })
    if not data:
        return None

    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            st = split.get("stat", {})
            try:
                ip_parts = str(st.get("inningsPitched", "0")).split(".")
                ip = int(ip_parts[0]) + (int(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
            except:
                ip = 0
            gs = max(1, int(st.get("gamesStarted", 1)))
            return {
                "era": float(st.get("era", 4.25)),
                "k9": float(st.get("strikeoutsPer9Inn", 8.5)),
                "bb9": float(st.get("walksPer9Inn", 3.2)),
                "whip": float(st.get("whip", 1.30)),
                "ip_per_start": round(ip / gs, 2) if gs > 0 else 5.5,
                "ip_total": ip,
                "games_started": gs,
                "hr": int(st.get("homeRuns", 0)),
                "k": int(st.get("strikeOuts", 0)),
                "bb": int(st.get("baseOnBalls", 0)),
            }
    return None


def _compute_woba(obp, slg):
    """Approximate wOBA from OBP and SLG (R²=0.993 vs FanGraphs)."""
    return 0.72 * obp + 0.48 * slg - 0.08


def _compute_fip(hr, bb, k, ip, lg_fip=4.10):
    """Compute FIP from counting stats."""
    if ip <= 0:
        return lg_fip
    cFIP = lg_fip - (13 * 1.0 + 3 * 3.2 - 2 * 8.5)  # approximate constant
    return max(1.5, min(7.0, (13 * hr + 3 * bb - 2 * k) / ip + cFIP))


def _extract_umpire(game):
    """Extract home plate umpire from schedule game data."""
    officials = game.get("officials", [])
    for off in officials:
        if off.get("officialType") == "Home Plate":
            return off.get("official", {}).get("fullName")
    return None


def _fetch_rest_days(team_id, game_date_str):
    """Compute days since team's last game."""
    try:
        dt = datetime.strptime(game_date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        data = _mlb_get("schedule", {
            "sportId": 1, "teamId": team_id,
            "startDate": start, "endDate": end,
        })
        if data:
            last_date = None
            for d in data.get("dates", []):
                for g in d.get("games", []):
                    gd = g.get("gameDate", "")[:10]
                    if gd and (not last_date or gd > last_date):
                        last_date = gd
            if last_date:
                delta = (dt - datetime.strptime(last_date, "%Y-%m-%d")).days
                return max(0, min(7, delta))
    except:
        pass
    return 1  # default: played yesterday


def predict_mlb_full(input_data):
    """
    Full server-side MLB prediction.
    Fetches team stats, starters, umpire, rolling features, odds.
    Returns margin + O/U predictions.
    """
    from sports.mlb import predict_mlb, predict_mlb_ou, SEASON_CONSTANTS, DEFAULT_CONSTANTS, _is_dome

    game_id = input_data.get("game_id")
    home_team = (input_data.get("home_team") or "").upper().strip()
    away_team = (input_data.get("away_team") or "").upper().strip()
    game_date = input_data.get("game_date") or datetime.utcnow().strftime("%Y-%m-%d")

    try:
        season = int(game_date[:4])
    except:
        season = datetime.utcnow().year
    sc = SEASON_CONSTANTS.get(season, DEFAULT_CONSTANTS)

    # ── Step 1: Fetch schedule game data ──
    game = _fetch_schedule_game(game_id=game_id, home_team=home_team,
                                 away_team=away_team, game_date=game_date)
    if not game:
        return {"error": f"Game not found: {home_team} vs {away_team} on {game_date}"}

    # Extract team IDs and abbreviations
    h_team = game.get("teams", {}).get("home", {}).get("team", {})
    a_team = game.get("teams", {}).get("away", {}).get("team", {})
    h_id = h_team.get("id")
    a_id = a_team.get("id")
    home_abbr = _TEAM_ID_TO_ABBR.get(h_id, home_team)
    away_abbr = _TEAM_ID_TO_ABBR.get(a_id, away_team)

    # Starters
    h_starter = game.get("teams", {}).get("home", {}).get("probablePitcher", {})
    a_starter = game.get("teams", {}).get("away", {}).get("probablePitcher", {})
    h_starter_id = h_starter.get("id")
    a_starter_id = a_starter.get("id")
    h_starter_name = h_starter.get("fullName", "TBD")
    a_starter_name = a_starter.get("fullName", "TBD")

    # Umpire
    ump_name = _extract_umpire(game)

    # Park factor
    park_factor = _PARK_FACTORS.get(h_id, 1.00)
    is_dome = _is_dome(home_abbr)

    print(f"  [mlb_full] {away_abbr}@{home_abbr} | SP: {a_starter_name} vs {h_starter_name} | Ump: {ump_name}")

    # ── Step 2: Parallel data fetching ──
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_fetch_team_stats, h_id, season): "home_stats",
            pool.submit(_fetch_team_stats, a_id, season): "away_stats",
            pool.submit(_fetch_pitcher_stats, h_starter_id, season): "home_sp",
            pool.submit(_fetch_pitcher_stats, a_starter_id, season): "away_sp",
            pool.submit(_fetch_rest_days, h_id, game_date): "home_rest",
            pool.submit(_fetch_rest_days, a_id, game_date): "away_rest",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"  [mlb_full] {key} fetch failed: {e}")
                results[key] = None

    home_stats = results.get("home_stats") or {"hitting": {}, "pitching": {}}
    away_stats = results.get("away_stats") or {"hitting": {}, "pitching": {}}
    home_sp = results.get("home_sp")
    away_sp = results.get("away_sp")
    home_rest = results.get("home_rest") or 1
    away_rest = results.get("away_rest") or 1

    # ── Step 3: Compute raw features ──
    h_hit = home_stats["hitting"]
    a_hit = away_stats["hitting"]
    h_pitch = home_stats["pitching"]
    a_pitch = away_stats["pitching"]

    home_woba = _compute_woba(h_hit.get("obp", 0.320), h_hit.get("slg", 0.400))
    away_woba = _compute_woba(a_hit.get("obp", 0.320), a_hit.get("slg", 0.400))

    home_team_era = h_pitch.get("era", sc["lg_fip"])
    away_team_era = a_pitch.get("era", sc["lg_fip"])

    # Starter FIP — compute from stats if available, else use ERA
    if home_sp and home_sp.get("ip_total", 0) > 0:
        home_sp_fip = _compute_fip(home_sp["hr"], home_sp["bb"], home_sp["k"],
                                     home_sp["ip_total"], sc["lg_fip"])
    elif home_sp:
        home_sp_fip = home_sp.get("era", sc["lg_fip"])
    else:
        home_sp_fip = home_team_era

    if away_sp and away_sp.get("ip_total", 0) > 0:
        away_sp_fip = _compute_fip(away_sp["hr"], away_sp["bb"], away_sp["k"],
                                     away_sp["ip_total"], sc["lg_fip"])
    elif away_sp:
        away_sp_fip = away_sp.get("era", sc["lg_fip"])
    else:
        away_sp_fip = away_team_era

    # Weather — dome parks get neutral defaults
    if is_dome:
        temp_f, wind_mph, wind_out_flag = 70.0, 0.0, 0.0
    else:
        temp_f = 70.0   # TODO: integrate weather API
        wind_mph = 5.0
        wind_out_flag = 0.0

    # ── Step 4: Build payload for predict_mlb ──
    payload = {
        "home_team": home_abbr,
        "away_team": away_abbr,
        "game_date": game_date,
        "home_woba": round(home_woba, 3),
        "away_woba": round(away_woba, 3),
        "home_sp_fip": round(home_sp_fip, 2),
        "away_sp_fip": round(away_sp_fip, 2),
        "home_fip": round(home_team_era, 2),
        "away_fip": round(away_team_era, 2),
        "home_bullpen_era": round(home_team_era, 2),  # team ERA as proxy
        "away_bullpen_era": round(away_team_era, 2),
        "park_factor": park_factor,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_out_flag": wind_out_flag,
        "home_k9": home_sp.get("k9", h_pitch.get("k9", 8.5)) if home_sp else h_pitch.get("k9", 8.5),
        "away_k9": away_sp.get("k9", a_pitch.get("k9", 8.5)) if away_sp else a_pitch.get("k9", 8.5),
        "home_bb9": home_sp.get("bb9", h_pitch.get("bb9", 3.2)) if home_sp else h_pitch.get("bb9", 3.2),
        "away_bb9": away_sp.get("bb9", a_pitch.get("bb9", 3.2)) if away_sp else a_pitch.get("bb9", 3.2),
        "home_sp_ip": home_sp.get("ip_per_start", 5.5) if home_sp else 5.5,
        "away_sp_ip": away_sp.get("ip_per_start", 5.5) if away_sp else 5.5,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "ump_name": ump_name,
        "pred_home_runs": 0,  # no heuristic in full mode
        "pred_away_runs": 0,
        # Market data — filled below if available
        "market_spread_home": 0,
        "market_ou_total": 0,
        # Starter IDs for O/U v2 (sp_form from game logs)
        "home_starter_id": h_starter_id,
        "away_starter_id": a_starter_id,
    }

    # ── Step 4b: Fetch market odds from ESPN scoreboard (needed for O/U v2 model) ──
    _ESPN_ALIAS = {"CHW":"CWS","WSN":"WSH","AZ":"ARI"}
    try:
        espn_sb_url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={game_date.replace('-','')}"
        espn_r = requests.get(espn_sb_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if espn_r.ok:
            for ev in espn_r.json().get("events", []):
                competitors = ev.get("competitions", [{}])[0].get("competitors", [])
                ev_home_raw = next((c.get("team", {}).get("abbreviation", "") for c in competitors if c.get("homeAway") == "home"), "")
                ev_home = _ESPN_ALIAS.get(ev_home_raw.upper(), ev_home_raw.upper())
                if ev_home == home_abbr.upper():
                    odds_list = ev.get("competitions", [{}])[0].get("odds", [])
                    if odds_list:
                        odds = odds_list[0]
                        if odds.get("overUnder") is not None:
                            payload["market_ou_total"] = odds["overUnder"]
                        # Moneylines
                        ml = odds.get("moneyline", {})
                        h_ml = ml.get("home", {}).get("close", {}).get("odds")
                        a_ml = ml.get("away", {}).get("close", {}).get("odds")
                        if h_ml:
                            payload["market_home_ml"] = int(h_ml.replace("+", ""))
                        if a_ml:
                            payload["market_away_ml"] = int(a_ml.replace("+", ""))
                        # Run line
                        ps = odds.get("pointSpread", {})
                        h_rl = ps.get("home", {}).get("close", {}).get("line")
                        if h_rl:
                            payload["market_spread_home"] = float(h_rl)
                    break
            if payload["market_ou_total"]:
                print(f"  [mlb_full] Market: O/U={payload['market_ou_total']}, RL={payload.get('market_spread_home')}, ML={payload.get('market_home_ml')}/{payload.get('market_away_ml')}")
            else:
                print(f"  [mlb_full] No ESPN odds found for {away_abbr}@{home_abbr}")
    except Exception as e:
        print(f"  [mlb_full] ESPN odds fetch failed: {e}")

    # ── Step 4c: Compute sp_form_combined (used by BOTH ATS and O/U models) ──
    try:
        from mlb_ou_v2_serve import compute_sp_form_combined
        sp_form = compute_sp_form_combined(
            h_starter_id, a_starter_id,
            payload["home_sp_fip"], payload["away_sp_fip"]
        )
        payload["sp_form_combined"] = sp_form
        print(f"  [mlb_full] sp_form_combined: {sp_form:+.3f}")
    except Exception as e:
        payload["sp_form_combined"] = 0.0
        print(f"  [mlb_full] sp_form failed: {e} — defaulting to 0")

    # ── Step 4d: Compute lineup features EARLY (needed by BOTH O/U v3 and ATS v9) ──
    lineup_feats = {}
    if _HAS_V9:
        try:
            season = int(game_date[:4]) if game_date else datetime.now().year
            batter_stats = _fetch_batter_season_stats(season)
            game_pk = game.get("gamePk")
            h_lineup, a_lineup = fetch_pregame_lineups(
                game_pk=game_pk, game_date=game_date,
                home_abbr=home_abbr, away_abbr=away_abbr)
            if h_lineup and a_lineup and batter_stats:
                lineup_feats = compute_lineup_features(
                    h_lineup, a_lineup, batter_stats,
                    home_abbr=home_abbr, away_abbr=away_abbr)
                print(f"  [mlb_full] Lineup: {lineup_feats.get('home_matched',0)}/{lineup_feats.get('away_matched',0)} matched")
            else:
                print(f"  [mlb_full] No lineups available — lineup features = 0")
        except Exception as e:
            print(f"  [mlb_full] Lineup fetch error: {e}")

    # Pass lineup features to O/U payload (v3 uses these)
    payload["lineup_delta_sum"] = lineup_feats.get("lineup_delta_sum", 0)
    payload["lineup_total_top3"] = lineup_feats.get("lineup_total_top3", 0)
    payload["lineup_total_woba"] = lineup_feats.get("lineup_total_woba", 0)

    # ── Step 4e: Look up ump career RPG for O/U v3 ──
    try:
        from config import SUPABASE_URL, SUPABASE_KEY
        import requests as _req
        ump_career = _req.get(
            f"{SUPABASE_URL}/rest/v1/mlb_ump_career?ump_name=eq.{ump_name}&select=career_rpg",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=5).json() if ump_name else []
        if ump_career and ump_career[0].get("career_rpg"):
            payload["ump_career_rpg"] = float(ump_career[0]["career_rpg"])
            payload["ump_career_bb"] = 6.5  # TODO: add to table
            print(f"  [mlb_full] Ump {ump_name}: career_rpg={payload['ump_career_rpg']:.2f}")
        else:
            payload["ump_career_rpg"] = 8.5
            payload["ump_career_bb"] = 6.5
    except Exception:
        payload["ump_career_rpg"] = 8.5
        payload["ump_career_bb"] = 6.5

    # ── Step 5: Call margin + O/U models ──
    margin_result = predict_mlb(payload)
    ou_result = predict_mlb_ou(payload)

    # ── Step 5b: v9 ATS model (reuse lineup features from step 4d) ──
    v9_result = {}
    if _HAS_V9:
        try:
            game_features = margin_result.get("_features", {})
            game_features["sp_form_combined"] = payload.get("sp_form_combined", 0)
            game_features["market_spread"] = payload.get("market_spread_home", 0)

            # Ump home win pct (v9.1 — CatBoost feature)
            ump_hwp = fetch_ump_home_win_pct(ump_name)
            game_features["ump_home_win_pct"] = ump_hwp
            if ump_hwp > 0:
                print(f"  [mlb_v9] Ump {ump_name}: home_win_pct={ump_hwp:.3f}")

            v9_result = predict_mlb_ats_v9(
                game_features, lineup_feats,
                market_spread=float(payload.get("market_spread_home", 0) or 0))
        except Exception as e:
            print(f"  [mlb_v9] Error: {e}")
            traceback.print_exc()

    # ── Step 6: Combine results ──
    combined = {
        "sport": "MLB",
        "mode": "full",
        "home_team": home_abbr,
        "away_team": away_abbr,
        "game_date": game_date,
        "game_id": game.get("gamePk"),
        "home_starter": h_starter_name,
        "away_starter": a_starter_name,
        "umpire": ump_name,
        "is_dome": is_dome,
        # Margin model
        "ml_margin": margin_result.get("ml_margin"),
        "ml_win_prob_home": margin_result.get("ml_win_prob_home"),
        "ml_win_prob_away": margin_result.get("ml_win_prob_away"),
        "bias_correction": margin_result.get("bias_correction"),
        "feature_coverage": margin_result.get("feature_coverage"),
        "models_agree": margin_result.get("models_agree"),
        "model_preds": margin_result.get("model_preds"),
        "rolling_stats_loaded": margin_result.get("rolling_stats_loaded"),
        # O/U model
        "pred_total": ou_result.get("pred_total"),
        "ou_edge": ou_result.get("ou_edge"),
        "ou_pick": ou_result.get("ou_pick"),
        "ou_units": ou_result.get("ou_units"),
        "ou_tier": ou_result.get("ou_tier"),
        "sp_form_combined": ou_result.get("sp_form_combined", payload.get("sp_form_combined")),
        "market_ou_total": payload.get("market_ou_total"),
        # Raw inputs for debugging
        "data_sources": {
            "home_woba": payload["home_woba"],
            "away_woba": payload["away_woba"],
            "home_sp_fip": payload["home_sp_fip"],
            "away_sp_fip": payload["away_sp_fip"],
            "home_team_era": home_team_era,
            "away_team_era": away_team_era,
            "park_factor": park_factor,
            "home_rest": home_rest,
            "away_rest": away_rest,
            "home_games": h_hit.get("games", 0),
            "away_games": a_hit.get("games", 0),
        },
        # SHAP
        "shap": margin_result.get("shap", [])[:10],
        "model_meta": margin_result.get("model_meta"),
        # v9 ATS (lineup-enhanced)
        "ats_v9_side": v9_result.get("ats_v9_side"),
        "ats_v9_units": v9_result.get("ats_v9_units", 0),
        "ats_v9_blend": v9_result.get("ats_v9_blend"),
        "ats_v9_cb": v9_result.get("ats_v9_cb"),
        "ats_v9_lasso": v9_result.get("ats_v9_lasso"),
        "ats_v9_models_agree": v9_result.get("ats_v9_models_agree"),
        "ats_v9_edge": v9_result.get("ats_v9_edge"),
        "lineup_available": bool(lineup_feats),
        "lineup_delta_sum": lineup_feats.get("lineup_delta_sum", 0),
        # O/U v3 extra fields
        "ou_res_avg": ou_result.get("ou_res_avg") or ou_result.get("residual"),
    }

    return combined


def predict_all_games_for_date(game_date):
    """Predict all PreGame games for a given date. Used by cron."""
    data = _mlb_get("schedule", {
        "sportId": 1, "date": game_date,
        "hydrate": "probablePitcher,teams,venue,linescore,officials",
    })
    if not data:
        return []

    results = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status in ("Final", "Live"):
                continue  # skip started/finished games

            game_id = g.get("gamePk")
            try:
                result = predict_mlb_full({"game_id": game_id, "game_date": game_date})
                if result and "error" not in result:
                    results.append(result)
            except Exception as e:
                print(f"  [mlb_full] Error predicting game {game_id}: {e}")

    return results
