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
            "hydrate": "probablePitcher,teams,venue,linescore,officials,weather",
        })
    elif game_date:
        data = _mlb_get("schedule", {
            "sportId": 1, "date": game_date,
            "hydrate": "probablePitcher,teams,venue,linescore,officials,weather",
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
    cFIP = lg_fip - (13 * 1.0 + 3 * 3.2 - 2 * 8.5) / 9  # FIP constant (per-IP basis)
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


def _blend_pitcher_stats(current, prior, label=""):
    """Blend current season stats with prior season, weighted by IP.
    
    Early season: starter has 12 IP in 2026, 180 IP in 2025.
    Blend = (12 × current + 180 × prior) / 192 → heavily weighted to prior.
    Mid season: 100 IP in 2026 → mostly current.
    
    This prevents 2-start FIPs from dominating predictions.
    """
    if not current and not prior:
        return None
    if not current:
        return prior  # No current season data — use prior as-is
    if not prior:
        return current  # No prior — use current as-is (veteran moved leagues, rookie)
    
    c_ip = current.get("ip_total", 0) or 0
    p_ip = prior.get("ip_total", 0) or 0
    
    if c_ip + p_ip == 0:
        return current
    
    # Blend rate stats by IP
    blended = dict(current)  # Start with current, override blended fields
    for stat in ["era", "k9", "bb9", "whip"]:
        c_val = current.get(stat, 0) or 0
        p_val = prior.get(stat, 0) or 0
        if c_ip > 0 and p_ip > 0:
            blended[stat] = round((c_ip * c_val + p_ip * p_val) / (c_ip + p_ip), 3)
        elif c_ip > 0:
            blended[stat] = c_val
        else:
            blended[stat] = p_val
    
    # For FIP components (hr, k, bb), keep current season raw counts
    # but store blended IP for FIP calculation
    blended["_blended_ip"] = c_ip + p_ip
    blended["_current_ip"] = c_ip
    blended["_prior_ip"] = p_ip
    
    # Blend IP per start (use current if enough starts, else blend)
    c_gs = current.get("games_started", 0) or 0
    p_gs = prior.get("games_started", 0) or 0
    if c_gs >= 3:
        blended["ip_per_start"] = current["ip_per_start"]
    elif c_gs > 0 and p_gs > 0:
        blended["ip_per_start"] = round(
            (c_gs * current.get("ip_per_start", 5.5) + p_gs * prior.get("ip_per_start", 5.5)) 
            / (c_gs + p_gs), 2)
    
    if c_ip < 30:  # Early season — log the blend
        print(f"  [mlb_full] {label} SP blended: {c_ip:.0f} IP (current) + {p_ip:.0f} IP (prior) → FIP components weighted")
    
    return blended


def _fetch_team_relief_era(team_id, season=None):
    """Fetch team bullpen ERA from MLB API relief-only stats."""
    if not team_id:
        return None
    if not season:
        season = datetime.utcnow().year
    try:
        data = _mlb_get(f"teams/{team_id}/stats", {
            "stats": "season", "season": season, "group": "pitching",
            "sitCodes": "rp",
        })
        if data:
            for split in data.get("stats", []):
                for s in split.get("splits", []):
                    st = s.get("stat", {})
                    era = float(st.get("era", 0))
                    if era > 0:
                        return era
    except Exception:
        pass
    return None


def _compute_platoon_delta(lineup_ids, opp_pitcher_id):
    """Compute platoon advantage for a lineup vs opposing starter's throw hand.
    
    Opposite hand matchup = advantage (+1 per batter)
    Same hand = disadvantage (0)
    Switch hitter = partial advantage (+0.5)
    Returns: average platoon score across lineup (range ~ -1.0 to +1.0)
    """
    if not lineup_ids or not opp_pitcher_id:
        return 0.0
    
    try:
        # Batch fetch all player handedness in one call
        all_ids = [str(opp_pitcher_id)] + [str(bid) for bid in lineup_ids if bid]
        id_str = ",".join(all_ids[:20])  # Cap at 20
        data = _mlb_get("people", {"personIds": id_str})
        if not data:
            return 0.0
        
        # Build handedness lookup
        hand_map = {}
        for p in data.get("people", []):
            pid = p.get("id")
            hand_map[pid] = {
                "bat": p.get("batSide", {}).get("code", "R"),
                "throw": p.get("pitchHand", {}).get("code", "R"),
            }
        
        # Get pitcher throw hand
        sp_throw = hand_map.get(int(opp_pitcher_id), {}).get("throw", "R")
        
        # Score each batter
        scores = []
        for bid in lineup_ids:
            if not bid:
                continue
            batter = hand_map.get(int(bid), {})
            bat_side = batter.get("bat", "R")
            if bat_side == "S":  # Switch hitter
                scores.append(0.5)
            elif bat_side != sp_throw:  # Opposite hand = advantage
                scores.append(1.0)
            else:  # Same hand = no advantage
                scores.append(0.0)
        
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as e:
        print(f"  [mlb_full] Platoon compute error: {e}")
        return 0.0


def _compute_bp_fatigue(team_id, game_date_str):
    """Estimate bullpen fatigue: IP thrown by relievers in last 3 days."""
    try:
        dt = datetime.strptime(game_date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        data = _mlb_get("schedule", {
            "sportId": 1, "teamId": team_id,
            "startDate": start, "endDate": end,
        })
        if not data:
            return 0
        games_played = sum(1 for d in data.get("dates", []) for g in d.get("games", [])
                          if g.get("status", {}).get("abstractGameState") == "Final")
        # Avg bullpen usage: ~3.5 IP per game
        return round(games_played * 3.5, 1)
    except Exception:
        return 0


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

    # No starters = no prediction — sp_form_delta (strongest feature) would be zero
    if not h_starter_id or not a_starter_id:
        print(f"  [mlb_full] ⚠️ Starters not announced for {away_abbr}@{home_abbr} — skipping")
        return {"error": "Starters not announced", "skip": True, "home_team": home_abbr, "away_team": away_abbr}

    # Umpire
    ump_name = _extract_umpire(game)

    # Park factor
    park_factor = _PARK_FACTORS.get(h_id, 1.00)
    is_dome = _is_dome(home_abbr)

    print(f"  [mlb_full] {away_abbr}@{home_abbr} | SP: {a_starter_name} vs {h_starter_name} | Ump: {ump_name}")

    # ── Step 2: Parallel data fetching (includes prior season for blending) ──
    prior_season = season - 1
    results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_fetch_team_stats, h_id, season): "home_stats",
            pool.submit(_fetch_team_stats, a_id, season): "away_stats",
            pool.submit(_fetch_pitcher_stats, h_starter_id, season): "home_sp",
            pool.submit(_fetch_pitcher_stats, a_starter_id, season): "away_sp",
            pool.submit(_fetch_pitcher_stats, h_starter_id, prior_season): "home_sp_prior",
            pool.submit(_fetch_pitcher_stats, a_starter_id, prior_season): "away_sp_prior",
            pool.submit(_fetch_team_relief_era, h_id, season): "home_bp_era",
            pool.submit(_fetch_team_relief_era, a_id, season): "away_bp_era",
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
    home_sp = _blend_pitcher_stats(results.get("home_sp"), results.get("home_sp_prior"), "home")
    away_sp = _blend_pitcher_stats(results.get("away_sp"), results.get("away_sp_prior"), "away")
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

    # Starter FIP — blend current + prior season for stability
    # Early season: 2 starts of data is noise. Prior season adds real signal.
    for side, sp, sp_prior_key, team_era, fip_var in [
        ("home", home_sp, "home_sp_prior", home_team_era, "home_sp_fip"),
        ("away", away_sp, "away_sp_prior", away_team_era, "away_sp_fip"),
    ]:
        prior = results.get(sp_prior_key)
        if sp and sp.get("ip_total", 0) > 0:
            c_ip = sp.get("_current_ip", sp.get("ip_total", 0))
            # If early season (<30 IP) and prior exists, combine raw counts for FIP
            if c_ip < 30 and prior and prior.get("ip_total", 0) > 0:
                total_hr = sp.get("hr", 0) + prior.get("hr", 0)
                total_bb = sp.get("bb", 0) + prior.get("bb", 0)
                total_k = sp.get("k", 0) + prior.get("k", 0)
                total_ip = sp.get("ip_total", 0) + prior.get("ip_total", 0)
                fip = _compute_fip(total_hr, total_bb, total_k, total_ip, sc["lg_fip"])
                print(f"  [mlb_full] {side} SP FIP blended: {c_ip:.0f}+{prior['ip_total']:.0f} IP → FIP={fip:.2f}")
            else:
                fip = _compute_fip(sp["hr"], sp["bb"], sp["k"], sp["ip_total"], sc["lg_fip"])
        elif sp:
            fip = sp.get("era", sc["lg_fip"])
        else:
            fip = team_era
        
        if side == "home":
            home_sp_fip = fip
        else:
            away_sp_fip = fip

    # Weather — extract from MLB API schedule hydration
    if is_dome:
        temp_f, wind_mph, wind_out_flag = 70.0, 0.0, 0.0
    else:
        weather = game.get("weather", {})
        try:
            temp_f = float(weather.get("temp", 70))
        except (ValueError, TypeError):
            temp_f = 70.0
        
        # Parse wind: "14 mph, In From LF" → speed=14, direction=in
        wind_str = weather.get("wind", "")
        try:
            wind_mph = float(wind_str.split(" mph")[0]) if "mph" in wind_str else 5.0
        except (ValueError, TypeError):
            wind_mph = 5.0
        
        # Wind out = blowing toward outfield (increases scoring)
        wind_lower = wind_str.lower()
        if "out to" in wind_lower or "out from" in wind_lower:
            wind_out_flag = 1.0
        else:
            wind_out_flag = 0.0
        
        if temp_f != 70 or wind_mph != 5:
            print(f"  [mlb_full] Weather: {temp_f:.0f}°F, wind {wind_mph:.0f} mph {'OUT' if wind_out_flag else 'in'} ({weather.get('condition', '')})")

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
        "home_bullpen_era": round(results.get("home_bp_era") or home_team_era, 2),
        "away_bullpen_era": round(results.get("away_bp_era") or away_team_era, 2),
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
        # Heuristic prediction (server-side — no frontend dependency)
        # Simple model: (team_woba / lg_woba) * lg_rpg/2 * park * (lg_fip / opp_sp_fip)
        "pred_home_runs": round(
            (home_woba / sc["lg_woba"]) * (sc["lg_rpg"] / 2) * park_factor * (sc["lg_fip"] / max(away_sp_fip, 2.0)) + 0.08,  # +0.08 HFA
            2),
        "pred_away_runs": round(
            (away_woba / sc["lg_woba"]) * (sc["lg_rpg"] / 2) * park_factor * (sc["lg_fip"] / max(home_sp_fip, 2.0)) - 0.08,  # -0.08 HFA
            2),
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
                # Fall back to market data passed in from caller (e.g., stored in Supabase)
                for mf in ["market_ou_total", "market_home_ml", "market_away_ml", "market_spread_home"]:
                    if input_data.get(mf) and not payload.get(mf):
                        payload[mf] = input_data[mf]
                if payload.get("market_ou_total"):
                    print(f"  [mlb_full] Market (from stored): O/U={payload['market_ou_total']}, RL={payload.get('market_spread_home')}")
                else:
                    print(f"  [mlb_full] No ESPN odds found for {away_abbr}@{home_abbr}")
    except Exception as e:
        print(f"  [mlb_full] ESPN odds fetch failed: {e}")

    # ── Step 4c: Compute sp_form_combined + individual deltas (v9 features) ──
    try:
        from mlb_ou_v2_serve import compute_sp_form_combined
        sp_form = compute_sp_form_combined(
            h_starter_id, a_starter_id,
            payload["home_sp_fip"], payload["away_sp_fip"]
        )
        payload["sp_form_combined"] = sp_form
        print(f"  [mlb_full] sp_form_combined: {sp_form:+.3f}")

        # v9: Compute individual SP form deltas (away is 100x more predictive than home)
        try:
            from mlb_ou_v3_serve import fetch_pitcher_recent_form
            for side, pid, fip_key in [("home", h_starter_id, "home_sp_fip"),
                                        ("away", a_starter_id, "away_sp_fip")]:
                form = fetch_pitcher_recent_form(pid) if pid else None
                if form and form.get("n_starts", 0) >= 1:
                    delta = form["recent_avg_runs"] - payload[fip_key]
                    payload[f"{side}_sp_form_delta"] = round(delta, 3)
                else:
                    payload[f"{side}_sp_form_delta"] = 0
        except Exception as e:
            payload["home_sp_form_delta"] = 0
            payload["away_sp_form_delta"] = 0
            print(f"  [mlb_full] SP form deltas failed: {e}")
    except Exception as e:
        payload["sp_form_combined"] = 0.0
        payload["home_sp_form_delta"] = 0
        payload["away_sp_form_delta"] = 0
        print(f"  [mlb_full] sp_form failed: {e} — defaulting to 0")

    # ── Step 4c2: Bullpen fatigue (recent bullpen IP) ──
    for side, tid in [("home", h_id), ("away", a_id)]:
        try:
            fatigue = _compute_bp_fatigue(tid, game_date)
            payload[f"{side}_bp_fatigue"] = fatigue
        except Exception:
            payload[f"{side}_bp_fatigue"] = 0

    # ── (Rolling FIP tested: r≈0 vs residual — market already prices it. Clamps handle edge cases.) ──

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
                payload["home_lineup_confirmed"] = 1
                payload["away_lineup_confirmed"] = 1                
                # Compute platoon advantage from lineup handedness vs opposing SP
                home_platoon = _compute_platoon_delta(h_lineup, a_starter_id)
                away_platoon = _compute_platoon_delta(a_lineup, h_starter_id)
                payload["home_platoon_delta"] = home_platoon
                payload["away_platoon_delta"] = away_platoon
                if abs(home_platoon - away_platoon) > 0.01:
                    print(f"  [mlb_full] Platoon: home={home_platoon:+.3f} away={away_platoon:+.3f} diff={home_platoon-away_platoon:+.3f}")
            else:
                print(f"  [mlb_full] No lineups available — lineup features = 0")
        except Exception as e:
            print(f"  [mlb_full] Lineup fetch error: {e}")

    # Pass lineup features to O/U payload (v3 uses these)
    payload["game_date"] = game_date
    payload["home_games"] = h_hit.get("games", 0)
    payload["away_games"] = a_hit.get("games", 0)
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

    # ── Step 4f: Clamps REMOVED — blended FIP handles early-season extremes ──
    # Previously clamped all SP FIPs to 2.5-6.5, which compressed all early-season
    # starters to identical values (37.1% ML). Blended prior+current FIP replaces this.
    # _CLAMPS = { ... }  # REMOVED

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
