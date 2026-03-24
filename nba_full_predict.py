"""
nba_full_predict.py — Server-side enriched NBA prediction (v26 Lasso)

ARCHITECTURE: Single ESPN summary call extracts ~50 of 66 features.
Supabase supplements rolling PBP stats. Elo from local file.

Data extracted from ESPN /summary endpoint:
  - Team stats: PPG, OPP PPG, FG%, 3P%, REB, AST, BLK, STL, TO
  - Pickcenter: spread, total, ML odds (opening AND closing)
  - Predictor: ESPN win probability
  - Leaders: Star player PPG, APG, RPG, MPG, FG%
  - Season series: H2H results
  - Standings: W/L/PCT/GB/streak per team
  - Last 5 games: scores, dates, opponents (rolling margins, rest, B2B)
  - Injuries: count of OUT players per team
  - Venue: capacity (crowd_pct)
  - Line movement: opening vs closing spread/ML (reverse_line_movement)

Route: POST /predict/nba/full
Input: {"game_id": "401810897"} or {"home_team":"CHI","away_team":"HOU","game_date":"2026-03-23"}
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import requests
import traceback as _tb
from datetime import datetime, timedelta

# ── Constants ──
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
PUBLIC_TEAMS = {"LAL","GSW","BOS","NYK","CHI","MIA","PHX","DAL","BKN"}
VENUE_CAPACITY = {
    "ATL":16600,"BOS":19156,"BKN":17732,"CHA":19077,"CHI":20917,"CLE":19432,
    "DAL":19200,"DEN":19520,"DET":20332,"GSW":18064,"HOU":18055,"IND":18165,
    "LAC":18997,"LAL":18997,"MEM":17794,"MIA":19600,"MIL":17341,"MIN":17136,
    "NOP":16867,"NYK":19812,"OKC":18203,"ORL":18846,"PHI":20478,"PHX":18055,
    "POR":19441,"SAC":17583,"SAS":18581,"TOR":19800,"UTA":18306,"WAS":20356,
}

def _map(a): return ESPN_ABBR_MAP.get(a, a)
def _ml_to_prob(ml):
    if not ml or ml == 0: return 0.5
    return 100/(ml+100) if ml > 0 else abs(ml)/(abs(ml)+100)
def _parse_streak(s):
    if not s: return 0
    try:
        if s.startswith("W"): return int(s[1:])
        if s.startswith("L"): return -int(s[1:])
    except: pass
    return 0
def _parse_line(s):
    if not s: return 0
    try: return float(str(s).replace("+",""))
    except: return 0
def _parse_odds(s):
    if not s: return 0
    try: return int(str(s).replace("+",""))
    except: return 0


# ═══════════════════════════════════════════════════════════
# ESPN SUMMARY PARSER
# ═══════════════════════════════════════════════════════════

def _parse_espn_summary(data, home_abbr, away_abbr):
    """Parse ESPN /summary into feature dict. Returns (row_dict, diag_list)."""
    row = {}
    diag = []
    
    def _safe_get(obj, *keys, default=None):
        """Safely traverse nested dicts that might contain strings."""
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k, default)
            else:
                return default
        return obj

    def _safe_abbr(obj):
        """Safely extract team abbreviation from a field that might be a string."""
        if isinstance(obj, dict):
            return _map(obj.get("abbreviation", ""))
        return ""

    # ── 1. Team stats from boxscore ──
    try:
        for tb in data.get("boxscore", {}).get("teams", []):
            abbr = _safe_abbr(tb.get("team", {}))
            side = "home" if abbr == home_abbr else "away" if abbr == away_abbr else None
            if not side: continue
            stats = {}
            for s in tb.get("statistics", []):
                if isinstance(s, dict):
                    stats[s.get("name", "")] = s.get("displayValue", "")
            def g(n, d=0):
                v = stats.get(n)
                if v is None: return d
                try: return float(v)
                except: return d

            row[f"{side}_ppg"] = g("avgPoints", 112)
            row[f"{side}_opp_ppg"] = g("avgPointsAgainst", 112)
            row[f"{side}_fgpct"] = g("fieldGoalPct", 47) / 100
            row[f"{side}_threepct"] = g("threePointFieldGoalPct", 36) / 100
            row[f"{side}_assists"] = g("avgAssists", 25)
            row[f"{side}_blocks"] = g("avgBlocks", 5)
            row[f"{side}_steals"] = g("avgSteals", 7.5)
            row[f"{side}_turnovers"] = g("avgTotalTurnovers", g("avgTeamTurnovers", 14))
            row[f"{side}_total_reb"] = g("avgRebounds", 44)
            row[f"{side}_streak"] = _parse_streak(stats.get("streak", ""))
            ppg = row[f"{side}_ppg"]; opp = row[f"{side}_opp_ppg"]
            row[f"{side}_tempo"] = round(max((ppg + opp) / 2 / 1.1, 80), 1)
            row[f"{side}_net_rtg"] = round(ppg - opp, 1)
            row[f"{side}_ato_ratio"] = round(row[f"{side}_assists"] / max(row[f"{side}_turnovers"], 1), 2)
    except Exception as _e:
        diag.append(f"Section 1 (boxscore): {_e}")

    # ── 2. Pickcenter (market + line movement) ──
    try:
        pc_list = data.get("pickcenter", [])
        if pc_list:
            pc = pc_list[0]
            row["market_spread_home"] = pc.get("spread", 0) or 0
            row["market_ou_total"] = pc.get("overUnder", 0) or 0
            ho = pc.get("homeTeamOdds", {}) if isinstance(pc.get("homeTeamOdds"), dict) else {}
            ao = pc.get("awayTeamOdds", {}) if isinstance(pc.get("awayTeamOdds"), dict) else {}
            row["home_ml"] = ho.get("moneyLine", 0) or 0
            row["away_ml"] = ao.get("moneyLine", 0) or 0

            # Opening vs closing lines
            ps = pc.get("pointSpread", {}) if isinstance(pc.get("pointSpread"), dict) else {}
            mline = pc.get("moneyline", {}) if isinstance(pc.get("moneyline"), dict) else {}
            open_sp = _parse_line(_safe_get(ps, "home", "open", "line"))
            close_sp = _parse_line(_safe_get(ps, "home", "close", "line"))
            open_ml_h = _parse_odds(_safe_get(mline, "home", "open", "odds"))
            close_ml_h = _parse_odds(_safe_get(mline, "home", "close", "odds"))

            row["_spread_move"] = (close_sp - open_sp) if (open_sp and close_sp) else 0
            row["_ml_move"] = round(_ml_to_prob(close_ml_h) - _ml_to_prob(open_ml_h), 4) if open_ml_h else 0
            row["_open_spread"] = open_sp
            row["_close_spread"] = close_sp

            # O/U movement
            ot = _safe_get(pc, "total", "over", "open", "line", default="")
            open_total = 0
            if ot and str(ot).startswith("o"):
                try: open_total = float(str(ot)[1:])
                except: pass
            row["ou_movement"] = round((row["market_ou_total"] or 0) - open_total, 1) if open_total else 0

            diag.append(f"Spread: {row['market_spread_home']} (open {open_sp})")
            diag.append(f"ML: {row['home_ml']}/{row['away_ml']}")
    except Exception as _e:
        diag.append(f"Section 2 (pickcenter): {_e}")

    # ── 3. Predictor (ESPN win probability) ──
    try:
        pred = data.get("predictor", {})
        if isinstance(pred, dict) and pred:
            hp = _safe_get(pred, "homeTeam", "gameProjection")
            if hp:
                row["espn_pregame_wp"] = float(hp) / 100
                row["espn_pregame_wp_pbp"] = float(hp) / 100
    except Exception as _e:
        diag.append(f"Section 3 (predictor): {_e}")

    # ── 4. Leaders (star player data) ──
    try:
        for lg in data.get("leaders", []):
            abbr = _safe_abbr(lg.get("team", {}))
            side = "home" if abbr == home_abbr else "away" if abbr == away_abbr else None
            if not side: continue
            for cat in lg.get("leaders", []):
                tops = cat.get("leaders", [])
                if not tops: continue
                top = tops[0]
                mv = 0
                try: mv = float(top.get("mainStat", {}).get("value", "0"))
                except: pass
                cn = cat.get("name", "")
                if cn == "pointsPerGame":
                    row[f"{side}_star1_ppg"] = mv
                    for st in top.get("statistics", []):
                        if isinstance(st, dict) and st.get("name") == "fieldGoalPct":
                            try: row[f"{side}_star1_fgpct"] = float(st["value"]) / 100
                            except: pass
                elif cn == "assistsPerGame":
                    for st in top.get("statistics", []):
                        if isinstance(st, dict) and st.get("name") == "avgMinutes":
                            try: row[f"{side}_star_mpg"] = float(st["value"])
                            except: pass
    except Exception as _e:
        diag.append(f"Section 4 (leaders): {_e}")

    # ── 5. Season series (H2H) ──
    try:
        for ss in data.get("seasonseries", []):
            completed = sum(1 for ev in ss.get("events", [])
                           if _safe_get(ev, "status", "statusType", "completed"))
            row["_h2h_n"] = completed
            margins = []
            for ev in ss.get("events", []):
                if not _safe_get(ev, "status", "statusType", "completed"): continue
                comps = ev.get("competitors", [])
                hc = next((c for c in comps if c.get("homeAway")=="home"), None)
                ac = next((c for c in comps if c.get("homeAway")=="away"), None)
                if hc and ac:
                    try:
                        hs, aws = int(hc["score"]), int(ac["score"])
                        ha = _safe_abbr(hc.get("team", {}))
                        margins.append((hs-aws) if ha == home_abbr else (aws-hs))
                    except: pass
            row["_h2h_margin"] = round(np.mean(margins), 1) if margins else 0
            break
    except Exception as _e:
        diag.append(f"Section 5 (H2H): {_e}")

    # ── 6. Standings ──
    try:
        for group in data.get("standings", {}).get("groups", []):
            for entry in group.get("standings", {}).get("entries", []):
                # Match entry to team
                entry_abbr = None
                entry_id = entry.get("id") or entry.get("uid", "")
                for abbr, eid in NBA_ESPN_IDS.items():
                    if str(eid) == str(entry_id):
                        entry_abbr = abbr; break
                if not entry_abbr:
                    _et = entry.get("team", "")
                    tn = str(_et).lower() if not isinstance(_et, dict) else str(_et.get("displayName", "")).lower()
                    for abbr in [home_abbr, away_abbr]:
                        city_map = {"ATL":"atlanta","BOS":"boston","CHI":"chicago","CLE":"cleveland",
                                    "DAL":"dallas","DEN":"denver","DET":"detroit","GSW":"golden",
                                    "HOU":"houston","IND":"indiana","LAC":"la clip","LAL":"la laker",
                                    "MEM":"memphis","MIA":"miami","MIL":"milwaukee","MIN":"minnesota",
                                    "NOP":"new orleans","NYK":"new york","OKC":"oklahoma","ORL":"orlando",
                                    "PHI":"philadelphia","PHX":"phoenix","POR":"portland","SAC":"sacramento",
                                    "SAS":"san antonio","TOR":"toronto","UTA":"utah","WAS":"washington",
                                    "BKN":"brooklyn"}
                        if city_map.get(abbr, "xxx") in tn:
                            entry_abbr = abbr; break
                if entry_abbr not in (home_abbr, away_abbr): continue
                side = "home" if entry_abbr == home_abbr else "away"
                for st in entry.get("stats", []):
                    if not isinstance(st, dict): continue
                    val = st.get("value")
                    if val is None: continue
                    try:
                        t = st.get("type", "")
                        if t == "wins": row[f"{side}_wins"] = int(val)
                        elif t == "losses": row[f"{side}_losses"] = int(val)
                        elif t == "winpercent": row[f"{side}_win_pct_standings"] = float(val)
                    except: pass
    except Exception as _e:
        diag.append(f"Section 6 (standings): {_e}")

    # ── 7. Last 5 games (rolling margins, rest, B2B) ──
    try:
        for l5 in data.get("lastFiveGames", []):
            abbr = _safe_abbr(l5.get("team", {}))
            side = "home" if abbr == home_abbr else "away" if abbr == away_abbr else None
            if not side: continue
            events = l5.get("events", [])
            margins, dates = [], []
            team_id = str(NBA_ESPN_IDS.get(abbr, ""))
            for ev in events:
                try:
                    hs = int(ev.get("homeTeamScore", 0) or 0)
                    aws = int(ev.get("awayTeamScore", 0) or 0)
                    if str(ev.get("homeTeamId", "")) == team_id:
                        margins.append(hs - aws)
                    else:
                        margins.append(aws - hs)
                    gd = ev.get("gameDate", "")
                    if gd: dates.append(gd[:10])
                except: pass
            if margins:
                row[f"{side}_margin_trend"] = round(np.mean(margins), 2)
                row[f"{side}_scoring_var"] = round(float(np.std(margins)), 2) if len(margins) >= 3 else 12.0
            if dates:
                try:
                    last = datetime.strptime(dates[0], "%Y-%m-%d")
                    row[f"{side}_days_rest"] = max(0, (datetime.now() - last).days - 1)
                    if len(dates) >= 2:
                        prev = datetime.strptime(dates[1], "%Y-%m-%d")
                        if (last - prev).days <= 1:
                            row[f"{side}_is_b2b"] = 1
                    cutoff = datetime.now() - timedelta(days=14)
                    row[f"{side}_games_14d"] = sum(1 for d in dates if datetime.strptime(d, "%Y-%m-%d") >= cutoff)
                except: pass
            wins_l5 = sum(1 for m in margins if m > 0)
            row[f"{side}_form_l5"] = round((wins_l5/max(len(margins),1))*2-1, 3) if margins else 0
    except Exception as _e:
        diag.append(f"Section 7 (last5): {_e}")

    # ── 8. Injuries ──
    try:
        for inj in data.get("injuries", []):
            abbr = _safe_abbr(inj.get("team", {}))
            side = "home" if abbr == home_abbr else "away" if abbr == away_abbr else None
            if not side: continue
            out = sum(1 for i in inj.get("injuries", [])
                     if isinstance(i, dict) and _safe_get(i, "type", "abbreviation") == "O")
            row[f"{side}_injuries_out"] = out
    except Exception as _e:
        diag.append(f"Section 8 (injuries): {_e}")

    # ── 9. Venue + Header ──
    try:
        cap = _safe_get(data, "gameInfo", "venue", "capacity") or VENUE_CAPACITY.get(home_abbr, 19000)
        row["_venue_cap"] = cap

        header = data.get("header", {})
        for comp in header.get("competitions", [{}]):
            gd = comp.get("date", "")
            if gd: row["_game_date_utc"] = gd[:10]
            for c in comp.get("competitors", []):
                a = _safe_abbr(c.get("team", {}))
                s = "home" if a == home_abbr else "away" if a == away_abbr else None
                if not s: continue
                for rec in c.get("record", []):
                    if isinstance(rec, dict) and rec.get("type") == "total":
                        try:
                            parts = rec["summary"].split("-")
                            row.setdefault(f"{s}_wins", int(parts[0]))
                            row.setdefault(f"{s}_losses", int(parts[1]))
                        except: pass
    except Exception as _e:
        diag.append(f"Section 9 (venue/header): {_e}")

    return row, diag


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _supabase_headers():
    from config import SUPABASE_URL, SUPABASE_KEY
    return (SUPABASE_URL, {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"})

def _lookup_sb(game_id=None, home=None, away=None, date=None):
    url, h = _supabase_headers()
    if game_id: q = f"{url}/rest/v1/nba_predictions?game_id=eq.{game_id}&select=*&limit=1"
    elif home and away and date:
        q = f"{url}/rest/v1/nba_predictions?home_team=eq.{home}&away_team=eq.{away}&game_date=eq.{date}&select=*&limit=1"
    else: return None
    try:
        rows = requests.get(q, headers=h, timeout=10).json()
        return rows[0] if rows else None
    except: return None

def _sb_rolling(team, before, n=15):
    url, h = _supabase_headers()
    games = []
    for side, sign in [("home_team",1),("away_team",-1)]:
        q = (f"{url}/rest/v1/nba_predictions?result_entered=eq.true&{side}=eq.{team}"
             f"&game_date=lt.{before}&order=game_date.desc&limit={n}"
             f"&select=game_date,actual_home_score,actual_away_score,market_spread_home")
        try:
            for r in requests.get(q, headers=h, timeout=10).json() or []:
                m = ((r.get("actual_home_score") or 0)-(r.get("actual_away_score") or 0))*sign
                sp = (r.get("market_spread_home") or 0)*sign
                games.append({"margin":m,"spread":sp})
        except: pass
    ats = [g["margin"]+g["spread"] for g in games[:10] if g["spread"]]
    return {"ats_avg": round(float(np.mean(ats)),2) if ats else 0}

def _load_elo():
    for p in ["nba_elo_ratings.json","models/nba_elo_ratings.json"]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f).get("ratings",{})
    return {}

def _load_model():
    for p in ["nba_model_local.pkl","models/nba_model_local.pkl"]:
        if os.path.exists(p):
            with open(p,"rb") as f: return pickle.load(f)
    return None

def _find_game_id(home, away, date):
    ds = date.replace("-","")
    try:
        for ev in requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ds}&limit=50", timeout=10).json().get("events",[]):
            comp = ev["competitions"][0]
            h = _map(next(c for c in comp["competitors"] if c["homeAway"]=="home")["team"]["abbreviation"])
            a = _map(next(c for c in comp["competitors"] if c["homeAway"]=="away")["team"]["abbreviation"])
            if h == home and a == away: return str(ev["id"])
    except: pass
    return None


# ═══════════════════════════════════════════════════════════
# MAIN PREDICTION
# ═══════════════════════════════════════════════════════════

def predict_nba_full(game: dict):
    game_id = game.get("game_id")
    home_abbr = _map(game.get("home_team", ""))
    away_abbr = _map(game.get("away_team", ""))
    game_date = game.get("game_date", datetime.now().strftime("%Y-%m-%d"))
    diag = {"sources": [], "warnings": []}

    # ═══ 1. Discover game_id ═══
    if not game_id and home_abbr and away_abbr:
        game_id = _find_game_id(home_abbr, away_abbr, game_date)

    # ═══ 2. ESPN Summary (PRIMARY data source) ═══
    espn = {}
    if game_id:
        try:
            r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}", timeout=15)
            if r.ok:
                raw = r.json()
                if not home_abbr or not away_abbr:
                    for comp in raw.get("header",{}).get("competitions",[{}]):
                        for c in comp.get("competitors",[]):
                            a = _map(c.get("team",{}).get("abbreviation",""))
                            if c.get("homeAway")=="home": home_abbr = a
                            elif c.get("homeAway")=="away": away_abbr = a
                        gd = comp.get("date","")
                        if gd: game_date = gd[:10]
                if home_abbr and away_abbr:
                    espn, espn_diag = _parse_espn_summary(raw, home_abbr, away_abbr)
                    diag["sources"].append(f"ESPN summary ({len(espn)} fields)")
                    diag["sources"].extend(espn_diag)
        except Exception as e:
            diag["warnings"].append(f"ESPN: {e}")

    if not home_abbr or not away_abbr:
        return {"error": "home_team and away_team required (or game_id)"}

    # ═══ 3. Supabase row (supplements) ═══
    sb = _lookup_sb(game_id, home_abbr, away_abbr, game_date)
    if sb:
        diag["sources"].append(f"Supabase ({len([v for v in sb.values() if v is not None])} cols)")

    # ═══ 4. Elo ═══
    elo = _load_elo()
    h_elo = elo.get(home_abbr, 1500); a_elo = elo.get(away_abbr, 1500)
    if elo: diag["sources"].append("Elo")

    # ═══ 5. Supabase rolling ATS ═══
    h_sr = _sb_rolling(home_abbr, game_date)
    a_sr = _sb_rolling(away_abbr, game_date)

    # ═══ 6. Assemble row ═══
    row = {}
    if sb:
        for k, v in sb.items():
            if v is not None and k != "id": row[k] = v
    for k, v in espn.items():
        if v is not None and not k.startswith("_"): row[k] = v

    row["game_date"] = game_date; row["home_team"] = home_abbr
    row["away_team"] = away_abbr; row["season"] = 2026
    row["home_elo"] = h_elo; row["away_elo"] = a_elo; row["elo_diff"] = h_elo - a_elo
    row["model_ml_home"] = espn.get("home_ml", 0)
    row.setdefault("market_spread_home", 0); row.setdefault("market_ou_total", 228)
    row.setdefault("espn_pregame_wp", 0.5); row.setdefault("espn_pregame_wp_pbp", 0.5)

    hw = row.get("home_wins", 20); hl = row.get("home_losses", 20)
    row["win_pct_home"] = round(hw / max(hw+hl, 1), 4)
    row.setdefault("home_form", espn.get("home_form_l5", 0))
    row.setdefault("away_form", espn.get("away_form_l5", 0))
    row.setdefault("home_days_rest", 2); row.setdefault("away_days_rest", 2)
    row.setdefault("away_travel_dist", 0)
    row.setdefault("pred_home_score", round(row.get("home_ppg", 112) + 1.5, 1))
    row.setdefault("pred_away_score", round(row.get("away_ppg", 112), 1))

    for k, v in {"home_ppg":112,"away_ppg":112,"home_opp_ppg":112,"away_opp_ppg":112,
                  "home_fgpct":0.471,"away_fgpct":0.471,"home_threepct":0.365,"away_threepct":0.365,
                  "home_ftpct":0.78,"away_ftpct":0.78,"home_tempo":100,"away_tempo":100,
                  "home_net_rtg":0,"away_net_rtg":0,"home_orb_pct":0.25,"away_orb_pct":0.25,
                  "home_fta_rate":0.28,"away_fta_rate":0.28,"home_ato_ratio":1.8,"away_ato_ratio":1.8,
                  "home_opp_fgpct":0.471,"away_opp_fgpct":0.471,"home_opp_threepct":0.365,
                  "away_opp_threepct":0.365,"home_assists":25,"away_assists":25,
                  "home_turnovers":14,"away_turnovers":14,"home_steals":7.5,"away_steals":7.5,
                  "home_blocks":5.0,"away_blocks":5.0}.items():
        row.setdefault(k, v)

    # ═══ 7. Pipeline ═══
    df = pd.DataFrame([row])
    if "actual_home_score" not in df.columns:
        df["actual_home_score"] = df["pred_home_score"]
        df["actual_away_score"] = df["pred_away_score"]

    try:
        from sports.nba import _nba_backfill_heuristic
        df = _nba_backfill_heuristic(df)
    except Exception as e: diag["warnings"].append(f"backfill: {e}")

    try:
        from enrich_nba_v2 import enrich as enrich_v2
        df = enrich_v2(df)
        diag["sources"].append("enrich_v2")
    except Exception as e: diag["warnings"].append(f"enrich_v2: {e}")

    # ═══ 8. Build features + override from ESPN ═══
    bundle = _load_model()
    if not bundle: return {"error": "NBA model not found"}

    try:
        from nba_build_features_v25 import nba_build_features
    except ImportError:
        from sports.nba import nba_build_features

    X = nba_build_features(df)
    feature_list = bundle["feature_list"]
    for f in feature_list:
        if f not in X.columns: X[f] = 0.0

    # ── Override features from ESPN that pipeline couldn't compute ──
    spread = row.get("market_spread_home", 0) or 0
    home_ml = espn.get("home_ml", 0)
    away_ml = espn.get("away_ml", 0)
    impl_h = _ml_to_prob(home_ml)
    impl_a = _ml_to_prob(away_ml)

    h_star = espn.get("home_star1_ppg", 0); a_star = espn.get("away_star1_ppg", 0)
    h_ppg = row.get("home_ppg", 112); a_ppg = row.get("away_ppg", 112)
    h_fg = espn.get("home_star1_fgpct", 0.45); a_fg = espn.get("away_star1_fgpct", 0.45)
    h_mpg = espn.get("home_star_mpg", 32); a_mpg = espn.get("away_star_mpg", 32)

    overrides = {}
    # Market
    ov = overrides
    ov["espn_pregame_wp"] = espn.get("espn_pregame_wp", 0)
    ov["espn_pregame_wp_pbp"] = espn.get("espn_pregame_wp_pbp", 0)
    ov["implied_prob_home"] = round(impl_h, 4) if home_ml else 0
    ov["overround"] = round(impl_h + impl_a - 1, 4) if (home_ml and away_ml) else 0
    ov["ml_spread_dislocation"] = round(impl_h - (1/(1+10**(spread/8)) if spread else 0.5), 4) if home_ml else 0
    ov["home_fav"] = 1 if spread < 0 else 0

    # Line movement (from opening vs closing)
    sm = espn.get("_spread_move", 0); mm = espn.get("_ml_move", 0)
    ov["reverse_line_movement"] = 1 if (sm and mm and ((sm>0 and mm<0) or (sm<0 and mm>0))) else 0
    ov["line_reversal"] = 1 if abs(mm) > 0.03 else 0

    # Context
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        ov["is_midweek"] = 1 if dt.weekday() in [1,2,3] else 0
        ov["post_trade_deadline"] = 1 if dt >= datetime(dt.year, 2, 10) else 0
    except: pass
    ov["away_is_public_team"] = 1 if away_abbr in PUBLIC_TEAMS else 0
    cap = espn.get("_venue_cap", VENUE_CAPACITY.get(home_abbr, 19000))
    ov["crowd_pct"] = round(min(1.0, 18500/max(cap, 1)), 4)

    # H2H
    ov["h2h_total_games"] = espn.get("_h2h_n", 0)

    # Star players
    if h_star and a_star:
        ov["star1_share_diff"] = round(h_star/max(h_ppg,80) - a_star/max(a_ppg,80), 4)
        ov["lineup_value_diff"] = round(h_star*h_fg*2 - a_star*a_fg*2, 2)
    ov["star_minutes_fatigue_diff"] = round(h_mpg - a_mpg, 2)

    # ATS rolling
    ov["roll_ats_margin_gated"] = round(h_sr.get("ats_avg",0) - a_sr.get("ats_avg",0), 2)

    # ── Rolling PBP stats from pre-computed nba_team_rolling ──
    try:
        from nba_game_stats import get_rolling_diffs
        roll_diffs = get_rolling_diffs(home_abbr, away_abbr)
        if roll_diffs:
            # Map to model feature names
            feat_map = {
                "roll_bench_pts_diff": "roll_bench_pts_diff",
                "roll_paint_pts_diff": "roll_paint_pts_diff",
                "roll_fast_break_pts_diff": "roll_fast_break_diff",
                "roll_second_chance_pts_diff": "roll_second_chance_diff",
                "roll_largest_lead_diff": "roll_largest_lead_diff",
                "roll_q4_scoring_diff": "roll_q4_diff",
                "roll_three_fg_rate_diff": "roll_three_fg_rate_diff",
                "roll_ft_trip_rate_diff": "roll_ft_trip_rate_diff",
                "roll_max_run_avg": "roll_max_run_avg",
                "second_chance_x_oreb": "second_chance_x_oreb",
            }
            for src, dst in feat_map.items():
                if src in roll_diffs and roll_diffs[src]:
                    ov[dst] = roll_diffs[src]
            diag["sources"].append(f"Rolling PBP ({len(roll_diffs)} features)")
    except Exception as e:
        diag["warnings"].append(f"rolling PBP: {e}")

    # Apply
    for feat, val in ov.items():
        if feat in X.columns and val is not None:
            X.at[X.index[0], feat] = val

    X_slim = X[feature_list]

    # ═══ 9. Predict ═══
    X_s = bundle["scaler"].transform(X_slim)
    margin = float(bundle["model"].predict(X_s)[0])
    raw_p = 1.0 / (1.0 + np.exp(-margin / 8.0))
    cal = bundle.get("calibrator")
    wp = float(cal.predict([raw_p])[0]) if cal else raw_p

    # Contributions
    contribs = bundle["model"].coef_ * X_s[0]
    shap_out = sorted([
        {"feature": f, "shap": round(float(c), 4), "value": round(float(X_slim[f].iloc[0]), 3)}
        for f, c in zip(feature_list, contribs)
    ], key=lambda x: abs(x["shap"]), reverse=True)

    nz = sum(1 for s in shap_out if s["value"] != 0)
    mkt = float(row.get("market_spread_home", 0) or 0)

    return {
        "sport": "NBA", "game_id": game_id,
        "home_team": home_abbr, "away_team": away_abbr, "game_date": game_date,
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(wp, 4), "ml_win_prob_away": round(1-wp, 4),
        "pred_home_score": round(float(row.get("home_ppg",112))+margin/2, 1),
        "pred_away_score": round(float(row.get("away_ppg",112))-margin/2, 1),
        "market_spread": mkt, "market_total": float(row.get("market_ou_total",0) or 0),
        "disagree": round(abs(margin-(-mkt)), 2) if mkt else 0,
        "shap": shap_out[:20],
        "feature_coverage": f"{nz}/{len(feature_list)}",
        "model_meta": {
            "n_train": bundle.get("n_games"), "mae_cv": bundle.get("cv_mae"),
            "model_type": bundle.get("architecture","Lasso_solo_v26"),
            "n_features": len(feature_list), "has_isotonic": cal is not None,
        },
        "diagnostics": diag,
    }
