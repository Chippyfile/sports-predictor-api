"""
nba_full_predict.py — Server-side enriched NBA prediction (v27 Lasso)

ARCHITECTURE: Single ESPN summary call extracts ~50 raw values.
Supabase supplements enrichment + rolling PBP stats. Elo from local file.
Model: Lasso (alpha=0.1) regressor, 38 features selected by L1 from 69 candidates.
Feature builder (nba_v27_features_live.py) is the SINGLE SOURCE OF TRUTH for
feature computation. This file only injects supplementary external data
(rolling PBP diffs, enrichment diffs, ref profiles) AFTER the builder runs.

AUDIT v2 FIXES:
  CRIT-1: Rolling PBP direct diff injection (no fake component splitting)
  CRIT-2: Enrichment from Supabase tables (no fabricated per-team values)
  CRIT-3: Removed enrich_v2() on single rows (was zeroing 22 features)
  CRIT-4: Fixed override mappings (entropy≠HHI, drb_pct≠roll_dreb)
  CRIT-5: Fixed ref foul_rate (was copy of home_whistle)
  HIGH-3: ftpct_diff uses team FT% not star player FT%
  HIGH-5: Bundle feature validation logging
  HIGH-6: Removed _nba_backfill_heuristic on live rows
  MED-6:  Feature coverage warning guard

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
PUBLIC_TEAMS = {"LAL","LAC","GSW","BOS","NYK","BKN","CHI","MIA","PHX","DAL","PHI","MIL","OKC","DEN","CLE"}
NBA_CONFERENCES = {
    "ATL":"East","BOS":"East","BKN":"East","CHA":"East","CHI":"East","CLE":"East",
    "DET":"East","IND":"East","MIA":"East","MIL":"East","NYK":"East","ORL":"East",
    "PHI":"East","TOR":"East","WAS":"East",
    "DAL":"West","DEN":"West","GSW":"West","HOU":"West","LAC":"West","LAL":"West",
    "MEM":"West","MIN":"West","NOP":"West","OKC":"West","PHX":"West","POR":"West",
    "SAC":"West","SAS":"West","UTA":"West",
}
VENUE_CAPACITY = {
    "ATL":16600,"BOS":19156,"BKN":17732,"CHA":19077,"CHI":20917,"CLE":19432,
    "DAL":19200,"DEN":19520,"DET":20332,"GSW":18064,"HOU":18055,"IND":18165,
    "LAC":18997,"LAL":18997,"MEM":17794,"MIA":19600,"MIL":17341,"MIN":17136,
    "NOP":16867,"NYK":19812,"OKC":18203,"ORL":18846,"PHI":20478,"PHX":18055,
    "POR":19441,"SAC":17583,"SAS":18581,"TOR":19800,"UTA":18306,"WAS":20356,
}

def _map(a): return ESPN_ABBR_MAP.get(a, a)
# ESPN team ID map for season stats endpoint
_ESPN_TEAM_IDS = {
    "ATL":1,"BOS":2,"BKN":17,"CHA":30,"CHI":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GSW":9,"GS":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MEM":29,
    "MIA":14,"MIL":15,"MIN":16,"NOP":3,"NO":3,"NYK":18,"NY":18,"OKC":25,
    "ORL":19,"PHI":20,"PHX":21,"POR":22,"SAC":23,"SAS":24,"SA":24,
    "TOR":28,"UTA":26,"UTAH":26,"WAS":27,"WSH":27,
}

def _fetch_team_season_stats(abbr):
    """Fetch season stats from ESPN team statistics endpoint."""
    team_id = _ESPN_TEAM_IDS.get(abbr.upper())
    if not team_id:
        return {}
    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/statistics",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8
        )
        if not r.ok:
            return {}
        cats = r.json().get("results", {}).get("stats", {}).get("categories", [])
        result = {}
        # Map ESPN stat names → our column suffixes
        stat_map = {
            "threePointPct":          ("threepct", 100),      # divide by 100
            "threePointFieldGoalPct": ("threepct", 100),
            "fieldGoalPct":           ("fgpct", 100),
            "freeThrowPct":           ("ftpct", 100),
            "avgPoints":              ("ppg", 1),
            "avgPointsAgainst":       ("opp_ppg", 1),
            "avgPointsAllowed":       ("opp_ppg", 1),
            "oppPoints":              ("opp_ppg", 1),
            "avgSteals":              ("steals", 1),
            "avgTurnovers":           ("turnovers", 1),
            "avgTeamTurnovers":       ("turnovers", 1),
            "avgTotalTurnovers":      ("turnovers", 1),
            "avgBlocks":              ("blocks", 1),
            "avgAssists":             ("assists", 1),
            "avgRebounds":            ("total_reb", 1),
            "avgOffensiveRebounds":   ("oreb", 1),
            "avgDefensiveRebounds":   ("dreb", 1),
            "avgFieldGoalsAttempted":  ("fga", 1),
            "avgFreeThrowsAttempted":  ("fta", 1),
            "avgFieldGoalsMade":       ("fgm", 1),
            "avgThreePointFieldGoalsAttempted": ("three_att", 1),
            "avgThreePointFieldGoalsMade": ("three_made", 1),
            "fieldGoalsAttempted":     ("fga", 1),
            "freeThrowsAttempted":     ("fta", 1),
        }
        for cat in cats:
            for s in cat.get("stats", []):
                name = s.get("name", "")
                val = s.get("value")
                if val is None or name not in stat_map:
                    continue
                col_suffix, divisor = stat_map[name]
                # First match wins — don't overwrite with alternative stat names
                result.setdefault(col_suffix, round(float(val) / divisor, 4))
        # Derive TS% if we have the components: TS% = PPG / (2 * (FGA + 0.44 * FTA))
        if "ppg" in result and "fga" in result and "fta" in result:
            tsa = result["fga"] + 0.44 * result["fta"]
            if tsa > 0:
                result["ts_pct"] = round(result["ppg"] / (2 * tsa), 4)
        # Derive three_fg_rate: 3PA / FGA
        if "three_att" in result and "fga" in result and result["fga"] > 0:
            result["three_fg_rate"] = round(result["three_att"] / result["fga"], 4)
        # Derive net_rtg approximation from PPG - OPP_PPG
        if "ppg" in result and "opp_ppg" in result:
            result["net_rtg"] = round(result["ppg"] - result["opp_ppg"], 1)
        return result
    except Exception:
        return {}


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

def _parse_espn_summary(data, home_abbr, away_abbr, game_date_str=""):
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
            # Prefer DraftKings provider for consistent juice/movement data
            pc = next(
                (p for p in pc_list if "draft" in (p.get("provider", {}).get("name", "") or "").lower()),
                pc_list[0]
            )
            row["market_spread_home"] = pc.get("spread", 0) or 0
            row["market_ou_total"] = pc.get("overUnder", 0) or 0
            ho = pc.get("homeTeamOdds", {}) if isinstance(pc.get("homeTeamOdds"), dict) else {}
            ao = pc.get("awayTeamOdds", {}) if isinstance(pc.get("awayTeamOdds"), dict) else {}
            row["home_ml"] = ho.get("moneyLine", 0) or 0
            row["away_ml"] = ao.get("moneyLine", 0) or 0
            # v27 feature keys: moneyline + spread juice for implied_prob / juice_imbalance
            row["home_moneyline"]   = row["home_ml"]
            row["away_moneyline"]   = row["away_ml"]
            row["home_spread_odds"] = ho.get("spreadOdds", 0) or 0
            row["away_spread_odds"] = ao.get("spreadOdds", 0) or 0

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
            # v27 feature keys used by nba_v27_features_live.py
            row["spread_open"]  = open_sp or 0
            row["spread_close"] = close_sp or 0

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
                        elif isinstance(st, dict) and st.get("name") == "freeThrowPct":
                            try: row[f"{side}_ftpct_leader"] = float(st["value"]) / 100
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
            # FIX: statusType is a sibling of status, not nested inside it
            completed = sum(1 for ev in ss.get("events", [])
                           if _safe_get(ev, "statusType", "completed"))
            row["_h2h_n"] = completed
            margins = []
            for ev in ss.get("events", []):
                if not _safe_get(ev, "statusType", "completed"): continue
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
            # h2h_avg_margin: expose without underscore so it flows into row
            row["h2h_avg_margin"] = row["_h2h_margin"]
            # is_revenge_home: home team lost last completed meeting
            if margins:
                row["is_revenge_home"] = 1 if margins[-1] < 0 else 0
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
                        elif t == "streak": row[f"{side}_streak"] = _parse_streak(str(st.get("displayValue", "")))
                    except: pass
    except Exception as _e:
        diag.append(f"Section 6 (standings): {_e}")

    # ── 7. Last 5 games (rolling margins, rest, B2B) ──
    try:
        # Use game date (not server time) for rest/B2B calculations
        try:
            _game_dt = datetime.strptime(game_date_str or "", "%Y-%m-%d")
        except:
            _game_dt = datetime.now()

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
                    row[f"{side}_days_rest"] = max(0, (_game_dt - last).days - 1)
                    # B2B: team's last game was the day before this game
                    if (_game_dt - last).days <= 1:
                        row[f"{side}_is_b2b"] = 1
                    cutoff = _game_dt - timedelta(days=14)
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

    # ── 9. Venue + Header + Officials ──
    try:
        cap = _safe_get(data, "gameInfo", "venue", "capacity") or VENUE_CAPACITY.get(home_abbr, 19000)
        row["_venue_cap"] = cap

        # Officials (for ref_home_whistle)
        officials = data.get("gameInfo", {}).get("officials", [])
        ref_names = [o.get("displayName", "") for o in officials if o.get("displayName")]
        for i, name in enumerate(ref_names[:3]):
            row[f"_ref_{i+1}"] = name

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
    # Load directly from Supabase every time — bypasses per-worker in-memory cache
    # (gunicorn multi-process: each worker has own _models dict, can't share cache)
    # v27 model is 9KB so direct fetch is fast
    try:
        import requests as _req, base64 as _b64, io as _io, joblib as _jl
        from config import SUPABASE_URL, SUPABASE_KEY
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        resp = _req.get(
            f"{SUPABASE_URL}/rest/v1/model_store?name=eq.nba&select=data",
            headers=headers, timeout=30
        )
        if resp.ok:
            rows = resp.json()
            if rows and rows[0].get("data"):
                raw = _b64.b64decode(rows[0]["data"])
                return _jl.load(_io.BytesIO(raw))
    except Exception as e:
        print(f"  [nba] Supabase direct load failed: {e}")
    # No local fallback — Railway may have stale pkl from previous deploys
    # Local development: run python3 nba_v27_train.py to generate models/nba_v27.pkl
    return None

_ou_cache = None
_ou_cache_time = 0

def _load_ou_model():
    """Load NBA O/U model from Supabase model_store (cached 10 min)."""
    global _ou_cache, _ou_cache_time
    import time as _time
    if _ou_cache and (_time.time() - _ou_cache_time) < 600:
        return _ou_cache
    try:
        import requests as _req, base64 as _b64, io as _io, pickle as _pkl
        from config import SUPABASE_URL, SUPABASE_KEY
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        resp = _req.get(
            f"{SUPABASE_URL}/rest/v1/model_store?name=eq.nba_ou&select=data",
            headers=headers, timeout=30
        )
        if resp.ok:
            rows = resp.json()
            if rows and rows[0].get("data"):
                raw = _b64.b64decode(rows[0]["data"])
                _ou_cache = _pkl.loads(raw)
                _ou_cache_time = _time.time()
                return _ou_cache
    except Exception as e:
        print(f"  [nba_ou] load failed: {e}")
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
                    espn, espn_diag = _parse_espn_summary(raw, home_abbr, away_abbr, game_date)
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
    # Conference game flag
    row["conference_game"] = 1 if NBA_CONFERENCES.get(home_abbr) == NBA_CONFERENCES.get(away_abbr) else 0
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

    # ── Season stats from ESPN team endpoint FIRST (before df creation) ──
    # FIX: Was after df creation, so enrich_v2 couldn't see accurate shooting stats
    try:
        _ss_count = 0
        for side, abbr in [("home", home_abbr), ("away", away_abbr)]:
            season_stats = _fetch_team_season_stats(abbr)
            if season_stats:
                for stat, val in season_stats.items():
                    row[f"{side}_{stat}"] = val
                _ss_count += len(season_stats)
        if _ss_count:
            diag["sources"].append(f"Season stats ({_ss_count} stats)")
    except Exception as e:
        diag["warnings"].append(f"season_stats: {e}")

    # ── FIX HIGH-6: Do NOT call _nba_backfill_heuristic on live rows.
    # It's designed for historical training data backfill — on a single row it
    # overwrites ESPN-derived values with heuristic guesses.
    #
    # FIX CRIT-3: Do NOT call enrich_v2() on single rows.
    # enrich_v2 computes rolling window stats (trend, accel, skew, etc.)
    # requiring 10+ prior games. On a single row ALL rolling features → NaN/0,
    # creating massive train/serve skew.
    # Instead, enrichment comes from Supabase tables + direct diff injection.

    # ── Streak → after_loss flags ──
    try:
        h_streak_raw = espn.get("home_streak_raw", "")
        a_streak_raw = espn.get("away_streak_raw", "")
        if h_streak_raw.startswith("L"):
            row["home_after_loss"] = 1
        if a_streak_raw.startswith("L"):
            row["away_after_loss"] = 1
    except Exception as e:
        diag["warnings"].append(f"streak: {e}")

    # ── FIX: Key aliases (ESPN parser vs feature builder naming) ──
    row.setdefault("home_games_last_14", row.get("home_games_14d", 0))
    row.setdefault("away_games_last_14", row.get("away_games_14d", 0))

    # ═══ 8. Build v27 features ═══
    bundle = _load_model()
    if not bundle: return {"error": "NBA model not found"}
    # FIX HIGH-5: Log bundle info for verification
    feature_cols = bundle.get("feature_cols", bundle.get("feature_list", []))
    print(f"  [AUDIT] Bundle: {len(feature_cols)} features, "
          f"model_type={bundle.get('model_type')}, "
          f"trained_at={bundle.get('trained_at')}")

    from nba_v27_features_live import build_v27_features

    # ── FIX CRIT-1: Fetch rolling PBP diffs — store for direct override after build ──
    # Do NOT split diffs into fake per-team components (both sides get same avg → diff=0).
    # Instead store the pre-computed diffs and inject into feat_df AFTER build_v27_features.
    _roll_diffs = {}
    try:
        from nba_game_stats import get_rolling_diffs
        _roll_diffs = get_rolling_diffs(home_abbr, away_abbr) or {}
        if _roll_diffs:
            diag["sources"].append(f"Rolling PBP ({len(_roll_diffs)} features)")
    except Exception as e:
        diag["warnings"].append(f"rolling PBP: {e}")

    # ── FIX CRIT-2: Fetch enrichment from Supabase tables + live diff module ──
    # Do NOT fabricate per-team values from diffs with arbitrary centering.
    # Fetch per-team enrichment from nba_team_enrichment table first,
    # then store diffs separately for direct override injection.
    enrichment = {"home": {}, "away": {}}
    _enrich_diffs = {}
    try:
        from db import sb_get
        for side, abbr in [("home", home_abbr), ("away", away_abbr)]:
            rows_enr = sb_get("nba_team_enrichment",
                              f"team_abbr=eq.{abbr}&order=updated_at.desc&limit=1") or []
            if rows_enr:
                enrichment[side] = rows_enr[0]
                # AUDIT-v3: Check enrichment staleness
                _updated = rows_enr[0].get("updated_at", "")
                if _updated:
                    try:
                        _enr_dt = datetime.fromisoformat(_updated.replace("Z", "+00:00").replace("+00:00", ""))
                        _age_hrs = (datetime.utcnow() - _enr_dt).total_seconds() / 3600
                        if _age_hrs > 48:
                            diag["warnings"].append(f"STALE enrichment for {abbr}: {_age_hrs:.0f}hrs old")
                    except Exception:
                        pass
        n_enr = sum(len(v) for v in enrichment.values())
        if n_enr:
            diag["sources"].append(f"Supabase enrichment ({n_enr} fields)")
    except Exception as e:
        diag["warnings"].append(f"enrichment_supabase: {e}")

    # Also try live-computed diffs (for direct override after build)
    try:
        from nba_enrichment import get_enrichment_diffs
        _enrich_diffs = get_enrichment_diffs(home_abbr, away_abbr) or {}
        if _enrich_diffs:
            diag["sources"].append(f"Enrichment diffs ({len(_enrich_diffs)} features)")
        # If Supabase enrichment was empty, try per-team module
        if not enrichment["home"] and not enrichment["away"]:
            try:
                from nba_enrichment import get_team_enrichment
                h_enr = get_team_enrichment(home_abbr) or {}
                a_enr = get_team_enrichment(away_abbr) or {}
                if h_enr: enrichment["home"] = h_enr
                if a_enr: enrichment["away"] = a_enr
            except ImportError:
                pass
    except Exception as e:
        diag["warnings"].append(f"enrichment_diffs: {e}")

    # ── Referee lookup: official.nba.com (posted ~9AM ET) ──
    ref_profile = {}
    try:
        from nba_ref_scraper import get_refs_for_game
        scraped = get_refs_for_game(home_abbr, away_abbr)
        if scraped:
            for k, v in scraped.items():
                espn[f"_{k}"] = v  # _ref_1, _ref_2, _ref_3
            from db import sb_get
            all_refs = sb_get("nba_ref_profiles", "select=ref_name,home_whistle,avg_home_margin,avg_foul_rate,ou_bias,pace_impact&limit=100") or []
            ref_map = {r["ref_name"]: r for r in all_refs}
            matched = [ref_map[n] for n in scraped.values() if n in ref_map]
            if matched:
                ref_profile = {
                    "home_whistle": sum(r["home_whistle"] for r in matched) / len(matched),
                    # FIX CRIT-5: Use actual foul rate, not home_whistle copy
                    "foul_rate":    sum(r.get("avg_foul_rate", r["home_whistle"]) for r in matched) / len(matched),
                    # Pass ou_bias + pace_impact so all 4 ref features populate
                    "ou_bias":      sum(r.get("ou_bias", 0) for r in matched) / len(matched),
                    "pace_impact":  sum(r.get("pace_impact", 0) for r in matched) / len(matched),
                }
                diag["sources"].append(f"Refs: {', '.join(scraped.values())} ({len(matched)} profiled)")
            else:
                diag["sources"].append(f"Refs: {', '.join(scraped.values())} (no profiles yet)")
    except Exception as e:
        diag["warnings"].append(f"ref_scraper: {e}")

    # Get dynamic league average TS% (rolling 1-2 seasons from nba_historical)
    _lg_ts = 0.575  # static fallback
    try:
        from dynamic_constants import compute_nba_league_averages, NBA_DEFAULT_AVERAGES
        _nba_avgs = compute_nba_league_averages() or NBA_DEFAULT_AVERAGES
        _lg_ts = _nba_avgs.get("ts_pct", 0.575)
    except ImportError:
        pass

    feat_df = build_v27_features(row, enrichment=enrichment, ref_profile=ref_profile, league_avg_ts=_lg_ts)

    # Select only the features this model was trained on (feature_cols set above)
    for f in feature_cols:
        if f not in feat_df.columns:
            feat_df[f] = 0.0
    X = feat_df[feature_cols]

    # ESPN-derived values used by overrides block below
    spread = row.get("market_spread_home", 0) or 0
    home_ml = row.get("home_ml", 0) or row.get("home_moneyline", 0) or espn.get("home_ml", 0)
    away_ml = row.get("away_ml", 0) or row.get("away_moneyline", 0) or espn.get("away_ml", 0)
    impl_h = _ml_to_prob(home_ml)
    impl_a = _ml_to_prob(away_ml)
    h_star = espn.get("home_star1_ppg", 0); a_star = espn.get("away_star1_ppg", 0)
    h_ppg = row.get("home_ppg", 112); a_ppg = row.get("away_ppg", 112)
    h_fg = espn.get("home_star1_fgpct", 0.45); a_fg = espn.get("away_star1_fgpct", 0.45)
    h_mpg = espn.get("home_star_mpg", 32); a_mpg = espn.get("away_star_mpg", 32)

    overrides = {}
    # Market
    ov = overrides
    # FIX AUDIT-v3: Only override ESPN WP if ESPN actually returned predictor data.
    # Use `is not None` check — 0.0 is a valid (if unlikely) win probability.
    _espn_wp = espn.get("espn_pregame_wp")
    _espn_wp_pbp = espn.get("espn_pregame_wp_pbp")
    if _espn_wp is not None and _espn_wp > 0.01:
        ov["espn_pregame_wp"] = _espn_wp
    if _espn_wp_pbp is not None and _espn_wp_pbp > 0.01:
        ov["espn_pregame_wp_pbp"] = _espn_wp_pbp
    ov["implied_prob_home"] = round(impl_h, 4) if home_ml else 0
    ov["overround"] = round(impl_h + impl_a - 1, 4) if (home_ml and away_ml) else 0
    ov["ml_spread_dislocation"] = round(impl_h - (1/(1+10**(spread/8)) if spread else 0.5), 4) if home_ml else 0
    ov["home_fav"] = 1 if spread < 0 else 0

    # public_home_spread_pct: proxy for public money direction
    # DK ML implied prob is influenced by public money; ESPN predictor is model-based
    # Positive = public favoring home more than model suggests
    espn_wp = espn.get("espn_pregame_wp", 0) or row.get("espn_pregame_wp", 0.5)
    if home_ml and espn_wp and espn_wp > 0:
        dk_implied = impl_h / max(impl_h + impl_a, 0.01)  # vig-removed DK prob
        ov["public_home_spread_pct"] = round(dk_implied - espn_wp, 4)

    # Line movement (from opening vs closing)
    # Sign conventions: spread more negative = home favored, ML prob higher = home favored
    # So concordant movement gives sm<0 & mm>0 (or sm>0 & mm<0) → sm*mm < 0
    # True reversal: one moves toward home, other away → sm*mm > 0
    sm = espn.get("_spread_move", 0); mm = espn.get("_ml_move", 0)
    ov["reverse_line_movement"] = 1 if (sm and mm and sm * mm > 0) else 0
    ov["line_reversal"] = round(abs(mm), 4)  # continuous, training mean ~0.064
    ov["sharp_spread_signal"] = round(sm, 2)  # spread movement (close - open)
    ov["sharp_ml_signal"] = round(mm, 4)      # ML prob movement (close - open)

    # ── Team stat overrides REMOVED (AUDIT-v3) ──
    # efg_diff, turnovers_diff, win_pct_diff, ftpct_diff, b2b_diff, games_last_14_diff,
    # games_diff, home_after_loss, away_after_loss are ALL computed by the builder
    # from the same row data. Recomputing here with different fallback defaults
    # (0.471 vs 0.46, 14 vs builder's row-derived value) creates silent train/serve skew.
    # The builder (nba_v27_features_live.py) is now the SINGLE SOURCE OF TRUTH.

    # ── Schedule context (builder may not have these if date parse fails) ──
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        ov["is_midweek"] = 1 if dt.weekday() in [1,2,3] else 0
        ov["post_trade_deadline"] = 1 if dt >= datetime(dt.year, 2, 10) else 0
    except: pass
    ov["away_is_public_team"] = 1 if away_abbr in PUBLIC_TEAMS else 0
    cap = espn.get("_venue_cap", VENUE_CAPACITY.get(home_abbr, 19000))
    ov["crowd_pct"] = round(min(1.0, 18500/max(cap, 1)), 4)

    # H2H (from ESPN season series — builder reads from row which has it)
    _h2h_n = espn.get("_h2h_n", 0)
    if _h2h_n:
        ov["h2h_total_games"] = _h2h_n

    # Star players → lineup_value_diff (enrichment may be stale, ESPN stars are fresh)
    if h_star and a_star:
        ov["lineup_value_diff"] = round(h_star*h_fg*2 - a_star*a_fg*2, 2)

    # ── Elo (FIXED: training uses home_form/away_form, not raw Elo) ──
    # AUDIT-v3 diagnostic: log which elo source was used
    h_form = float(row.get("home_form", 0))
    a_form = float(row.get("away_form", 0))
    _elo_source = "none"
    if h_form != 0 or a_form != 0:
        ov["elo_diff"] = round(h_form - a_form, 4)
        _elo_source = f"form(h={h_form:.3f},a={a_form:.3f})"
    else:
        # Fallback: normalize raw elo to training scale (-2 to +2)
        _raw = h_elo - a_elo
        ov["elo_diff"] = round(float(np.clip(_raw / 200, -2, 2)), 4)
        _elo_source = f"raw_elo(h={h_elo},a={a_elo},norm={ov['elo_diff']:.3f})"
    print(f"  [AUDIT] elo_diff={ov['elo_diff']:.4f} source={_elo_source}")

    # ATS rolling
    ov["roll_ats_margin_gated"] = round(h_sr.get("ats_avg",0) - a_sr.get("ats_avg",0), 2)

    # ── FIX CRIT-1: Rolling PBP direct diff injection ──
    # Inject pre-computed diffs directly — no component splitting needed.
    _roll_direct_map = {
        "roll_bench_pts_diff":      "roll_bench_pts_diff",
        "roll_paint_pts_diff":      "roll_paint_pts_diff",
        "roll_fast_break_pts_diff": "roll_fast_break_diff",
        "roll_ft_trip_rate_diff":   "roll_ft_trip_rate_diff",
        "roll_three_fg_rate_diff":  "roll_three_fg_rate_diff",
        "roll_q4_scoring_diff":     "roll_q4_diff",
        "roll_max_run_avg":         "roll_max_run_avg",
        "roll_dreb_diff":           "roll_dreb_diff",
        "roll_paint_fg_rate_diff":  "roll_paint_fg_rate_diff",
    }
    for src_key, model_feat in _roll_direct_map.items():
        val = _roll_diffs.get(src_key)
        if val is not None:
            ov[model_feat] = float(val)

    # ── FIX CRIT-4: Enrichment direct diff injection ──
    # Correct feature name mappings (was: entropy→HHI, drb_pct→roll_dreb — both wrong)
    _enrich_direct_map = {
        "scoring_entropy_diff":     "scoring_entropy_diff",
        "scoring_hhi_diff":         "scoring_hhi_diff",
        "ceiling_diff":             "ceiling_diff",
        "floor_diff":               "floor_diff",
        "consistency_diff":         "consistency_diff",
        "bimodal_diff":             "bimodal_diff",
        "opp_suppression_diff":     "opp_suppression_diff",
        "score_kurtosis_diff":      "score_kurtosis_diff",
        "margin_accel_diff":        "margin_accel_diff",
        "momentum_halflife_diff":   "momentum_halflife_diff",
        "win_aging_diff":           "win_aging_diff",
        "pyth_residual_diff":       "pyth_residual_diff",
        "pyth_luck_diff":           "pyth_luck_diff",
        "recovery_diff":            "recovery_diff",
        "pace_control_diff":        "pace_control_diff",
        "pace_leverage":            "pace_leverage",
        "three_value_diff":         "three_value_diff",
        "ts_regression_diff":       "ts_regression_diff",
        "three_pt_regression_diff": "three_pt_regression_diff",
        "matchup_efg":              "matchup_efg",
        "matchup_ft":               "matchup_ft",
        "matchup_orb":              "matchup_orb",
        "lineup_value_diff":        "lineup_value_diff",
    }
    for src_key, model_feat in _enrich_direct_map.items():
        val = _enrich_diffs.get(src_key)
        if val is not None:
            ov[model_feat] = float(val)

    # ── Interaction features ──
    # clutch_x_tight_spread: Q4 scoring diff (clutch proxy) × tight spread indicator
    q4_diff = ov.get("roll_q4_diff", 0)
    if q4_diff and spread:
        tight = 1.0 if abs(spread) <= 5 else 0.5 if abs(spread) <= 8 else 0.0
        ov["clutch_x_tight_spread"] = round(q4_diff * tight, 3)

    # ── Referee features — fallback when scraper didn't run ──
    # ESPN parser puts ref names in row["_ref_1"], row["_ref_2"], row["_ref_3"]
    # If scraper already built ref_profile, skip. Otherwise do full Supabase lookup.
    ref_names = [espn.get(f"_ref_{i}", "") for i in range(1, 4)]
    if any(ref_names) and not ref_profile.get("home_whistle"):
        try:
            from db import sb_get
            all_refs = sb_get("nba_ref_profiles",
                              "select=ref_name,home_whistle,avg_foul_rate,ou_bias,pace_impact&limit=100") or []
            ref_map = {r["ref_name"]: r for r in all_refs}
            matched = [ref_map[n] for n in ref_names if n and n in ref_map]
            if matched:
                ov["ref_home_whistle"] = sum(r["home_whistle"] for r in matched) / len(matched)
                ov["ref_foul_proxy"] = sum(r.get("avg_foul_rate", r["home_whistle"]) for r in matched) / len(matched)
                ov["ref_ou_bias"] = sum(r.get("ou_bias", 0) for r in matched) / len(matched)
                ov["ref_pace_impact"] = sum(r.get("pace_impact", 0) for r in matched) / len(matched)
                diag["sources"].append(f"Refs: {', '.join(n for n in ref_names if n)} ({len(matched)} profiled via ESPN)")
            else:
                diag["sources"].append(f"Refs: {', '.join(n for n in ref_names if n)} (no profiles found)")
        except Exception as e:
            diag["warnings"].append(f"ref_fallback: {e}")

    # Apply ESPN overrides onto v27 feat_df (cast to float to avoid dtype errors)
    feat_df = feat_df.astype(float)
    for feat, val in ov.items():
        if feat in feat_df.columns and val is not None:
            try:
                feat_df.at[feat_df.index[0], feat] = float(val)
            except Exception:
                pass

    # Build X from verified v27 feature set
    X_slim = feat_df[feature_cols]

    # ═══ 9. Predict ═══
    X_s = bundle["scaler"].transform(X_slim)
    reg = bundle.get("reg", bundle.get("model"))
    margin = float(reg.predict(X_s)[0])
    clf = bundle.get("clf")
    if clf:
        raw_p = float(clf.predict_proba(X_s)[0][1])
    else:
        raw_p = 1.0 / (1.0 + np.exp(-margin / 8.0))
    cal = bundle.get("calibrator") or bundle.get("isotonic")
    wp = float(cal.predict([raw_p])[0]) if cal else raw_p

    # Contributions
    try:
        reg_model = bundle.get("reg", bundle.get("model"))
        contribs = reg_model.coef_ * X_s[0]
        shap_out = sorted([
            {"feature": f, "shap": round(float(c), 4), "value": round(float(X_slim[f].iloc[0]), 3)}
            for f, c in zip(feature_cols, contribs)
        ], key=lambda x: abs(x["shap"]), reverse=True)
    except Exception as e:
        shap_out = []
        diag["warnings"].append(f"shap: {e}")

    nz = sum(1 for s in shap_out if s["value"] != 0)
    # FIX MED-6: Warn on low feature coverage
    coverage_pct = nz / len(feature_cols) if feature_cols else 0
    if coverage_pct < 0.50:
        diag["warnings"].append(
            f"LOW COVERAGE: {nz}/{len(feature_cols)} features non-zero ({coverage_pct:.0%})")
    # AUDIT-v3: Always log feature coverage for monitoring
    print(f"  [AUDIT] Features: {nz}/{len(feature_cols)} non-zero ({coverage_pct:.0%}), "
          f"margin={margin:+.2f}, wp={wp:.4f}")
    mkt = float(row.get("market_spread_home", 0) or 0)

    # Debug: return all features if requested
    debug = game.get("debug", False)

    # ═══ O/U TOTAL MODEL (separate CatBoost, same features) ═══
    ou_predicted_total = None
    ou_edge = None
    ou_pick = None
    try:
        ou_bundle = _load_ou_model()
        if ou_bundle and ou_bundle.get("reg") and ou_bundle.get("scaler"):
            ou_feature_cols = ou_bundle.get("ou_feature_cols", feature_cols)
            # AUDIT-v3 CRIT-3: Warn if O/U model is using spread model features
            if "ou_feature_cols" not in ou_bundle:
                diag["warnings"].append("O/U model missing ou_feature_cols — using spread model features (may hurt accuracy)")
            available_ou = [f for f in ou_feature_cols if f in feature_cols or f in row]
            if len(available_ou) >= len(ou_feature_cols) * 0.7:
                ou_vals = {}
                for f in ou_feature_cols:
                    if f in feature_cols:
                        ou_vals[f] = float(X_slim[f].iloc[0])
                    elif f in row:
                        ou_vals[f] = float(row.get(f, 0) or 0)
                    else:
                        ou_vals[f] = 0.0
                X_ou = pd.DataFrame([ou_vals])[ou_feature_cols]
                X_ou_s = ou_bundle["scaler"].transform(X_ou)
                ou_predicted_total = float(ou_bundle["reg"].predict(X_ou_s)[0])
                ou_bias = ou_bundle.get("bias_correction", 0)
                ou_predicted_total += ou_bias
                mkt_ou = float(row.get("market_ou_total", 0) or 0)
                if mkt_ou > 0:
                    ou_edge = round(ou_predicted_total - mkt_ou, 1)
                    if abs(ou_edge) >= 3:
                        ou_pick = "OVER" if ou_edge > 0 else "UNDER"
                diag["sources"].append(f"O/U model ({len(ou_feature_cols)} features)")
    except Exception as e:
        diag["warnings"].append(f"ou_model: {e}")

    return {
        "sport": "NBA", "game_id": game_id,
        "home_team": home_abbr, "away_team": away_abbr, "game_date": game_date,
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(wp, 4), "ml_win_prob_away": round(1-wp, 4),
        "pred_home_score": round(float(row.get("home_ppg",112))+margin/2, 1),
        "pred_away_score": round(float(row.get("away_ppg",112))-margin/2, 1),
        "market_spread": mkt, "market_total": float(row.get("market_ou_total",0) or 0),
        "disagree": round(abs(margin-(-mkt)), 2) if mkt else 0,
        "shap": shap_out,  # all 38 features
        "feature_coverage": f"{nz}/{len(feature_cols)}",
        "model_meta": {
            "n_train": bundle.get("n_games"), "mae_cv": bundle.get("cv_mae"),
            "model_type": bundle.get("model_type", bundle.get("architecture","unknown")),
            "n_features": len(feature_cols), "has_isotonic": cal is not None,
        },
        "diagnostics": diag,
        "ou_predicted_total": round(ou_predicted_total, 1) if ou_predicted_total else None,
        "ou_edge": ou_edge,
        "ou_pick": ou_pick,
    }
