"""
nba_full_predict.py — Server-side enriched NBA prediction (v26 Lasso)

Supabase-first approach:
  1. Look up game row in nba_predictions (59 columns from nbaSync.js)
  2. Add Elo ratings from local nba_elo_ratings.json
  3. Add ESPN pickcenter (spread/total/ML/win probability) if not already present
  4. Fetch recent games from Supabase for rolling enrichment features
  5. Run heuristic backfill + enrich_v2 + feature builder + Lasso v26

Route: POST /predict/nba/full
Input: {"game_id": "..."} or {"home_team": "BOS", "away_team": "LAL", "game_date": "2026-03-23"}
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import requests
import traceback as _tb
from datetime import datetime, timedelta

# ── Team mappings ──
NBA_ESPN_IDS = {
    "ATL":1,"BOS":2,"BKN":17,"CHA":30,"CHI":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GSW":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MEM":29,"MIA":14,
    "MIL":15,"MIN":16,"NOP":3,"NYK":18,"OKC":25,"ORL":19,"PHI":20,"PHX":21,
    "POR":22,"SAC":23,"SAS":24,"TOR":28,"UTA":26,"WAS":27,
}
ESPN_ABBR_MAP = {
    "GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS",
    "WSH":"WAS","UTAH":"UTA","UTH":"UTA","PHO":"PHX","BKLYN":"BKN","BK":"BKN",
}

def _map(a):
    return ESPN_ABBR_MAP.get(a, a)


# ═════════════════════════════════════════════════════════════
# DATA LOOKUPS
# ═════════════════════════════════════════════════════════════

def _supabase_headers():
    from config import SUPABASE_URL, SUPABASE_KEY
    return (
        SUPABASE_URL,
        {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
    )


def _lookup_game_row(game_id=None, home_team=None, away_team=None, game_date=None):
    """Look up a game from Supabase nba_predictions. Returns dict or None."""
    url, headers = _supabase_headers()
    if game_id:
        q = f"{url}/rest/v1/nba_predictions?game_id=eq.{game_id}&select=*&limit=1"
    elif home_team and away_team and game_date:
        q = (f"{url}/rest/v1/nba_predictions"
             f"?home_team=eq.{home_team}&away_team=eq.{away_team}"
             f"&game_date=eq.{game_date}&select=*&limit=1")
    else:
        return None
    try:
        resp = requests.get(q, headers=headers, timeout=10)
        rows = resp.json() if resp.ok else []
        return rows[0] if rows else None
    except Exception:
        return None


def _lookup_recent_games(team_abbr, before_date, n=15):
    """Fetch team's recent completed games from Supabase."""
    url, headers = _supabase_headers()
    games = []
    for side, opp_side, sign in [("home_team", "away_team", 1), ("away_team", "home_team", -1)]:
        q = (f"{url}/rest/v1/nba_predictions"
             f"?result_entered=eq.true&{side}=eq.{team_abbr}"
             f"&game_date=lt.{before_date}"
             f"&order=game_date.desc&limit={n}"
             f"&select=game_date,home_team,away_team,actual_home_score,actual_away_score,"
             f"market_spread_home")
        try:
            resp = requests.get(q, headers=headers, timeout=10)
            rows = resp.json() if resp.ok else []
        except Exception:
            rows = []
        for r in rows:
            hs = r.get("actual_home_score") or 0
            aws = r.get("actual_away_score") or 0
            margin = (hs - aws) * sign
            sp = (r.get("market_spread_home") or 0) * sign
            opp = r.get(opp_side, "")
            games.append({
                "date": r["game_date"], "margin": margin, "won": margin > 0,
                "spread": sp, "opponent": opp,
            })
    games.sort(key=lambda g: g["date"], reverse=True)
    return games[:n]


def _fetch_espn_pickcenter(game_id):
    """Fetch spread/total/ML/win_prob from ESPN summary."""
    if not game_id:
        return {}
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            return {}
        data = r.json()
    except Exception:
        return {}
    result = {}
    for pc in data.get("pickcenter", []):
        result["market_spread_home"] = pc.get("spread", 0) or 0
        result["market_ou_total"] = pc.get("overUnder", 0) or 0
        ho = pc.get("homeTeamOdds", {})
        ao = pc.get("awayTeamOdds", {})
        result["home_ml_close"] = ho.get("moneyLine", 0) or 0
        result["away_ml_close"] = ao.get("moneyLine", 0) or 0
        break
    wp = data.get("winprobability", [])
    if wp:
        result["espn_pregame_wp"] = wp[0].get("homeWinPercentage", 0.5)
        result["espn_pregame_wp_pbp"] = wp[0].get("homeWinPercentage", 0.5)
    return result


def _find_game_id_from_scoreboard(home_abbr, away_abbr, game_date):
    """Discover game_id from ESPN scoreboard."""
    ds = game_date.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ds}&limit=50"
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            return None
        for ev in r.json().get("events", []):
            comp = ev.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            h = _map(home.get("team", {}).get("abbreviation", ""))
            a = _map(away.get("team", {}).get("abbreviation", ""))
            if h == home_abbr and a == away_abbr:
                return str(ev.get("id", ""))
    except Exception:
        pass
    return None


def _load_elo():
    for path in ["nba_elo_ratings.json", "models/nba_elo_ratings.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("ratings", {})
    return {}


def _load_model():
    for path in ["nba_model_local.pkl", "models/nba_model_local.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


# ═════════════════════════════════════════════════════════════
# ROLLING STATS FROM SUPABASE HISTORY
# ═════════════════════════════════════════════════════════════

def _compute_rolling(recent):
    """Compute rolling features from recent game list."""
    if not recent:
        return {}
    margins = [g["margin"] for g in recent[:10]]
    last5 = recent[:5]

    wins = sum(1 for g in recent if g["won"])
    win_pct = wins / max(len(recent), 1)

    # Streak
    streak = 0
    if recent:
        d = recent[0]["won"]
        for g in recent:
            if g["won"] == d: streak += 1
            else: break
        if not d: streak = -streak

    # Form (recency-weighted)
    form = sum((1 if g["won"] else -1) * (len(last5) - i)
               for i, g in enumerate(last5)) / 15.0 if last5 else 0

    # ATS rolling
    ats_margins = [g["margin"] + g["spread"] for g in recent[:10] if g.get("spread", 0) != 0]

    # Days rest
    days_rest = 2
    if recent:
        try:
            last_d = datetime.strptime(recent[0]["date"], "%Y-%m-%d")
            days_rest = max(0, (datetime.now() - last_d).days - 1)
        except: pass

    return {
        "win_pct": round(win_pct, 4),
        "wins": wins, "losses": len(recent) - wins,
        "streak": streak,
        "form": round(form, 4),
        "margin_mean": round(float(np.mean(margins)), 2) if margins else 0,
        "margin_std": round(float(np.std(margins)), 2) if len(margins) >= 3 else 12.0,
        "days_rest": days_rest,
        "ats_margin_avg": round(float(np.mean(ats_margins)), 2) if ats_margins else 0,
        "games_played": len(recent),
    }


# ═════════════════════════════════════════════════════════════
# MAIN PREDICTION
# ═════════════════════════════════════════════════════════════

def predict_nba_full(game: dict):
    """Full server-side NBA prediction with Supabase-first data enrichment."""

    game_id = game.get("game_id")
    home_abbr = _map(game.get("home_team", ""))
    away_abbr = _map(game.get("away_team", ""))
    game_date = game.get("game_date", datetime.now().strftime("%Y-%m-%d"))

    diag = {"sources": [], "warnings": []}

    # ═══ STEP 1: Look up game row in Supabase ═══
    sb_row = _lookup_game_row(game_id=game_id, home_team=home_abbr,
                               away_team=away_abbr, game_date=game_date)

    if sb_row:
        n_filled = len([v for v in sb_row.values() if v is not None])
        diag["sources"].append(f"Supabase nba_predictions ({n_filled} cols)")
        home_abbr = sb_row.get("home_team", home_abbr)
        away_abbr = sb_row.get("away_team", away_abbr)
        game_id = sb_row.get("game_id", game_id)
        game_date = sb_row.get("game_date", game_date)
    else:
        diag["warnings"].append("Game not found in Supabase — using ESPN + defaults")

    if not home_abbr or not away_abbr:
        return {"error": "home_team and away_team required (or game_id)"}

    # ═══ STEP 2: Find game_id if missing ═══
    if not game_id:
        game_id = _find_game_id_from_scoreboard(home_abbr, away_abbr, game_date)
        if game_id:
            diag["sources"].append(f"game_id={game_id} from scoreboard")

    # ═══ STEP 3: ESPN pickcenter (only if Supabase doesn't have market data) ═══
    has_market = sb_row and sb_row.get("market_spread_home") and sb_row["market_spread_home"] != 0
    espn_mkt = {}
    if not has_market and game_id:
        espn_mkt = _fetch_espn_pickcenter(game_id)
        if espn_mkt.get("market_spread_home"):
            diag["sources"].append("ESPN pickcenter (live)")

    # ═══ STEP 4: Elo ratings ═══
    elo = _load_elo()
    h_elo = elo.get(home_abbr, 1500)
    a_elo = elo.get(away_abbr, 1500)
    if elo:
        diag["sources"].append("Elo ratings")

    # ═══ STEP 5: Recent games from Supabase (rolling enrichment) ═══
    h_recent = _lookup_recent_games(home_abbr, game_date)
    a_recent = _lookup_recent_games(away_abbr, game_date)
    h_roll = _compute_rolling(h_recent)
    a_roll = _compute_rolling(a_recent)
    if h_recent:
        diag["sources"].append(f"Rolling: {home_abbr} ({len(h_recent)} games)")
    if a_recent:
        diag["sources"].append(f"Rolling: {away_abbr} ({len(a_recent)} games)")

    # ═══ STEP 6: Build unified row ═══
    row = {}

    # Start with Supabase data (59 columns)
    if sb_row:
        for k, v in sb_row.items():
            if v is not None and k != "id":
                row[k] = v

    # ESPN pickcenter overrides (only fills gaps)
    if espn_mkt:
        for k, v in espn_mkt.items():
            if v and (k not in row or not row.get(k)):
                row[k] = v

    # Elo
    row["home_elo"] = h_elo
    row["away_elo"] = a_elo
    row["elo_diff"] = h_elo - a_elo

    # Core fields
    row["game_date"] = game_date
    row["home_team"] = home_abbr
    row["away_team"] = away_abbr
    row["season"] = 2026

    # Rolling stats fill gaps
    if not row.get("home_wins"):
        row["home_wins"] = h_roll.get("wins", 20)
        row["home_losses"] = h_roll.get("losses", 20)
    if not row.get("away_wins"):
        row["away_wins"] = a_roll.get("wins", 20)
        row["away_losses"] = a_roll.get("losses", 20)
    if not row.get("home_form"):
        row["home_form"] = h_roll.get("form", 0)
        row["away_form"] = a_roll.get("form", 0)
    if row.get("home_days_rest") is None:
        row["home_days_rest"] = h_roll.get("days_rest", 2)
        row["away_days_rest"] = a_roll.get("days_rest", 2)

    # Rolling margin context for enrichment
    row.setdefault("home_margin_trend", h_roll.get("margin_mean", 0))
    row.setdefault("away_margin_trend", a_roll.get("margin_mean", 0))
    row.setdefault("home_streak", h_roll.get("streak", 0))
    row.setdefault("away_streak", a_roll.get("streak", 0))
    row.setdefault("home_scoring_var", h_roll.get("margin_std", 12))
    row.setdefault("away_scoring_var", a_roll.get("margin_std", 12))

    # Defaults for anything still missing
    defaults = {
        "home_ppg": 112, "away_ppg": 112,
        "home_opp_ppg": 112, "away_opp_ppg": 112,
        "home_fgpct": 0.471, "away_fgpct": 0.471,
        "home_threepct": 0.365, "away_threepct": 0.365,
        "home_ftpct": 0.78, "away_ftpct": 0.78,
        "home_tempo": 100, "away_tempo": 100,
        "home_net_rtg": 0, "away_net_rtg": 0,
        "home_orb_pct": 0.25, "away_orb_pct": 0.25,
        "home_fta_rate": 0.28, "away_fta_rate": 0.28,
        "home_ato_ratio": 1.8, "away_ato_ratio": 1.8,
        "home_opp_fgpct": 0.471, "away_opp_fgpct": 0.471,
        "home_opp_threepct": 0.365, "away_opp_threepct": 0.365,
        "home_assists": 25, "away_assists": 25,
        "home_turnovers": 14, "away_turnovers": 14,
        "home_steals": 7.5, "away_steals": 7.5,
        "home_blocks": 5.0, "away_blocks": 5.0,
        "market_spread_home": 0, "market_ou_total": 228,
        "win_pct_home": 0.5,
        "pred_home_score": 112, "pred_away_score": 112,
        "espn_pregame_wp": 0.5, "espn_pregame_wp_pbp": 0.5,
        "away_travel_dist": 0, "model_ml_home": 0,
    }
    for k, v in defaults.items():
        row.setdefault(k, v)

    # ═══ STEP 7: Pipeline — backfill + enrich + features + predict ═══
    df = pd.DataFrame([row])

    try:
        from sports.nba import _nba_backfill_heuristic
        df = _nba_backfill_heuristic(df)
    except Exception as e:
        diag["warnings"].append(f"backfill: {e}")

    try:
        from enrich_nba_v2 import enrich as enrich_v2
        df = enrich_v2(df)
        diag["sources"].append("enrich_v2")
    except Exception as e:
        diag["warnings"].append(f"enrich_v2: {e}")

    # ═══ STEP 8: Build features + predict ═══
    bundle = _load_model()
    if not bundle:
        return {"error": "NBA model not found (nba_model_local.pkl)"}

    try:
        from nba_build_features_v25 import nba_build_features
    except ImportError:
        from sports.nba import nba_build_features

    X = nba_build_features(df)

    feature_list = bundle["feature_list"]
    for f in feature_list:
        if f not in X.columns:
            X[f] = 0.0
    X_slim = X[feature_list]

    X_s = bundle["scaler"].transform(X_slim)
    margin = float(bundle["model"].predict(X_s)[0])

    # Win probability
    raw_prob = 1.0 / (1.0 + np.exp(-margin / 8.0))
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        try:
            win_prob = float(calibrator.predict([raw_prob])[0])
        except Exception:
            win_prob = raw_prob
    else:
        win_prob = raw_prob

    # Feature contributions
    coefs = bundle["model"].coef_
    scaled_vals = X_s[0]
    contributions = coefs * scaled_vals
    shap_out = [
        {"feature": f, "shap": round(float(c), 4), "value": round(float(X_slim[f].iloc[0]), 3)}
        for f, c in zip(feature_list, contributions)
    ]
    shap_out.sort(key=lambda x: abs(x["shap"]), reverse=True)

    nonzero = sum(1 for s in shap_out if s["value"] != 0)
    h_ppg = float(row.get("home_ppg", 112))
    a_ppg = float(row.get("away_ppg", 112))
    mkt_spread = float(row.get("market_spread_home", 0) or 0)

    return {
        "sport": "NBA",
        "game_id": game_id,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "game_date": game_date,
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "pred_home_score": round(h_ppg + margin / 2, 1),
        "pred_away_score": round(a_ppg - margin / 2, 1),
        "market_spread": mkt_spread,
        "market_total": float(row.get("market_ou_total", 0) or 0),
        "disagree": round(abs(margin - (-mkt_spread)), 2) if mkt_spread else 0,
        "shap": shap_out[:20],
        "feature_coverage": f"{nonzero}/{len(feature_list)}",
        "model_meta": {
            "n_train": bundle.get("n_games"),
            "mae_cv": bundle.get("cv_mae"),
            "trained_at": bundle.get("trained_at"),
            "model_type": bundle.get("architecture", "Lasso_solo_v26"),
            "n_features": len(feature_list),
            "has_isotonic": calibrator is not None,
        },
        "diagnostics": diag,
    }
