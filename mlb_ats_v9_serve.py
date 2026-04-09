"""
mlb_ats_v9_serve.py — Serve-time MLB ATS v9 prediction
=======================================================
Loads mlb_ats_v9 model from Supabase, computes lineup features
at serve time, runs CatBoost + Lasso, blends, returns ATS pick.

Called by mlb_full_predict.py after margin prediction.

AUDIT FIXES:
  - Team matching uses team IDs (not name substrings)
  - Lineup rolling history fetched from recent boxscores
  - Graceful fallback: lineup features = 0 if unavailable
"""

import numpy as np
import requests
import traceback

MLB_API = "https://statsapi.mlb.com/api/v1"

# ── Caches (persist for Railway session) ──
_v9_bundle = None
_v9_load_error = ""
_batter_cache = {}
_team_rolling_cache = {}  # team_id -> list of recent lineup wOBAs

# ── Team ID <-> Abbreviation ──
_TEAM_ID_TO_ABBR = {
    108:"LAA",109:"ARI",110:"BAL",111:"BOS",112:"CHC",113:"CIN",114:"CLE",
    115:"COL",116:"DET",117:"HOU",118:"KC",119:"LAD",120:"WSH",121:"NYM",
    133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",138:"STL",139:"TB",
    140:"TEX",141:"TOR",142:"MIN",143:"PHI",144:"ATL",145:"CWS",146:"MIA",
    147:"NYY",158:"MIL",
}
_ABBR_TO_TEAM_ID = {v: k for k, v in _TEAM_ID_TO_ABBR.items()}


def _load_v9_model():
    global _v9_bundle, _v9_load_error
    if _v9_bundle is not None:
        return _v9_bundle
    try:
        from db import load_model
        _v9_bundle = load_model("mlb_ats_v9")
        if _v9_bundle and _v9_bundle.get("models"):
            names = _v9_bundle.get("model_names", [])
            feats = _v9_bundle.get("feature_sets", [])
            print(f"  [mlb_v9] Loaded: {names}, features: {[len(f) for f in feats]}")
            return _v9_bundle
        else:
            _v9_load_error = "bundle empty"
            return None
    except Exception as e:
        _v9_load_error = str(e)
        print(f"  [mlb_v9] Load error: {e}")
        return None


def _fetch_batter_season_stats(season=2026):
    """Fetch all batter season stats from MLB API. Cached per session."""
    global _batter_cache
    cache_key = f"batters_{season}"
    if cache_key in _batter_cache:
        return _batter_cache[cache_key]

    try:
        url = (f"{MLB_API}/stats?stats=season&group=hitting&season={season}"
               f"&sportIds=1&limit=1000&playerPool=ALL")
        r = requests.get(url, timeout=15)
        if not r.ok:
            return {}

        splits = r.json().get("stats", [{}])[0].get("splits", [])
        lookup = {}
        for s in splits:
            pid = s.get("player", {}).get("id")
            stat = s.get("stat", {})
            pa = int(stat.get("plateAppearances", 0) or 0)
            if pid and pa >= 10:
                ab = int(stat.get("atBats", 0) or 0)
                hits = int(stat.get("hits", 0) or 0)
                doubles = int(stat.get("doubles", 0) or 0)
                triples = int(stat.get("triples", 0) or 0)
                hr = int(stat.get("homeRuns", 0) or 0)
                bb = int(stat.get("baseOnBalls", 0) or 0)
                hbp = int(stat.get("hitByPitch", 0) or 0)
                singles = hits - doubles - triples - hr
                denom = ab + bb + hbp + 0.001
                woba = (0.69*bb + 0.72*hbp + 0.89*singles + 1.27*doubles + 1.62*triples + 2.10*hr) / denom

                lookup[pid] = {
                    "woba": round(woba, 4),
                    "ops": float(stat.get("ops", 0) or 0),
                    "iso": round(float(stat.get("slg", 0) or 0) - float(stat.get("avg", 0) or 0), 4),
                    "pa": pa,
                }

        _batter_cache[cache_key] = lookup
        print(f"  [mlb_v9] Loaded {len(lookup)} batter stats for {season}")
        return lookup
    except Exception as e:
        print(f"  [mlb_v9] Batter stats error: {e}")
        return {}


def _fetch_team_rolling_lineup(team_id, n_recent=10):
    """Fetch recent boxscores for a team to compute rolling lineup wOBA.
    Uses 10 games to match training pipeline. Cached per session."""
    global _team_rolling_cache
    if team_id in _team_rolling_cache:
        return _team_rolling_cache[team_id]

    try:
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        r = requests.get(f"{MLB_API}/schedule", params={
            "sportId": 1, "teamId": team_id,
            "startDate": start, "endDate": end,
            "gameType": "R",
        }, timeout=10)
        if not r.ok:
            _team_rolling_cache[team_id] = []
            return []

        # Get completed game_pks (most recent last)
        game_pks = []
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameCode") == "F":
                    game_pks.append(g["gamePk"])

        if not game_pks:
            _team_rolling_cache[team_id] = []
            return []

        # Only fetch last N boxscores to limit API calls
        batter_stats = _fetch_batter_season_stats()
        lineup_wobas = []

        for gpk in game_pks[-n_recent:]:
            try:
                r2 = requests.get(f"{MLB_API}/game/{gpk}/boxscore", timeout=8)
                if not r2.ok:
                    continue
                data = r2.json()
                home_id = data.get("teams", {}).get("home", {}).get("team", {}).get("id")
                side = "home" if home_id == team_id else "away"
                order = data.get("teams", {}).get(side, {}).get("battingOrder", [])[:9]
                if order:
                    wobas = [batter_stats.get(pid, {}).get("woba", 0.315) for pid in order]
                    lineup_wobas.append(np.mean(wobas))
            except Exception:
                continue

        _team_rolling_cache[team_id] = lineup_wobas
        return lineup_wobas
    except Exception as e:
        print(f"  [mlb_v9] Rolling lineup error for team {team_id}: {e}")
        _team_rolling_cache[team_id] = []
        return []


def fetch_pregame_lineups(game_pk=None, game_date=None, home_abbr=None, away_abbr=None):
    """Fetch pre-game lineups. Tries game_pk first, falls back to date+team IDs."""
    try:
        # Method 1: By game_pk (most reliable)
        if game_pk:
            r = requests.get(f"{MLB_API}/schedule", params={
                "sportId": 1, "gamePk": game_pk, "hydrate": "lineups",
            }, timeout=10)
            if r.ok:
                for d in r.json().get("dates", []):
                    for g in d.get("games", []):
                        h_lineup = [p.get("id") for p in g.get("lineups", {}).get("homePlayers", [])]
                        a_lineup = [p.get("id") for p in g.get("lineups", {}).get("awayPlayers", [])]
                        if h_lineup and a_lineup:
                            return h_lineup[:9], a_lineup[:9]

        # Method 2: By date + team IDs
        if game_date and home_abbr and away_abbr:
            home_tid = _ABBR_TO_TEAM_ID.get(home_abbr)
            away_tid = _ABBR_TO_TEAM_ID.get(away_abbr)
            if home_tid and away_tid:
                r = requests.get(f"{MLB_API}/schedule", params={
                    "sportId": 1, "date": game_date, "hydrate": "lineups",
                }, timeout=10)
                if r.ok:
                    for d in r.json().get("dates", []):
                        for g in d.get("games", []):
                            h_tid = g["teams"]["home"]["team"].get("id")
                            a_tid = g["teams"]["away"]["team"].get("id")
                            if h_tid == home_tid and a_tid == away_tid:
                                h_lineup = [p.get("id") for p in g.get("lineups", {}).get("homePlayers", [])]
                                a_lineup = [p.get("id") for p in g.get("lineups", {}).get("awayPlayers", [])]
                                if h_lineup and a_lineup:
                                    return h_lineup[:9], a_lineup[:9]

        return None, None
    except Exception as e:
        print(f"  [mlb_v9] Lineup fetch error: {e}")
        return None, None


def compute_lineup_features(home_lineup, away_lineup, batter_stats,
                             home_abbr=None, away_abbr=None):
    """Compute lineup features for a single game."""
    feats = {}

    for prefix, lineup, abbr in [("home", home_lineup, home_abbr),
                                   ("away", away_lineup, away_abbr)]:
        wobas, opss, isos = [], [], []
        matched = 0

        for pid in lineup:
            if pid in batter_stats:
                s = batter_stats[pid]
                wobas.append(s["woba"])
                opss.append(s["ops"])
                isos.append(s["iso"])
                matched += 1
            else:
                wobas.append(0.315)
                opss.append(0.710)
                isos.append(0.140)

        if not wobas:
            wobas = [0.315] * 9
            opss = [0.710] * 9
            isos = [0.140] * 9

        lineup_woba = np.mean(wobas)
        lineup_ops = np.mean(opss)
        lineup_iso = np.mean(isos)
        top3_woba = np.mean(wobas[:3]) if len(wobas) >= 3 else lineup_woba
        bot3_woba = np.mean(wobas[6:9]) if len(wobas) >= 9 else np.mean(wobas[-3:])
        top_heavy = top3_woba / lineup_woba if lineup_woba > 0.1 else 1.0

        feats[f"{prefix}_lineup_woba"] = round(lineup_woba, 4)
        feats[f"{prefix}_lineup_ops"] = round(lineup_ops, 4)
        feats[f"{prefix}_lineup_iso"] = round(lineup_iso, 4)
        feats[f"{prefix}_top3_woba"] = round(top3_woba, 4)
        feats[f"{prefix}_bot3_woba"] = round(bot3_woba, 4)
        feats[f"{prefix}_top_heavy"] = round(top_heavy, 4)
        feats[f"{prefix}_matched"] = matched

        # ── Rolling comparison (lineup change signal) ──
        team_id = _ABBR_TO_TEAM_ID.get(abbr)
        if team_id:
            history = _fetch_team_rolling_lineup(team_id)
            if len(history) >= 3:
                rolling_avg = np.mean(history)
                feats[f"{prefix}_woba_vs_rolling"] = round(lineup_woba - rolling_avg, 4)
            else:
                feats[f"{prefix}_woba_vs_rolling"] = 0
        else:
            feats[f"{prefix}_woba_vs_rolling"] = 0

    # ── Diffs (ATS) ──
    feats["lineup_woba_diff"] = round(feats.get("home_lineup_woba", 0.315) - feats.get("away_lineup_woba", 0.315), 4)
    feats["lineup_ops_diff"] = round(feats.get("home_lineup_ops", 0.710) - feats.get("away_lineup_ops", 0.710), 4)
    feats["lineup_iso_diff"] = round(feats.get("home_lineup_iso", 0.140) - feats.get("away_lineup_iso", 0.140), 4)
    feats["lineup_delta_diff"] = round(feats.get("home_woba_vs_rolling", 0) - feats.get("away_woba_vs_rolling", 0), 4)
    feats["lineup_top_heavy_diff"] = round(feats.get("home_top_heavy", 1) - feats.get("away_top_heavy", 1), 4)

    # ── O/U totals ──
    feats["lineup_delta_sum"] = round(feats.get("home_woba_vs_rolling", 0) + feats.get("away_woba_vs_rolling", 0), 4)
    feats["lineup_total_woba"] = round(feats.get("home_lineup_woba", 0.315) + feats.get("away_lineup_woba", 0.315), 4)
    feats["lineup_total_iso"] = round(feats.get("home_lineup_iso", 0.14) + feats.get("away_lineup_iso", 0.14), 4)
    feats["lineup_total_top3"] = round(feats.get("home_top3_woba", 0.34) + feats.get("away_top3_woba", 0.34), 4)

    return feats


def predict_mlb_ats_v9(game_features, lineup_features=None, market_spread=0):
    """Run MLB ATS v9 model."""
    result = {
        "ats_v9_side": None, "ats_v9_units": 0, "ats_v9_blend": None,
        "ats_v9_cb": None, "ats_v9_lasso": None, "ats_v9_models_agree": None,
    }

    if not market_spread or abs(market_spread) < 0.1:
        return result

    bundle = _load_v9_model()
    if not bundle:
        return result

    try:
        models = bundle.get("models", [])
        scalers = bundle.get("scalers", [])
        feature_sets = bundle.get("feature_sets", [])
        model_names = bundle.get("model_names", [])
        weights = bundle.get("model_weights", [0.2, 0.8])

        if len(models) < 2:
            return result

        all_feats = dict(game_features or {})
        if lineup_features:
            all_feats.update(lineup_features)

        predictions = []
        for model, scaler, feat_list, name in zip(models, scalers, feature_sets, model_names):
            fv = [float(all_feats.get(f, 0) or 0) for f in feat_list]
            X = np.array([fv])
            X_s = scaler.transform(X)
            pred = float(model.predict(X_s)[0])
            predictions.append(pred)

            if "Lasso" in name:
                result["ats_v9_lasso"] = round(pred, 3)
            elif "CatBoost" in name:
                result["ats_v9_cb"] = round(pred, 3)

        margin = sum(p * w for p, w in zip(predictions, weights))
        result["ats_v9_blend"] = round(margin, 3)

        lasso_pred, cb_pred = predictions[0], predictions[1]
        agree = (lasso_pred > 0 and cb_pred > 0) or (lasso_pred < 0 and cb_pred < 0)
        result["ats_v9_models_agree"] = agree

        mkt_implied = -market_spread
        edge = abs(margin - mkt_implied)

        if edge >= 2.5 and agree:
            units = 2
        elif edge >= 2.0 and agree:
            units = 1
        else:
            units = 0

        result["ats_v9_units"] = units
        if units > 0:
            result["ats_v9_side"] = "HOME" if margin > mkt_implied else "AWAY"
            result["ats_v9_edge"] = round(edge, 2)

        n_nz = sum(1 for f in feature_sets[1] if abs(all_feats.get(f, 0) or 0) > 1e-6)
        print(f"  [mlb_v9] margin={margin:+.2f}, edge={edge:.2f}, cb={cb_pred:+.2f}, "
              f"lasso={lasso_pred:+.2f}, agree={agree}, units={units}, "
              f"coverage={n_nz}/{len(feature_sets[1])}")

    except Exception as e:
        print(f"  [mlb_v9] Error: {e}")
        print(f"  {traceback.format_exc()}")

    return result
