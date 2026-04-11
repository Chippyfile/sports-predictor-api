"""
Multi-Sport Predictor API v3 — Modular Flask Backend
Slim entry point. All logic lives in imported modules.
"""
import os
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import numpy as np
import pandas as pd

# ── Create Flask app ───────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Import shared infrastructure ───────────────────────────────
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_DIR
from db import sb_get, save_model, load_model
from ml_utils import HAS_XGB, accuracy_report
from ml_utils import StackedRegressor, StackedClassifier; import sys; sys.modules["__main__"].StackedRegressor = StackedRegressor; sys.modules["__main__"].StackedClassifier = StackedClassifier
from nba_ensemble import EnsembleRegressor  # needed for joblib to deserialize NBA v27 ensemble model

# ── Import sport modules ──────────────────────────────────────
from sports.mlb import train_mlb, predict_mlb, calibrate_mlb_dispersion
from sports.nba import train_nba, predict_nba, nba_build_features
from sports.ncaa import train_ncaa, predict_ncaa, ncaa_build_features
from ncaa_full_predict import predict_ncaa_full
from nba_full_predict import predict_nba_full
try:
    from nba_game_stats import process_completed_games, backfill_game_stats
except ImportError:
    def process_completed_games(*a, **k): return {"error": "nba_game_stats not available"}
    def backfill_game_stats(*a, **k): return {"error": "nba_game_stats not available"}
from sports.nfl import train_nfl, predict_nfl, nfl_build_features
from sports.ncaaf import train_ncaaf, predict_ncaaf, ncaaf_build_features
# REMOVED: nba_backfill cleaned from repo
try:
    from quick_backtest import quick_backtest_nba, quick_backtest_ncaa, quick_backtest_mlb
except ImportError:
    def quick_backtest_nba(*a, **k): return {"error": "quick_backtest not available"}
    def quick_backtest_ncaa(*a, **k): return {"error": "quick_backtest not available"}
    def quick_backtest_mlb(*a, **k): return {"error": "quick_backtest not available"}
try:
    from season_holdout_backtest import season_holdout_nba, season_holdout_ncaa, season_holdout_mlb, season_holdout_all
except ImportError:
    def season_holdout_nba(*a, **k): return {"error": "season_holdout_backtest not available"}
    def season_holdout_ncaa(*a, **k): return {"error": "season_holdout_backtest not available"}
    def season_holdout_mlb(*a, **k): return {"error": "season_holdout_backtest not available"}
    def season_holdout_all(*a, **k): return {"error": "season_holdout_backtest not available"}
try:
    from monte_carlo import monte_carlo
except ImportError:
    def monte_carlo(*a, **k): return {"error": "monte_carlo not available"}
try:
    from cron import _active_sports, _log_training, _should_promote
except ImportError:
    def _active_sports(): return []
    def _log_training(*a, **k): pass
    def _should_promote(*a, **k): return False


# ═══════════════════════════════════════════════════════════════
# ROUTES — Index & Health
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "service": "Multi-Sport Predictor API v3 (modular)",
        "xgboost": HAS_XGB,
        "endpoints": [
            "GET  /health",
            "POST /train/<sport>",
            "POST /train/all",
            "POST /train/all-logged",
            "POST /calibrate/mlb",
            "POST /predict/<sport>",
            "POST /monte-carlo",
            "GET  /accuracy/<sport>",
            "GET  /accuracy/all",
            "GET  /model-info/<sport>",
            "POST /backtest/mlb",
            "POST /backtest/ncaa",
            "GET  /backtest/nba",
            "POST /cron/auto-train",
            "GET  /cron/status",
            "POST /compute/ncaa-efficiency",
            "GET  /ratings/ncaa",
            "GET  /cron/ncaa-daily?mode=predict|refresh|grade|auto",
            "GET  /cron/ncaa-backfill?season=2026  (backfill elo/form/etc.)",
            "GET  /cron/nba-daily?mode=predict|grade|auto",
            "GET  /cron/nba-player-update  (hoopR rolling BPM/VORP)",
            "GET  /cron/mlb-daily?mode=predict|grade|auto",
            "GET  /cron/ncaa-ats-record",
        ],
    })

@app.route("/health")
def health():
    trained = []
    for s in ["mlb", "nba", "ncaa", "nfl", "ncaaf"]:
        try:
            if load_model(s):
                trained.append(s)
        except Exception:
            pass  # stale or incompatible pkl — skip, don't crash
    disp = None
    try:
        disp = load_model("mlb_dispersion")
    except Exception:
        pass
    return jsonify({
        "status": "healthy",
        "trained_models": trained,
        "mlb_dispersion": disp if disp else "not calibrated — POST /calibrate/mlb",
        "timestamp": datetime.utcnow().isoformat(),
    })
    
@app.route("/reload-model/<sport>", methods=["POST"])
def route_reload_model(sport):
    """Force reload model from Supabase (bypasses in-memory cache)."""
    from db import _models
    _models.pop(sport, None)
    path = os.path.join(MODEL_DIR, f"{sport}.pkl")
    if os.path.exists(path):
        os.remove(path)
    bundle = load_model(sport)
    if bundle:
        return jsonify({"status": "reloaded", "sport": sport,
                       "mae_cv": bundle.get("mae_cv"),
                       "trained_at": bundle.get("trained_at")})
    return jsonify({"error": f"No model found for {sport}"}), 404

@app.route("/debug/reload-model/<sport>", methods=["POST"])
def route_debug_reload(sport):
    import requests, base64, io, traceback, joblib
    import base64, io, traceback
    from db import _models
    _models.pop(sport, None)
    path = os.path.join(MODEL_DIR, f"{sport}.pkl")
    if os.path.exists(path):
        os.remove(path)
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/model_store?name=eq.{sport}&select=data",
            headers=headers, timeout=120)
        if not resp.ok:
            return jsonify({"error": "supabase_fetch_failed", "status": resp.status_code, "body": resp.text[:500]})
        rows = resp.json()
        if not rows or not rows[0].get("data"):
            return jsonify({"error": "no_data_in_response", "rows_count": len(rows)})
        data_len = len(rows[0]["data"])
        raw = base64.b64decode(rows[0]["data"])
        # Fix numpy version mismatch for MT19937 BitGenerator
        import numpy.random._pickle as _nrp
        _orig_ctor = _nrp.__bit_generator_ctor
        def _patched_ctor(bit_generator_name):
            if 'MT19937' in str(bit_generator_name):
                from numpy.random import MT19937
                return MT19937()
            return _orig_ctor(bit_generator_name)
        _nrp.__bit_generator_ctor = _patched_ctor
        obj = joblib.load(io.BytesIO(raw))
        _models[sport] = obj
        return jsonify({"status": "ok", "mae": obj.get("mae_cv"), "data_len": data_len, "features": len(obj.get("feature_cols", []))})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ═══════════════════════════════════════════════════════════════
# ROUTES — Training
# ═══════════════════════════════════════════════════════════════

@app.route("/train/mlb", methods=["POST"])
def route_train_mlb():
    """GUARDED: Production model uses mlb_ensemble_retrain.py (29 features).
    This endpoint trains a 41-feature CatBoost-solo model that would overwrite it.
    Pass ?force=true to bypass this guard."""
    if not request.args.get("force"):
        return jsonify({
            "error": "BLOCKED — use local mlb_ensemble_retrain.py --upload instead",
            "reason": "This endpoint trains a 41-feature model that overwrites the 29-feature production ensemble",
            "bypass": "Add ?force=true to override (not recommended)"
        }), 403
    return jsonify(train_mlb())

@app.route("/train/nba", methods=["POST"])
def route_train_nba(): return jsonify(train_nba())

@app.route("/train/ncaa", methods=["POST"])
def route_train_ncaa(): return jsonify(train_ncaa())

@app.route("/train/nfl", methods=["POST"])
def route_train_nfl(): return jsonify(train_nfl())

@app.route("/train/ncaaf", methods=["POST"])
def route_train_ncaaf(): return jsonify(train_ncaaf())

@app.route("/train/all", methods=["POST"])
def route_train_all():
    return jsonify({
        "mlb": train_mlb(), "nba": train_nba(), "ncaa": train_ncaa(),
        "nfl": train_nfl(), "ncaaf": train_ncaaf(),
    })

# ═══════════════════════════════════════════════════════════════
# ROUTES — Async Training (bypasses proxy timeout)
# POST /train/async/<sport>  — returns immediately, trains in background
# POST /train/async/all      — queues all sports sequentially
# GET  /train/async/status   — check progress
# ═══════════════════════════════════════════════════════════════

import threading

_async_status = {}  # sport -> {"status": "running"|"done"|"error", "result": ..., "started_at": ...}

def _run_async(sport, fn):
    _async_status[sport] = {"status": "running", "started_at": datetime.utcnow().isoformat()}
    try:
        result = fn()
        _async_status[sport] = {
            "status": "done",
            "result": result,
            "started_at": _async_status[sport]["started_at"],
            "finished_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        _async_status[sport] = {
            "status": "error",
            "error": str(e),
            "started_at": _async_status[sport]["started_at"],
            "finished_at": datetime.utcnow().isoformat(),
        }

@app.route("/train/async/<sport>", methods=["POST"])
def route_train_async(sport):
    fns = {"mlb": train_mlb, "nba": train_nba, "ncaa": train_ncaa,
           "nfl": train_nfl, "ncaaf": train_ncaaf}
    fn = fns.get(sport.lower())
    if not fn:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    if _async_status.get(sport, {}).get("status") == "running":
        return jsonify({"status": "already_running", "sport": sport})
    t = threading.Thread(target=_run_async, args=(sport, fn), daemon=True)
    t.start()
    return jsonify({"status": "started", "sport": sport, "check": f"/train/async/status"})

@app.route("/train/async/all", methods=["POST"])
def route_train_async_all():
    fns = {"mlb": train_mlb, "nba": train_nba, "ncaa": train_ncaa,
           "nfl": train_nfl, "ncaaf": train_ncaaf}
    def _run_all():
        for sport, fn in fns.items():
            _run_async(sport, fn)
    already_running = [s for s, v in _async_status.items() if v.get("status") == "running"]
    if already_running:
        return jsonify({"status": "already_running", "sports": already_running})
    t = threading.Thread(target=_run_all, daemon=True)
    t.start()
    return jsonify({"status": "started", "sports": list(fns.keys()), "check": "/train/async/status"})

@app.route("/train/async/status")
def route_train_async_status():
    return jsonify({
        "jobs": _async_status,
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/calibrate/mlb", methods=["POST"])
def route_calibrate_mlb():
    return jsonify(calibrate_mlb_dispersion())


# ═══════════════════════════════════════════════════════════════
# ROUTES — Prediction & Monte Carlo
# ═══════════════════════════════════════════════════════════════

@app.route("/predict/<sport>", methods=["POST"])
def route_predict(sport):
    game = request.get_json(force=True, silent=True) or {}
    fns = {"mlb": predict_mlb, "nba": predict_nba, "ncaa": predict_ncaa,
           "nfl": predict_nfl, "ncaaf": predict_ncaaf}
    fn = fns.get(sport.lower())
    if not fn:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    try:
        return jsonify(fn(game))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[predict/{sport}] ERROR: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/predict/ncaa/full", methods=["POST"])
def route_predict_ncaa_full():
    """Full NCAA prediction with backend data lookup — 146/146 features."""
    game = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(predict_ncaa_full(game))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[predict/ncaa/full] ERROR: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/predict/nba/full", methods=["POST"])
def route_predict_nba_full():
    """Full NBA prediction with server-side enrichment — v26 Lasso."""
    game = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(predict_nba_full(game))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[predict/nba/full] ERROR: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/predict/mlb/ou", methods=["POST"])
def route_predict_mlb_ou():
    """MLB Over/Under prediction using dedicated total-runs model."""
    game = request.get_json(force=True, silent=True) or {}
    try:
        from sports.mlb import predict_mlb_ou
        return jsonify(predict_mlb_ou(game))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[predict/mlb/ou] ERROR: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/predict/mlb/full", methods=["POST"])
def route_predict_mlb_full():
    """Full server-side MLB prediction — fetches all data from MLB Stats API + Supabase."""
    game = request.get_json(force=True, silent=True) or {}
    try:
        from mlb_full_predict import predict_mlb_full
        return jsonify(predict_mlb_full(game))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[predict/mlb/full] ERROR: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/nba/backfill-stats", methods=["POST"])
def route_nba_backfill_stats():
    """Backfill nba_game_stats + nba_team_rolling from recent completed games."""
    data = request.get_json(force=True, silent=True) or {}
    days = data.get("days_back", 30)
    force = data.get("force", False)
    try:
        n = backfill_game_stats(days_back=days, force=force)
        return jsonify({"processed": n})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/nba/backfill-enrichment", methods=["POST"])
def route_nba_backfill_enrichment():
    """Recompute nba_team_enrichment for all 30 teams from nba_game_stats."""
    try:
        from nba_enrichment import recompute_all_enrichment
        n = recompute_all_enrichment()
        return jsonify({"enriched": n})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/nba/backfill-refs", methods=["POST"])
def route_nba_backfill_refs():
    """Build NBA referee profiles from nba_game_stats history."""
    try:
        from nba_enrichment import build_ref_profiles
        profiles = build_ref_profiles(min_games=10)
        return jsonify({"profiles": len(profiles)})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/nba/debug-rolling", methods=["GET"])
def route_nba_debug_rolling():
    """Debug: show raw rolling table + last 3 game_stats rows for a team."""
    import requests as _req
    team = request.args.get("team", "CHA")
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    rolling = _req.get(
        f"{SUPABASE_URL}/rest/v1/nba_team_rolling?team_abbr=eq.{team}&select=*",
        headers=headers, timeout=10).json()
    raw_games = _req.get(
        f"{SUPABASE_URL}/rest/v1/nba_game_stats?team_abbr=eq.{team}&order=game_date.desc&limit=3&select=*",
        headers=headers, timeout=10).json()
    return jsonify({"team": team, "rolling": rolling, "last_3_games": raw_games})

@app.route("/nba/test-extract", methods=["GET"])
def route_nba_test_extract():
    """Debug: run full extraction on one game, or show enrichment data."""
    import traceback
    game_id = request.args.get("game_id", "")
    team = request.args.get("team", "")

    if team:
        # Show enrichment data for a team
        try:
            from nba_enrichment import get_team_enrichment
            data = get_team_enrichment(team)
            return jsonify({"team": team, "enrichment": data})
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()})

    if not game_id:
        game_id = "401810878"
    try:
        from nba_game_stats import _fetch_boxscore_stats
        result = _fetch_boxscore_stats(game_id)
        if result is None:
            return jsonify({"error": "fetch returned None", "game_id": game_id})
        summary = {}
        for abbr, stats in result.items():
            summary[abbr] = {
                "bench_pts": stats.get("bench_pts"),
                "three_fg_rate": stats.get("three_fg_rate"),
                "ft_trip_rate": stats.get("ft_trip_rate"),
                "paint_pts": stats.get("paint_pts"),
                "oreb": stats.get("oreb"),
                "q4_scoring": stats.get("q4_scoring"),
                "max_run": stats.get("max_run"),
            }
        return jsonify({"game_id": game_id, "teams": summary})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

@app.route("/nba/debug-espn-boxscore", methods=["GET"])
def route_nba_debug_boxscore():
    """Debug: test both public and web ESPN APIs."""
    import requests as _req
    game_id = request.args.get("game_id", "401810878")
    result = {}

    # Test public API
    try:
        r1 = _req.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}", timeout=15)
        result["public_api"] = {"status": r1.status_code, "has_players": "players" in r1.json().get("boxscore", {})} if r1.ok else {"status": r1.status_code}
        if r1.ok:
            data = r1.json()
            # Check officials
            officials = data.get("gameInfo", {}).get("officials", [])
            result["officials"] = [
                {"name": o.get("displayName", ""), "position": o.get("position", {}).get("displayName", "")}
                for o in officials
            ]
            for pb in data.get("boxscore", {}).get("players", []):
                abbr = pb.get("team", {}).get("abbreviation", "?")
                sg = pb.get("statistics", [{}])[0] if pb.get("statistics") else {}
                result["public_api"][f"players_{abbr}"] = {
                    "keys": sg.get("keys", []),
                    "n_athletes": len(sg.get("athletes", [])),
                    "sample_starter": next((a.get("starter") for a in sg.get("athletes", []) if a.get("stats")), None),
                }
    except Exception as e:
        result["public_api"] = {"error": str(e)}

    # Test web API
    try:
        r2 = _req.get(
            f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?region=us&lang=en&contentorigin=espn&event={game_id}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15)
        result["web_api"] = {"status": r2.status_code}
        if r2.ok:
            data2 = r2.json()
            result["web_api"]["has_players"] = "players" in data2.get("boxscore", {})
            # Check officials on web API too
            web_officials = data2.get("gameInfo", {}).get("officials", [])
            result["web_api"]["officials"] = [o.get("displayName", "") for o in web_officials]
            result["web_api"]["has_players"] = "players" in data2.get("boxscore", {})
            for pb in data2.get("boxscore", {}).get("players", []):
                abbr = pb.get("team", {}).get("abbreviation", "?")
                sg = pb.get("statistics", [{}])[0] if pb.get("statistics") else {}
                result["web_api"][f"players_{abbr}"] = {
                    "n_athletes": len(sg.get("athletes", [])),
                    "sample_starter": next((a.get("starter") for a in sg.get("athletes", []) if a.get("stats")), None),
                }
    except Exception as e:
        result["web_api"] = {"error": str(e)}

    return jsonify(result)

@app.route("/monte-carlo", methods=["POST"])
def route_monte_carlo():
    body = request.get_json(force=True, silent=True) or {}
    sport = body.get("sport", "NBA").upper()
    home_mean = float(body.get("home_mean", 110))
    away_mean = float(body.get("away_mean", 110))
    n_sims = min(int(body.get("n_sims", 10000)), 100_000)
    ou_line = float(body["ou_line"]) if body.get("ou_line") is not None else None
    game_id = body.get("game_id")
    return jsonify(monte_carlo(sport, home_mean, away_mean, n_sims, ou_line, game_id))


# ═══════════════════════════════════════════════════════════════
# ROUTES — Accuracy & Model Info
# ═══════════════════════════════════════════════════════════════

SPORT_TABLES = {
    "mlb": ("mlb_predictions", "MLB"),
    "nba": ("nba_predictions", "NBA"),
    "ncaa": ("ncaa_predictions", "NCAAB"),
    "nfl": ("nfl_predictions", "NFL"),
    "ncaaf": ("ncaaf_predictions", "NCAAF"),
}

@app.route("/accuracy/<sport>")
def route_accuracy(sport):
    cfg = SPORT_TABLES.get(sport.lower())
    if not cfg:
        return jsonify({"error": f"Unknown sport: {sport}"}), 400
    return jsonify(accuracy_report(*cfg))

@app.route("/accuracy/all")
def route_accuracy_all():
    return jsonify({k: accuracy_report(*v) for k, v in SPORT_TABLES.items()})

@app.route("/model-info/<sport>")
def route_model_info(sport):
    try:
        bundle = load_model(sport.lower())
    except Exception:
        bundle = None
    if not bundle:
        return jsonify({"error": f"{sport} model not trained yet"})
    info = {
        "sport": sport.upper(),
        "model_type": bundle.get("model_type", ""),
        "n_train": bundle.get("n_train"),
        "mae_cv": bundle.get("mae_cv"),
        "trained_at": bundle.get("trained_at"),
        "features": bundle.get("feature_cols"),
    }
    if sport.lower() == "mlb":
        try:
            info["dispersion"] = load_model("mlb_dispersion")
        except Exception:
            info["dispersion"] = None
    return jsonify(info)


# ═══════════════════════════════════════════════════════════════
# ROUTES — Cron & Auto-Training
# ═══════════════════════════════════════════════════════════════

@app.route("/cron/auto-train", methods=["POST"])
def route_cron():
    """Daily auto-training with shadow model comparison."""
    import traceback, time as _time, json
    start = _time.time()
    trigger = request.args.get("trigger", "cron")
    force = request.args.get("force", "").lower() == "true"

    if request.args.get("sports"):
        sports = [s.strip().lower() for s in request.args["sports"].split(",")]
    elif force:
        sports = ["mlb", "nba", "ncaa", "nfl", "ncaaf"]
    else:
        sports = _active_sports()

    train_fns = {
        "mlb": train_mlb, "nba": train_nba, "ncaa": train_ncaa,
        "nfl": train_nfl, "ncaaf": train_ncaaf,
    }

    results = {}
    for sport in sports:
        fn = train_fns.get(sport)
        if not fn:
            results[sport] = {"status": "unknown_sport"}
            continue
        sport_start = _time.time()
        try:
            new_result = fn()
            duration = _time.time() - sport_start
            if "error" in new_result:
                results[sport] = {"status": "skipped", "reason": new_result["error"]}
                _log_training(sport, "skipped", new_result, duration=duration, trigger=trigger)
                continue
            should, reason, prev_mae = _should_promote(sport, new_result)
            new_result["_promoted"] = should
            new_result["_promote_reason"] = reason
            if prev_mae is not None:
                new_result["_mae_previous"] = prev_mae
            _log_training(sport, "promoted" if should else "shadow", new_result, duration=duration, trigger=trigger)
            results[sport] = {
                "status": "promoted" if should else "shadow",
                "reason": reason,
                "mae_cv": new_result.get("mae_cv"),
                "n_train": new_result.get("n_train"),
                "model_type": new_result.get("model_type", ""),
                "duration_sec": round(duration, 2),
            }
        except Exception as e:
            duration = _time.time() - sport_start
            _log_training(sport, "error", error=e, duration=duration, trigger=trigger)
            results[sport] = {"status": "error", "error": str(e)}

    total_duration = _time.time() - start
    return jsonify({
        "status": "complete",
        "trigger": trigger,
        "sports_trained": sports,
        "results": results,
        "duration_sec": round(total_duration, 2),
    })

@app.route("/cron/status")
def route_cron_status():
    """Model freshness & last training run."""
    from db import load_model
    models_status = {}
    for sport in ["mlb", "nba", "ncaa", "nfl", "ncaaf"]:
        try:
            bundle = load_model(sport)
        except Exception:
            bundle = None
        if bundle:
            trained_at = bundle.get("trained_at", "")
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(trained_at)).total_seconds() / 3600
            except:
                age = None
            models_status[sport] = {
                "trained": True,
                "model_type": bundle.get("model_type", ""),
                "n_train": bundle.get("n_train"),
                "mae_cv": bundle.get("mae_cv"),
                "trained_at": trained_at,
                "age_hours": round(age, 1) if age else None,
                "freshness": "fresh" if age and age < 36 else "stale" if age else "unknown",
            }
        else:
            models_status[sport] = {"trained": False, "freshness": "no_model"}

    # Last cron run from training_log
    last_run = {}
    try:
        log_rows = sb_get("training_log", "order=created_at.desc&limit=1")
        if log_rows:
            last_run = {
                "run_at": log_rows[0].get("created_at"),
                "status": log_rows[0].get("status"),
                "duration_sec": log_rows[0].get("duration_sec"),
            }
    except:
        pass

    return jsonify({
        "models": models_status,
        "active_sports": _active_sports(),
        "last_cron_run": last_run,
        "next_run": "Daily at 08:00 UTC (4 AM ET)",
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/train/all-logged", methods=["POST"])
def route_train_all_logged():
    """Alias: trigger auto-train with force + manual tag."""
    from flask import redirect
    return route_cron()


# ═══════════════════════════════════════════════════════════════
# ROUTES — Backtests (imported from backtests.py)
# The backtest module has handler functions; we register routes here.
# ═══════════════════════════════════════════════════════════════

try:
    from backtests import (
        route_backtest_mlb, route_backtest_nba, route_backtest_ncaa,
        route_model_info as _backtest_model_info,
        nba_confidence_calibration, ncaa_confidence_calibration, mlb_confidence_calibration,
        route_backtest_current_model,
        historical_ats_ncaa, historical_ats_nba, historical_ats_mlb, historical_ats_all,
        spread_edge_ncaa, spread_edge_nba, spread_edge_mlb,
    )
except ImportError:
    route_backtest_mlb = route_backtest_nba = route_backtest_ncaa = None
    _backtest_model_info = nba_confidence_calibration = ncaa_confidence_calibration = mlb_confidence_calibration = None
    route_backtest_current_model = None
    historical_ats_ncaa = historical_ats_nba = historical_ats_mlb = historical_ats_all = None
    spread_edge_ncaa = spread_edge_nba = spread_edge_mlb = None

# Only register if they exist (graceful degradation)
import inspect
_backtest_routes = {
    "/backtest/mlb": (["GET", "POST"], "route_backtest_mlb", route_backtest_mlb),
    "/backtest/nba": (["GET", "POST"], "route_backtest_nba", route_backtest_nba),
    "/backtest/ncaa": (["POST"], "route_backtest_ncaa", route_backtest_ncaa),
    "/backtest/nba-confidence": (["GET"], "nba_confidence", nba_confidence_calibration),
    "/backtest/ncaa-confidence": (["GET"], "ncaa_confidence", ncaa_confidence_calibration),
    "/backtest/mlb-confidence": (["GET"], "mlb_confidence", mlb_confidence_calibration),
    "/backtest/mlb/current-model": (["POST"], "backtest_mlb_current", route_backtest_current_model),
    "/historical-ats/ncaa": (["GET"], "historical_ats_ncaa", historical_ats_ncaa),
    "/historical-ats/nba": (["GET"], "historical_ats_nba", historical_ats_nba),
    "/historical-ats/mlb": (["GET"], "historical_ats_mlb", historical_ats_mlb),
    "/historical-ats/all": (["GET"], "historical_ats_all", historical_ats_all),
    "/spread-edge/ncaa": (["GET"], "spread_edge_ncaa", spread_edge_ncaa),
    "/spread-edge/nba": (["GET"], "spread_edge_nba", spread_edge_nba),
    "/spread-edge/mlb": (["GET"], "spread_edge_mlb", spread_edge_mlb),
}
for path, (methods, name, fn) in _backtest_routes.items():
    try:
        app.add_url_rule(path, name, fn, methods=methods)
    except Exception as e:
        print(f"  [routes] Could not register {path}: {e}")


# ═══════════════════════════════════════════════════════════════
# ROUTES — NCAA Ratings
# ═══════════════════════════════════════════════════════════════

try:
    from ncaa_ratings import run_ncaa_efficiency_computation
except ImportError:
    run_ncaa_efficiency_computation = None

@app.route("/compute/ncaa-efficiency", methods=["POST"])
def route_ncaa_efficiency():
    try:
        result = run_ncaa_efficiency_computation()
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/ratings/ncaa")
def route_ncaa_ratings():
    rows = sb_get("ncaa_team_ratings", "order=adj_em.desc&limit=400")
    # Ensure rank_adj_em exists (compute from sort order if missing)
    for i, r in enumerate(rows):
        if r.get("rank_adj_em") is None:
            r["rank_adj_em"] = i + 1
    updated_at = rows[0].get("updated_at", "") if rows else ""
    return jsonify({"ratings": rows, "updated_at": updated_at})

@app.route("/ratings/ncaa/<team_id>")
def route_ncaa_team_rating(team_id):
    rows = sb_get("ncaa_team_ratings", f"team_id=eq.{team_id}")
    return jsonify(rows[0] if rows else {"error": "Team not found"})


# ═══════════════════════════════════════════════════════════════
# ROUTES — Debug
# ═══════════════════════════════════════════════════════════════

@app.route("/debug/supabase")
def debug_supabase():
    results = {}
    for table in ["mlb_predictions", "nba_predictions", "ncaa_predictions"]:
        try:
            rows = sb_get(table, "select=count&limit=1")
            results[table] = {"accessible": True}
        except Exception as e:
            results[table] = {"accessible": False, "error": str(e)}
    results["env"] = {
        "SUPABASE_URL": "Set" if SUPABASE_URL else "Missing",
        "SUPABASE_KEY": "Set" if SUPABASE_KEY else "Missing",
    }
    return jsonify(results)

@app.route("/debug/train-mlb", methods=["POST"])
def debug_train_mlb():
    """GUARDED: Same as /train/mlb — use local retrain instead."""
    if not request.args.get("force"):
        return jsonify({"error": "BLOCKED — use mlb_ensemble_retrain.py --upload", "bypass": "?force=true"}), 403
    import traceback
    try:
        return jsonify(train_mlb())
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# ROUTES — NCAA Daily Cron (fully automated predictions + grading)
# Schedule in Railway:
#   8:00 AM EST  (13:00 UTC) → /cron/ncaa-daily?mode=predict  (predict today + tomorrow)
#   11:30 PM EST (04:30 UTC) → /cron/ncaa-daily?mode=grade    (grade results)
# No midday refresh needed — 8AM predict pulls today + tomorrow's games.
# For pre-tip updates (spread movement, officials), use frontend refresh button.
# ═══════════════════════════════════════════════════════════════

_ncaa_cron_lock = False

@app.route("/cron/ncaa-daily", methods=["GET", "POST"])
def route_ncaa_daily():
    """
    Automated NCAA sync — no login required.
    Modes:
      predict  — fetch today's games, run ML predictions, save picks + ATS signals
      refresh  — re-fetch spreads/starters for today's unplayed games, update predictions
      grade    — fill final scores, grade ML/ATS/O-U for completed games
      auto     — (default) predict if morning, grade if night, refresh if midday
    """
    import time as _time, traceback
    from datetime import datetime, timezone, timedelta
    global _ncaa_cron_lock

    if _ncaa_cron_lock:
        return jsonify({"status": "already_running"}), 429

    _ncaa_cron_lock = True
    start = _time.time()

    try:
        mode = request.args.get("mode", "auto")
        now_utc = datetime.now(timezone.utc)
        now_est = now_utc - timedelta(hours=5)
        today_pst = request.args.get("date") or now_est.strftime("%Y-%m-%d")
        hour_est = now_est.hour

        # Auto mode: predict in morning, grade at night
        if mode == "auto":
            if hour_est < 14:
                mode = "predict"
            else:
                mode = "grade"

        import requests as _req
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

        results = {"mode": mode, "date": today_pst, "hour_est": hour_est}

        if mode in ("predict", "refresh"):
            # Fetch today's AND tomorrow's games from ESPN
            # This way the 8AM EST cron covers all games without needing midday refreshes
            tomorrow_pst = (now_est + timedelta(days=1)).strftime("%Y-%m-%d")
            dates_to_pull = [today_pst, tomorrow_pst]

            all_events = []
            for pull_date in dates_to_pull:
                compact = pull_date.replace("-", "")
                espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={compact}&groups=50&limit=500"
                espn_resp = _req.get(espn_url, timeout=30)
                day_events = espn_resp.json().get("events", []) if espn_resp.ok else []
                # Tag each event with its game date
                for ev in day_events:
                    ev["_pull_date"] = pull_date
                all_events.extend(day_events)
                print(f"  [cron/ncaa] {pull_date}: {len(day_events)} games from ESPN")

            events = all_events
            results["espn_games"] = len(events)
            results["dates_pulled"] = dates_to_pull

            if not events:
                results["status"] = "no_games_today"
                return jsonify(results)

            # Check which games already have predictions (both dates)
            existing_ids = {}
            for pull_date in dates_to_pull:
                existing = _req.get(
                    f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_date=eq.{pull_date}&select=game_id,result_entered,spread_home,market_spread_home",
                    headers=headers, timeout=30
                )
                for r in (existing.json() if existing.ok else []):
                    existing_ids[r["game_id"]] = r
            results["existing_predictions"] = len(existing_ids)

            # Run predictions for games without them (predict mode)
            # or re-predict all unplayed games (refresh mode)
            predicted = 0
            refreshed = 0
            errors = 0

            for event in events:
                comp = event.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {})
                if status.get("completed") and not request.args.get("force"):
                    continue  # skip finished games

                game_id = event.get("id")
                # Use actual game date from event (handles tomorrow's games)
                event_date_str = event.get("_pull_date", today_pst)
                home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
                away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                home_id = home.get("team", {}).get("id")
                away_id = away.get("team", {}).get("id")
                neutral = comp.get("neutralSite", False)

                if not home_id or not away_id:
                    continue

                # Skip if already predicted (predict mode only)
                if mode == "predict" and game_id in existing_ids:
                    continue

                # Call /predict/ncaa/full for this game
                try:
                    pred = predict_ncaa_full({
                        "home_team_id": str(home_id),
                        "away_team_id": str(away_id),
                        "neutral_site": neutral,
                        "game_date": event_date_str,
                        "game_id": str(game_id),
                    })

                    if pred and not pred.get("error") and pred.get("ml_win_prob_home") is not None:
                        # Build the row to save
                        home_name = home.get("team", {}).get("displayName", "")
                        away_name = away.get("team", {}).get("displayName", "")
                        home_abbr = home.get("team", {}).get("abbreviation", "")
                        away_abbr = away.get("team", {}).get("abbreviation", "")

                        # Get market spread from pickcenter
                        audit = pred.get("audit_data", {})

                        # ESPN pickcenter spread
                        pickcenter = []
                        try:
                            summary = _req.get(
                                f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
                                timeout=10
                            )
                            if summary.ok:
                                pickcenter = summary.json().get("pickcenter", [])
                        except:
                            pass

                        espn_spread = None
                        espn_ou = None
                        espn_home_ml = None
                        espn_away_ml = None
                        for pc in pickcenter:
                            if pc.get("homeTeamOdds", {}).get("spreadOdds"):
                                espn_spread = pc.get("spread")
                                espn_ou = pc.get("overUnder")
                                espn_home_ml = pc.get("homeTeamOdds", {}).get("moneyLine")
                                espn_away_ml = pc.get("awayTeamOdds", {}).get("moneyLine")
                                break
                        if espn_spread is None:
                            for pc in pickcenter:
                                if pc.get("spread") is not None:
                                    espn_spread = pc.get("spread")
                                    espn_ou = pc.get("overUnder")
                                    espn_home_ml = pc.get("homeTeamOdds", {}).get("moneyLine")
                                    espn_away_ml = pc.get("awayTeamOdds", {}).get("moneyLine")
                                    break

                        margin = pred["ml_margin"]
                        win_prob = pred["ml_win_prob_home"]

                        row = {
                            "game_date": event_date_str,
                            "game_id": str(game_id),
                            "home_team_id": str(home_id),
                            "away_team_id": str(away_id),
                            "home_team_name": home_name,
                            "away_team_name": away_name,
                            "home_team": home_abbr,
                            "away_team": away_abbr,
                            "neutral_site": neutral,
                            "spread_home": round(margin, 1),
                            "win_pct_home": round(win_prob, 4),
                            "ml_win_prob_home": round(win_prob, 4),
                            "pred_home_score": pred.get("pred_home_score"),
                            "pred_away_score": pred.get("pred_away_score"),
                            "home_adj_em": audit.get("home_adj_em"),
                            "away_adj_em": audit.get("away_adj_em"),
                            "home_sos": audit.get("home_sos"),
                            "away_sos": audit.get("away_sos"),
                            "home_opp_fgpct": audit.get("home_opp_fgpct"),
                            "away_opp_fgpct": audit.get("away_opp_fgpct"),
                            "home_opp_threepct": audit.get("home_opp_threepct"),
                            "away_opp_threepct": audit.get("away_opp_threepct"),
                            "home_conference": audit.get("home_conference"),
                            "away_conference": audit.get("away_conference"),
                            # v19: All display stats from expanded audit
                            "home_ppg": audit.get("home_ppg"),
                            "away_ppg": audit.get("away_ppg"),
                            "home_opp_ppg": audit.get("home_opp_ppg"),
                            "away_opp_ppg": audit.get("away_opp_ppg"),
                            "home_fgpct": audit.get("home_fgpct"),
                            "away_fgpct": audit.get("away_fgpct"),
                            "home_threepct": audit.get("home_threepct"),
                            "away_threepct": audit.get("away_threepct"),
                            "home_ftpct": audit.get("home_ftpct"),
                            "away_ftpct": audit.get("away_ftpct"),
                            "home_tempo": audit.get("home_tempo"),
                            "away_tempo": audit.get("away_tempo"),
                            "home_wins": audit.get("home_wins"),
                            "away_wins": audit.get("away_wins"),
                            "home_losses": audit.get("home_losses"),
                            "away_losses": audit.get("away_losses"),
                            "home_assists": audit.get("home_assists"),
                            "away_assists": audit.get("away_assists"),
                            "home_turnovers": audit.get("home_turnovers"),
                            "away_turnovers": audit.get("away_turnovers"),
                            "home_steals": audit.get("home_steals"),
                            "away_steals": audit.get("away_steals"),
                            "home_blocks": audit.get("home_blocks"),
                            "away_blocks": audit.get("away_blocks"),
                            "home_orb_pct": audit.get("home_orb_pct"),
                            "away_orb_pct": audit.get("away_orb_pct"),
                            "home_fta_rate": audit.get("home_fta_rate"),
                            "away_fta_rate": audit.get("away_fta_rate"),
                            "home_ato_ratio": audit.get("home_ato_ratio"),
                            "away_ato_ratio": audit.get("away_ato_ratio"),
                            "home_form": audit.get("home_form"),
                            "away_form": audit.get("away_form"),
                            "home_rank": audit.get("home_rank"),
                            "away_rank": audit.get("away_rank"),
                            "confidence": "HIGH",
                            "result_entered": False,
                        }

                        # Market spread
                        if espn_spread is not None:
                            row["market_spread_home"] = espn_spread
                        if espn_ou is not None:
                            row["market_ou_total"] = espn_ou

                        # ML odds + edge (stored for single source of truth)
                        if espn_home_ml is not None:
                            row["market_home_ml"] = espn_home_ml
                        if espn_away_ml is not None:
                            row["market_away_ml"] = espn_away_ml
                        if espn_home_ml and espn_away_ml:
                            h_imp = abs(espn_home_ml) / (abs(espn_home_ml) + 100) if espn_home_ml < 0 else 100 / (espn_home_ml + 100)
                            a_imp = abs(espn_away_ml) / (abs(espn_away_ml) + 100) if espn_away_ml < 0 else 100 / (espn_away_ml + 100)
                            vig_total = h_imp + a_imp
                            h_true = h_imp / vig_total if vig_total > 0 else 0.5
                            ml_edge = win_prob - h_true
                            row["ml_edge_pct"] = round(abs(ml_edge) * 100, 2)
                            row["ml_bet_side"] = "HOME" if ml_edge >= 0 else "AWAY"

                        # O/U prediction (v26 direct + v4 triple agreement)
                        if pred.get("ou_predicted_total") is not None:
                            row["ou_total"] = pred["ou_predicted_total"]
                        if pred.get("ou_pick") is not None:
                            row["ou_pick"] = pred["ou_pick"]
                        if pred.get("ou_tier") is not None:
                            row["ou_tier"] = pred["ou_tier"]
                        if pred.get("ou_res_avg") is not None:
                            row["ou_res_avg"] = pred["ou_res_avg"]
                        if pred.get("ou_cls_avg") is not None:
                            row["ou_cls_avg"] = pred["ou_cls_avg"]
                        if pred.get("ou_ats_total") is not None:
                            row["ou_ats_total"] = pred["ou_ats_total"]
                        if pred.get("ou_edge") is not None:
                            row["ou_edge"] = pred["ou_edge"]

                        # ATS pick (v27) — with data quality gate + direction flip
                        if row.get("spread_home") is not None and row.get("market_spread_home") is not None:
                            model_margin = row["spread_home"]
                            mkt_implied = -row["market_spread_home"]
                            disagree = abs(model_margin - mkt_implied)
                            direction_flip = (model_margin > 0) != (mkt_implied > 0)  # model & market disagree on winner
                            row["ats_disagree"] = round(disagree, 2)

                            # DATA QUALITY GATE
                            fc_str = pred.get("feature_coverage", "0/1")
                            try:
                                fc_num, fc_den = fc_str.split("/")
                                fc_pct = int(fc_num) / max(int(fc_den), 1)
                            except (ValueError, AttributeError):
                                fc_pct = 0
                            has_key_stats = (
                                audit.get("home_adj_em") is not None and
                                pred.get("pred_home_score") is not None
                            )
                            data_quality_ok = fc_pct >= 0.50 or has_key_stats

                            # Direction flip at 3+ pts, same direction at 4+ pts
                            ats_threshold = 3 if direction_flip else 4
                            if disagree >= ats_threshold and data_quality_ok:
                                row["ats_side"] = "HOME" if model_margin > mkt_implied else "AWAY"
                                if direction_flip:
                                    row["ats_units"] = 3 if disagree >= 7 else (2 if disagree >= 5 else 1)
                                else:
                                    row["ats_units"] = 3 if disagree >= 10 else (2 if disagree >= 7 else 1)
                                row["ats_pick_spread"] = row["market_spread_home"]
                            else:
                                row["ats_units"] = 0
                                if not data_quality_ok and disagree >= ats_threshold:
                                    print(f"  [cron/ncaa] ⚠ {home_abbr}v{away_abbr}: {disagree:.1f}pt edge SUPPRESSED — low data ({fc_str}, adj_em={'yes' if audit.get('home_adj_em') else 'no'})")

                        # Upsert to Supabase
                        if mode == "refresh":
                            # PATCH existing row (refresh = update, not insert)
                            upsert_resp = _req.patch(
                                f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_id=eq.{game_id}",
                                headers={**headers, "Content-Type": "application/json"},
                                json=row,
                                timeout=30
                            )
                        else:
                            # POST new row (no merge-duplicates — it silently fails)
                            upsert_resp = _req.post(
                                f"{SUPABASE_URL}/rest/v1/ncaa_predictions",
                                headers={**headers, "Content-Type": "application/json"},
                                json=row,
                                timeout=30
                            )
                        if upsert_resp.status_code >= 400:
                            errors += 1
                            print(f"  [cron/ncaa] ⚠ Supabase save failed {game_id}: {upsert_resp.status_code} {upsert_resp.text[:200]}")
                            continue

                        if mode == "predict":
                            predicted += 1
                        else:
                            refreshed += 1

                except Exception as e:
                    errors += 1
                    print(f"  [cron/ncaa] Error predicting {game_id}: {e}")

            results["predicted"] = predicted
            results["refreshed"] = refreshed
            results["errors"] = errors

        elif mode == "grade":
            # Grade all ungraded games (today and recent days)
            graded = 0
            for days_ago in range(0, 7):  # check today + last 6 days
                check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                pending = _req.get(
                    f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_date=eq.{check_date}&result_entered=eq.false"
                    f"&select=id,game_id,home_team_id,away_team_id,win_pct_home,spread_home,market_spread_home,market_ou_total,ou_total,pred_home_score,pred_away_score,ats_units,ats_side",
                    headers=headers, timeout=30
                )
                pending_rows = pending.json() if pending.ok else []
                if not pending_rows:
                    continue

                # Fetch scores from ESPN
                compact_date = check_date.replace("-", "")
                espn_resp = _req.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={compact_date}&groups=50&limit=500",
                    timeout=30
                )
                events = espn_resp.json().get("events", []) if espn_resp.ok else []

                for event in events:
                    comp = event.get("competitions", [{}])[0]
                    status = comp.get("status", {}).get("type", {})
                    if not status.get("completed"):
                        continue

                    game_id = event.get("id")
                    matched = next((r for r in pending_rows if str(r.get("game_id")) == str(game_id)), None)
                    if not matched:
                        continue

                    home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
                    away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
                    if not home or not away:
                        continue

                    try:
                        home_score = int(home.get("score", 0))
                        away_score = int(away.get("score", 0))
                    except:
                        continue

                    actual_margin = home_score - away_score
                    model_picked_home = (matched.get("win_pct_home") or 0.5) >= 0.5
                    home_won = home_score > away_score
                    ml_correct = model_picked_home == home_won

                    # ATS (rl_correct)
                    mkt_spread = matched.get("market_spread_home")
                    rl_correct = None
                    if mkt_spread is not None:
                        ats_result = actual_margin + mkt_spread
                        if ats_result > 0:
                            rl_correct = True
                        elif ats_result < 0:
                            rl_correct = False

                    # O/U
                    total = home_score + away_score
                    ou_line = matched.get("market_ou_total") or matched.get("ou_total")
                    ou_model = matched.get("ou_total")  # model's predicted total
                    ou_correct = None
                    if ou_line and total != ou_line:
                        actual_over = total > ou_line
                        ou_correct = "OVER" if actual_over else "UNDER"
                    elif ou_line and total == ou_line:
                        ou_correct = "PUSH"

                    # O/U model correctness (did model's edge agree with outcome?)
                    ou_model_correct = None
                    if ou_model and ou_line and ou_correct and ou_correct != "PUSH":
                        try:
                            model_edge = float(ou_model) - float(ou_line)
                            if abs(model_edge) >= 5:  # only grade 5+ edge
                                model_says_over = model_edge > 0
                                actual_over = ou_correct == "OVER"
                                ou_model_correct = model_says_over == actual_over
                        except:
                            pass

                    # ATS pick grading (v27)
                    ats_correct = None
                    if matched.get("ats_units") and matched.get("ats_units") > 0 and matched.get("ats_side") and mkt_spread is not None:
                        ats_result = actual_margin + mkt_spread
                        if ats_result != 0:
                            home_covered = ats_result > 0
                            picked_home = matched["ats_side"] == "HOME"
                            ats_correct = picked_home == home_covered

                    patch = {
                        "actual_home_score": home_score,
                        "actual_away_score": away_score,
                        "result_entered": True,
                        "ml_correct": ml_correct,
                        "rl_correct": rl_correct,
                        "ou_correct": ou_correct,
                    }
                    if ats_correct is not None:
                        patch["ats_correct"] = ats_correct

                    _req.patch(
                        f"{SUPABASE_URL}/rest/v1/ncaa_predictions?id=eq.{matched['id']}",
                        headers={**headers, "Content-Type": "application/json"},
                        json=patch,
                        timeout=15
                    )
                    graded += 1

            results["graded"] = graded

            # ── Daily grading summary ──
            daily_summary = {}
            for days_ago in range(0, 3):
                check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                day_rows = _req.get(
                    f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_date=eq.{check_date}&result_entered=eq.true"
                    f"&select=ml_correct,ats_correct,ats_units,ou_correct,ou_total,market_ou_total,spread_home,market_spread_home",
                    headers=headers, timeout=30
                )
                rows = day_rows.json() if day_rows.ok else []
                if not rows:
                    continue

                ml_w = sum(1 for r in rows if r.get("ml_correct") is True)
                ml_l = sum(1 for r in rows if r.get("ml_correct") is False)
                ml_total = ml_w + ml_l

                # ATS: only games with ats_units > 0 (4+ edge bets)
                ats_rows = [r for r in rows if r.get("ats_units") and r["ats_units"] > 0]
                ats_w = sum(1 for r in ats_rows if r.get("ats_correct") is True)
                ats_l = sum(1 for r in ats_rows if r.get("ats_correct") is False)
                ats_total = ats_w + ats_l

                # O/U: games where model had 5+ edge vs market
                ou_w, ou_l = 0, 0
                for r in rows:
                    ou_model = r.get("ou_total")
                    ou_mkt = r.get("market_ou_total")
                    if ou_model and ou_mkt:
                        try:
                            ou_edge = float(ou_model) - float(ou_mkt)
                        except:
                            continue
                        if abs(ou_edge) >= 5:
                            oc = r.get("ou_correct")
                            if ou_edge > 0 and oc == "OVER":
                                ou_w += 1
                            elif ou_edge < 0 and oc == "UNDER":
                                ou_w += 1
                            elif oc and oc != "PUSH":
                                ou_l += 1
                ou_total = ou_w + ou_l

                day_summary = {
                    "ml": f"{ml_w}/{ml_total} ({ml_w/ml_total*100:.1f}%)" if ml_total else "0/0",
                    "ats_4plus": f"{ats_w}/{ats_total} ({ats_w/ats_total*100:.1f}%)" if ats_total else "0/0",
                    "ats_bets": ats_total,
                    "ou_5plus": f"{ou_w}/{ou_total} ({ou_w/ou_total*100:.1f}%)" if ou_total else "0/0",
                    "ou_bets": ou_total,
                    "total_games": len(rows),
                }
                daily_summary[check_date] = day_summary
                print(f"  [cron/ncaa] {check_date}: ML {day_summary['ml']} | ATS(4+) {day_summary['ats_4plus']} ({ats_total} bets) | O/U(5+) {day_summary['ou_5plus']} ({ou_total} bets)")

            results["daily_summary"] = daily_summary

            # ── Capture pickcenter data (spread movement) for graded games ──
            # ESPN pickcenter has open/close lines after games finish.
            # This feeds the spread_movement training feature.
            pc_captured = 0
            for days_ago in range(0, 3):
                check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                compact_date = check_date.replace("-", "")
                try:
                    espn_resp = _req.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={compact_date}&groups=50&limit=500",
                        timeout=30
                    )
                    day_events = espn_resp.json().get("events", []) if espn_resp.ok else []
                except:
                    day_events = []

                for event in day_events:
                    comp = event.get("competitions", [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed"):
                        continue
                    game_id = event.get("id")
                    try:
                        summary = _req.get(
                            f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
                            timeout=10
                        )
                        if not summary.ok:
                            continue
                        pc = summary.json().get("pickcenter", [])
                        if not pc:
                            continue

                        pc_patch = {}
                        for entry in pc:
                            ps = entry.get("pointSpread", {})
                            if ps:
                                home = ps.get("home", {})
                                h_open = home.get("open", {}).get("line")
                                h_close = home.get("close", {}).get("line")
                                if h_open is not None and h_close is not None:
                                    try:
                                        pc_patch["dk_spread_open"] = float(h_open)
                                        pc_patch["dk_spread_close"] = float(h_close)
                                        pc_patch["dk_spread_movement"] = round(float(h_close) - float(h_open), 1)
                                    except:
                                        pass
                            total = entry.get("total", {})
                            if total:
                                over = total.get("over", {})
                                o_open = over.get("open", {}).get("line", "")
                                o_close = over.get("close", {}).get("line", "")
                                try:
                                    o_open_val = float(str(o_open).replace("o", "").replace("u", ""))
                                    o_close_val = float(str(o_close).replace("o", "").replace("u", ""))
                                    pc_patch["dk_total_open"] = o_open_val
                                    pc_patch["dk_total_close"] = o_close_val
                                    pc_patch["dk_total_movement"] = round(o_close_val - o_open_val, 1)
                                except:
                                    pass
                            if pc_patch:
                                break

                        if pc_patch:
                            _req.patch(
                                f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}",
                                headers={**headers, "Content-Type": "application/json"},
                                json=pc_patch,
                                timeout=15
                            )
                            pc_captured += 1
                            _time.sleep(0.1)  # rate limit ESPN

                    except Exception as e:
                        pass  # non-critical, skip silently

            results["pickcenter_captured"] = pc_captured
            print(f"  [cron/ncaa] Pickcenter captured: {pc_captured} games")

        duration = _time.time() - start
        results["duration_sec"] = round(duration, 1)
        results["status"] = "complete"
        return jsonify(results)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _ncaa_cron_lock = False


# ═══════════════════════════════════════════════════════════════
# NCAA BACKFILL — compute advanced features for new games
# ═══════════════════════════════════════════════════════════════

@app.route("/cron/ncaa-backfill", methods=["GET", "POST"])
def route_ncaa_backfill():
    """Compute elo/form/pit_sos/etc. for ncaa_historical rows with NULLs.

    Runs the same logic as backfill_advanced_features.py but limited to
    recent NULL rows to stay within Railway's 5-minute timeout.
    """
    import time as _time, math
    import requests as _req
    import numpy as _np
    from collections import defaultdict, deque
    start = _time.time()

    WINDOW = 20; ELO_K = 32; FORM_DECAY = 0.1
    season = request.args.get("season", "2026")

    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

        # 1. Find rows needing backfill
        _sel = "id,game_date,home_team_id,away_team_id,actual_home_score,actual_away_score,home_rank,away_rank,neutral_site"
        r = _req.get(
            f"{SUPABASE_URL}/rest/v1/ncaa_historical?season=eq.{season}&actual_home_score=not.is.null&home_form=is.null"
            f"&select={_sel}&order=game_date.asc&limit=500",
            headers=headers, timeout=30)
        null_rows = r.json() if r.ok else []

        if not null_rows:
            return jsonify({"status": "ok", "message": "No NULL rows to backfill", "duration_sec": round(_time.time() - start, 1)})

        # 2. Pull ALL season games for chronological computation (paginated, 1000/page)
        # Use game_date range (indexed) instead of season filter (slow)
        _season_ranges = {
            "2026": ("2025-11-01", "2026-04-30"),
            "2025": ("2024-11-01", "2025-04-30"),
            "2024": ("2023-11-01", "2024-04-30"),
            "2023": ("2022-11-01", "2023-04-30"),
            "2022": ("2021-11-01", "2022-04-30"),
        }
        _date_start, _date_end = _season_ranges.get(season, (f"{int(season)-1}-11-01", f"{season}-04-30"))

        def _paginated_get(base_url):
            all_data, offset = [], 0
            while True:
                rr = _req.get(f"{base_url}&limit=1000&offset={offset}",
                    headers=headers, timeout=60)
                if not rr.ok: break
                page = rr.json()
                if not isinstance(page, list) or not page: break
                all_data.extend(page)
                if len(page) < 1000: break
                offset += 1000
                _time.sleep(0.5)
            return all_data

        all_games = _paginated_get(
            f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_date=gte.{_date_start}&game_date=lte.{_date_end}"
            f"&actual_home_score=not.is.null&select={_sel}&order=game_date.asc")

        # Also pull turnovers/steals for opp_to_rate
        stats_rows = _paginated_get(
            f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_date=gte.{_date_start}&game_date=lte.{_date_end}"
            f"&actual_home_score=not.is.null&select=id,home_turnovers,away_turnovers,home_steals,away_steals,home_tempo,away_tempo"
            f"&order=game_date.asc")
        stats_map = {sr["id"]: sr for sr in stats_rows}

        target_ids = set(r["id"] for r in null_rows)

        # 3. Process chronologically
        team_games = defaultdict(list)
        team_elo = defaultdict(lambda: 1500.0)
        team_home_m = defaultdict(lambda: deque(maxlen=WINDOW))
        team_away_m = defaultdict(lambda: deque(maxlen=WINDOW))
        team_opp_elos = defaultdict(lambda: deque(maxlen=WINDOW))
        updates = []

        for g in all_games:
            gid = g.get("id")
            hid = str(g.get("home_team_id", ""))
            aid = str(g.get("away_team_id", ""))
            hs = g.get("actual_home_score")
            aws = g.get("actual_away_score")
            if not hid or not aid or hs is None or aws is None:
                continue

            hs, aws = float(hs), float(aws)
            margin = hs - aws
            date = g.get("game_date", "")
            neutral = g.get("neutral_site", False)
            h_rank = int(g.get("home_rank") or 200)
            a_rank = int(g.get("away_rank") or 200)

            if gid in target_ids:
                patch = {}

                # Elo
                patch["home_elo"] = round(team_elo[hid], 1)
                patch["away_elo"] = round(team_elo[aid], 1)

                # Form
                for side, tid in [("home", hid), ("away", aid)]:
                    games = team_games[tid]
                    if games:
                        form = sum((1 if gm > 0 else -1) * math.exp(-FORM_DECAY * (len(games) - 1 - i))
                                   for i, gm in enumerate(games))
                        patch[f"{side}_form"] = round(form, 4)
                    else:
                        patch[f"{side}_form"] = 0.0

                    # Momentum halflife
                    recent = games[-10:]
                    if len(recent) >= 3:
                        w = sum(m * math.exp(-FORM_DECAY * (len(recent) - 1 - i)) for i, m in enumerate(recent))
                        ws = sum(math.exp(-FORM_DECAY * (len(recent) - 1 - i)) for i in range(len(recent)))
                        patch[f"{side}_momentum_halflife"] = round(w / max(ws, 1), 4)

                    # Blowout asym
                    if games:
                        bw = sum(1 for m in games if m >= 15)
                        bl = sum(1 for m in games if m <= -15)
                        patch[f"{side}_blowout_asym"] = round((bw - bl) / max(len(games), 1), 4)

                    # Overreaction
                    if len(games) >= 3:
                        diffs = [games[j] - games[j-1] for j in range(1, len(games))]
                        patch[f"{side}_overreaction"] = round(float(_np.std(diffs)), 4)

                    # Pit SOS
                    opp_elos = list(team_opp_elos[tid])
                    if len(opp_elos) >= 3:
                        patch[f"{side}_pit_sos"] = round(sum(opp_elos) / len(opp_elos), 1)

                    # Opp suppression
                    ranked = [gm for gm, rk in zip(games, [200]*len(games)) if True]  # simplified
                    # Use team_games metadata for ranked opponents — simplified for cron

                # Home/away margins
                hm = list(team_home_m[hid])
                am = list(team_away_m[hid])
                if len(hm) >= 3:
                    patch["home_home_margin"] = round(sum(hm) / len(hm), 4)
                if len(am) >= 3:
                    patch["home_away_margin"] = round(sum(am) / len(am), 4)
                hm_a = list(team_home_m[aid])
                am_a = list(team_away_m[aid])
                if len(hm_a) >= 3:
                    patch["away_home_margin"] = round(sum(hm_a) / len(hm_a), 4)
                if len(am_a) >= 3:
                    patch["away_away_margin"] = round(sum(am_a) / len(am_a), 4)

                # Opp PPG
                h_opp_scores = [float(gg["opp_score"]) for gg in team_games.get(f"{hid}_full", []) if gg.get("opp_score")]
                a_opp_scores = [float(gg["opp_score"]) for gg in team_games.get(f"{aid}_full", []) if gg.get("opp_score")]
                if len(h_opp_scores) >= 3:
                    patch["home_opp_ppg"] = round(sum(h_opp_scores) / len(h_opp_scores), 1)
                if len(a_opp_scores) >= 3:
                    patch["away_opp_ppg"] = round(sum(a_opp_scores) / len(a_opp_scores), 1)

                # opp_to_rate and to_conversion from stats
                st = stats_map.get(gid, {})
                a_to = st.get("away_turnovers"); h_to = st.get("home_turnovers")
                a_tempo = st.get("away_tempo"); h_tempo = st.get("home_tempo")
                h_steals = st.get("home_steals"); a_steals = st.get("away_steals")
                if a_to and a_tempo and float(a_tempo) > 0:
                    patch["home_opp_to_rate"] = round(float(a_to) / float(a_tempo), 4)
                if h_to and h_tempo and float(h_tempo) > 0:
                    patch["away_opp_to_rate"] = round(float(h_to) / float(h_tempo), 4)
                if h_steals and a_to and float(a_to) > 0:
                    patch["home_to_conversion"] = round(float(h_steals) / float(a_to), 4)
                if a_steals and h_to and float(h_to) > 0:
                    patch["away_to_conversion"] = round(float(a_steals) / float(h_to), 4)

                if patch:
                    updates.append((gid, patch))

            # Update accumulators AFTER (no leakage)
            h_elo, a_elo = team_elo[hid], team_elo[aid]
            exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400))
            act_h = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            team_elo[hid] += ELO_K * (act_h - exp_h)
            team_elo[aid] += ELO_K * ((1 - act_h) - (1 - exp_h))
            team_opp_elos[hid].append(a_elo)
            team_opp_elos[aid].append(h_elo)

            team_games[hid].append(margin)
            team_games[aid].append(-margin)
            team_games[f"{hid}_full"].append({"margin": margin, "opp_score": aws})
            team_games[f"{aid}_full"].append({"margin": -margin, "opp_score": hs})

            if not neutral:
                team_home_m[hid].append(margin)
                team_away_m[aid].append(-margin)

        # 4. Write updates
        success = 0
        for row_id, patch in updates:
            rr = _req.patch(
                f"{SUPABASE_URL}/rest/v1/ncaa_historical?id=eq.{row_id}",
                headers={**headers, "Content-Type": "application/json", "Prefer": "return=minimal"},
                json=patch, timeout=15)
            if rr.ok:
                success += 1

        duration = _time.time() - start
        return jsonify({
            "status": "ok",
            "null_rows_found": len(null_rows),
            "total_season_games": len(all_games),
            "updates_written": success,
            "duration_sec": round(duration, 1),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500



_nba_cron_lock = False

@app.route("/cron/nba-daily", methods=["GET", "POST"])
def route_nba_daily():
    import time as _time, traceback
    from datetime import datetime, timezone, timedelta
    global _nba_cron_lock
    if _nba_cron_lock:
        return jsonify({"status": "already_running"}), 429
    _nba_cron_lock = True
    start = _time.time()
    try:
        mode = request.args.get("mode", "auto")
        now_utc = datetime.now(timezone.utc)
        now_est = now_utc - timedelta(hours=5)
        today_est = request.args.get("date") or now_est.strftime("%Y-%m-%d")
        hour_est = now_est.hour
        if mode == "auto":
            mode = "predict" if hour_est < 17 else "grade"
        import requests as _req
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        results = {"mode": mode, "date": today_est, "hour_est": hour_est}
        _AM = {"GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS","WSH":"WAS","UTAH":"UTA","UTH":"UTA","PHO":"PHX","BKLYN":"BKN","BK":"BKN"}
        def _m(a): return _AM.get(a, a)

        if mode == "predict":
            compact = today_est.replace("-", "")
            espn_resp = _req.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={compact}&limit=50", timeout=15)
            events = espn_resp.json().get("events", []) if espn_resp.ok else []
            results["espn_games"] = len(events)
            if not events:
                results["status"] = "no_games_today"
                return jsonify(results)
            existing = _req.get(
                f"{SUPABASE_URL}/rest/v1/nba_predictions?game_date=eq.{today_est}"
                f"&select=id,game_id,ats_units,market_spread_home,market_ou_total,opening_home_ml,opening_away_ml",
                headers=headers, timeout=15)
            existing_map = {}  # game_id → {id, has_ats, market_*}
            if existing.ok:
                for r in existing.json():
                    if r.get("game_id"):
                        existing_map[str(r["game_id"])] = {
                            "id": r["id"],
                            "has_ats": r.get("ats_units") is not None,
                            "market_spread_home": r.get("market_spread_home"),
                            "market_ou_total": r.get("market_ou_total"),
                            "opening_home_ml": r.get("opening_home_ml"),
                            "opening_away_ml": r.get("opening_away_ml"),
                        }
            existing_ids = set(existing_map.keys())
            results["existing"] = len(existing_ids)
            predicted, errors = 0, 0

            # Collect games to predict + extract ESPN scoreboard odds (same pattern as MLB)
            games_to_predict = []
            odds_found = 0
            for event in events:
                comp = event.get("competitions", [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed"): continue
                state = comp.get("status", {}).get("type", {}).get("state", "")
                if state == "in": continue  # skip Live games
                game_id = str(event.get("id", ""))
                existing_info = existing_map.get(game_id)
                # Skip if already fully predicted (has ATS + market data) — unless force
                force = request.args.get("force", "").lower() in ("true", "1")
                if existing_info and existing_info.get("has_ats") and existing_info.get("market_spread_home") is not None and not force:
                    continue
                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home_c or not away_c: continue
                home_abbr = _m(home_c.get("team", {}).get("abbreviation", ""))
                away_abbr = _m(away_c.get("team", {}).get("abbreviation", ""))
                home_name = home_c.get("team", {}).get("displayName", "")
                away_name = away_c.get("team", {}).get("displayName", "")

                # ── Extract market odds from ESPN scoreboard (same as MLB) ──
                espn_odds = {}
                try:
                    odds_list = comp.get("odds", [])
                    if odds_list:
                        odds = odds_list[0]
                        if odds.get("overUnder") is not None:
                            espn_odds["market_ou_total"] = odds["overUnder"]
                        # Spread: odds.spread (NBA uses points, not run line)
                        if odds.get("spread") is not None:
                            espn_odds["market_spread_home"] = float(odds["spread"])
                        # Also check pointSpread block
                        ps = odds.get("pointSpread", {})
                        h_sp = ps.get("home", {}).get("close", {}).get("line")
                        if h_sp and "market_spread_home" not in espn_odds:
                            espn_odds["market_spread_home"] = float(h_sp)
                        # Moneylines
                        ml = odds.get("moneyline", {})
                        h_ml_str = ml.get("home", {}).get("close", {}).get("odds")
                        a_ml_str = ml.get("away", {}).get("close", {}).get("odds")
                        if h_ml_str:
                            espn_odds["opening_home_ml"] = int(str(h_ml_str).replace("+", ""))
                        if a_ml_str:
                            espn_odds["opening_away_ml"] = int(str(a_ml_str).replace("+", ""))
                        if espn_odds:
                            odds_found += 1
                except Exception as oe:
                    print(f"  [cron/nba] ESPN odds parse error {game_id}: {oe}")

                games_to_predict.append({
                    "game_id": game_id, "home_abbr": home_abbr, "away_abbr": away_abbr,
                    "home_name": home_name, "away_name": away_name, "game_date": today_est,
                    "espn_odds": espn_odds, "existing_info": existing_info,
                })
            results["games_to_predict"] = len(games_to_predict)
            results["espn_odds_found"] = odds_found

            def _predict_one(g):
                try:
                    pred = predict_nba_full({
                        "game_id": g["game_id"], "home_team": g["home_abbr"],
                        "away_team": g["away_abbr"], "game_date": g["game_date"],
                    })
                    if not pred or pred.get("error") or pred.get("ml_win_prob_home") is None:
                        return "empty"

                    margin = pred["ml_margin"]; wp = pred["ml_win_prob_home"]
                    audit = pred.get("audit_data", {})

                    row = {
                        "game_date": g["game_date"], "game_id": g["game_id"],
                        "home_team": g["home_abbr"], "away_team": g["away_abbr"],
                        "home_team_name": g["home_name"], "away_team_name": g["away_name"],
                        "spread_home": round(margin, 1), "win_pct_home": round(wp, 4),
                        "ml_win_prob_home": round(wp, 4),
                        "pred_home_score": pred.get("pred_home_score"),
                        "pred_away_score": pred.get("pred_away_score"),
                        "result_entered": False,
                        "ml_feature_coverage": pred.get("feature_coverage"),
                        "ml_model_type": pred.get("model_meta", {}).get("model_type"),
                        "home_ppg": audit.get("home_ppg"),
                        "away_ppg": audit.get("away_ppg"),
                        "home_opp_ppg": audit.get("home_opp_ppg"),
                        "away_opp_ppg": audit.get("away_opp_ppg"),
                        "home_net_rtg": audit.get("home_net_rtg"),
                        "away_net_rtg": audit.get("away_net_rtg"),
                        "home_pace": audit.get("home_pace"),
                        "away_pace": audit.get("away_pace"),
                        "home_wins": audit.get("home_wins"),
                        "away_wins": audit.get("away_wins"),
                        "home_losses": audit.get("home_losses"),
                        "away_losses": audit.get("away_losses"),
                    }

                    # ── Market odds from ESPN scoreboard (same pattern as MLB) ──
                    espn_odds = g.get("espn_odds", {})
                    existing_info = g.get("existing_info")

                    # 1. ESPN scoreboard odds (primary — from pre-fetched events)
                    if espn_odds.get("market_spread_home") is not None:
                        row["market_spread_home"] = espn_odds["market_spread_home"]
                    if espn_odds.get("market_ou_total") is not None:
                        row["market_ou_total"] = espn_odds["market_ou_total"]
                        row["ou_total"] = espn_odds["market_ou_total"]
                    if espn_odds.get("opening_home_ml") is not None:
                        row["opening_home_ml"] = espn_odds["opening_home_ml"]
                    if espn_odds.get("opening_away_ml") is not None:
                        row["opening_away_ml"] = espn_odds["opening_away_ml"]

                    # 2. Fallback: predict_nba_full ESPN pickcenter (if scoreboard missed)
                    if row.get("market_spread_home") is None:
                        mkt_sp = pred.get("market_spread")
                        if mkt_sp is not None and mkt_sp != 0:
                            row["market_spread_home"] = mkt_sp
                    if row.get("market_ou_total") is None:
                        mkt_tot = pred.get("market_total")
                        if mkt_tot is not None and mkt_tot != 0:
                            row["market_ou_total"] = mkt_tot
                            row["ou_total"] = mkt_tot

                    # 3. Preserve existing market data (if ESPN didn't return new)
                    if existing_info:
                        if row.get("market_spread_home") is None and existing_info.get("market_spread_home") is not None:
                            row["market_spread_home"] = existing_info["market_spread_home"]
                        if row.get("market_ou_total") is None and existing_info.get("market_ou_total") is not None:
                            row["market_ou_total"] = existing_info["market_ou_total"]
                        if row.get("opening_home_ml") is None and existing_info.get("opening_home_ml") is not None:
                            row["opening_home_ml"] = existing_info["opening_home_ml"]
                        if row.get("opening_away_ml") is None and existing_info.get("opening_away_ml") is not None:
                            row["opening_away_ml"] = existing_info["opening_away_ml"]

                    # ── ML edge (same as MLB) ──
                    nba_home_ml = row.get("opening_home_ml")
                    nba_away_ml = row.get("opening_away_ml")
                    if nba_home_ml and nba_away_ml:
                        h_imp = abs(nba_home_ml) / (abs(nba_home_ml) + 100) if nba_home_ml < 0 else 100 / (nba_home_ml + 100)
                        a_imp = abs(nba_away_ml) / (abs(nba_away_ml) + 100) if nba_away_ml < 0 else 100 / (nba_away_ml + 100)
                        vig_t = h_imp + a_imp
                        h_true = h_imp / vig_t if vig_t > 0 else 0.5
                        ml_edge = wp - h_true
                        row["ml_edge_pct"] = round(abs(ml_edge) * 100, 2)
                        row["ml_bet_side"] = "HOME" if ml_edge >= 0 else "AWAY"

                    # ── O/U v2 fields ──
                    for fld in ["ou_predicted_total","ou_edge","ou_pick","ou_tier","ou_res_avg"]:
                        if pred.get(fld) is not None: row[fld] = pred[fld]

                    # Player impact (BPM/VORP injury adjustment)
                    if pred.get("impact_adjustment"):
                        row["impact_adjustment"] = pred["impact_adjustment"]
                    if pred.get("missing_margin_diff"):
                        row["missing_margin_diff"] = pred["missing_margin_diff"]
                    if pred.get("home_out_players"):
                        row["home_out_players"] = pred["home_out_players"]
                    if pred.get("away_out_players"):
                        row["away_out_players"] = pred["away_out_players"]

                    # ── v28 ATS — residual model (CatBoost×0.7 + Lasso×0.3) ──
                    if pred.get("ats_side") and pred.get("ats_units", 0) > 0:
                        row["ats_side"] = pred["ats_side"]
                        row["ats_units"] = pred["ats_units"]
                        row["ats_pick_spread"] = pred.get("ats_pick_spread") or row.get("market_spread_home")
                    else:
                        row["ats_units"] = 0
                    # Store residual model metadata
                    if pred.get("ats_residual_blend") is not None:
                        row["ats_residual_blend"] = pred["ats_residual_blend"]
                    if pred.get("ats_residual_cb") is not None:
                        row["ats_residual_cb"] = pred["ats_residual_cb"]
                    if pred.get("ats_residual_lasso") is not None:
                        row["ats_residual_lasso"] = pred["ats_residual_lasso"]
                    if pred.get("ats_models_agree") is not None:
                        row["ats_models_agree"] = pred["ats_models_agree"]
                    row["ats_disagree"] = abs(pred.get("ats_residual_blend", 0) or 0)

                    # Confidence
                    row["confidence"] = pred.get("model_meta", {}).get("confidence") or (
                        "HIGH" if abs(margin) >= 7 else "MEDIUM" if abs(margin) >= 3 else "LOW")

                    # ── Data availability flags ──
                    row["lineup_available"] = bool(
                        pred.get("lineup_value_diff") or
                        pred.get("home_out_players") or
                        pred.get("away_out_players")
                    )

                    # ── Save: PATCH existing or POST new (same as MLB) ──
                    if existing_info:
                        patch_row = {k: v for k, v in row.items() if k not in ("game_date", "game_id")}
                        sv = _req.patch(f"{SUPABASE_URL}/rest/v1/nba_predictions?id=eq.{existing_info['id']}",
                            headers={**headers, "Content-Type": "application/json"},
                            json=patch_row, timeout=15)
                        if not sv.ok:
                            print(f"  [cron/nba] PATCH failed {g['game_id']}: {sv.status_code} {sv.text[:150]}")
                            return "error"
                    else:
                        sv = _req.post(f"{SUPABASE_URL}/rest/v1/nba_predictions",
                            headers={**headers, "Content-Type": "application/json",
                                     "Prefer": "return=representation"},
                            json=row, timeout=15)
                        if not sv.ok:
                            print(f"  [cron/nba] POST failed {g['game_id']}: {sv.status_code} {sv.text[:150]}")
                            return "error"

                    _mkt_str = f", mktSp={row.get('market_spread_home')}, mktOU={row.get('market_ou_total')}" if row.get("market_spread_home") is not None else ", NO ODDS"
                    _impact_str = f", impact={pred.get('impact_adjustment', 0):+.1f}" if pred.get("impact_adjustment") else ""
                    _ats_str = f", ATS={pred.get('ats_side')} {pred.get('ats_units', 0)}u (blend={pred.get('ats_residual_blend', 0):+.1f})" if pred.get("ats_units", 0) > 0 else ""
                    _ou_str = f", OU={row.get('ou_pick')} {row.get('ou_tier', 0)}u" if row.get("ou_pick") else ""
                    print(f"  [cron/nba] ✅ {g['away_abbr']}@{g['home_abbr']}: margin={margin:+.1f}, wp={wp:.3f}{_mkt_str}{_ats_str}{_ou_str}{_impact_str}")
                    return "ok"
                except Exception as e:
                    print(f"  [cron/nba] ❌ {g['game_id']}: {e}")
                    return "error"

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_predict_one, g): g for g in games_to_predict}
                for f in as_completed(futures):
                    r = f.result()
                    if r == "ok": predicted += 1
                    elif r == "error": errors += 1

            results["predicted"] = predicted; results["errors"] = errors

        elif mode == "grade":
            from nba_game_stats import process_completed_game
            graded, stats_extracted = 0, 0
            for days_ago in range(0, 7):
                check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                compact_date = check_date.replace("-", "")
                pending = _req.get(
                    f"{SUPABASE_URL}/rest/v1/nba_predictions?game_date=eq.{check_date}&result_entered=eq.false"
                    f"&select=id,game_id,home_team,away_team,win_pct_home,spread_home,market_spread_home,market_ou_total,pred_home_score,pred_away_score,ats_side,ats_units,ou_pick,ou_predicted_total",
                    headers=headers, timeout=15)
                pending_rows = pending.json() if pending.ok else []
                if not pending_rows: continue
                espn_resp = _req.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={compact_date}&limit=50", timeout=15)
                events = espn_resp.json().get("events", []) if espn_resp.ok else []
                for event in events:
                    comp = event.get("competitions", [{}])[0]
                    if not comp.get("status", {}).get("type", {}).get("completed"): continue
                    game_id = str(event.get("id", ""))
                    matched = next((r for r in pending_rows if str(r.get("game_id")) == game_id), None)
                    if not matched: continue
                    competitors = comp.get("competitors", [])
                    home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home_c or not away_c: continue
                    try: home_score = int(home_c.get("score", 0)); away_score = int(away_c.get("score", 0))
                    except: continue
                    actual_margin = home_score - away_score
                    ml_correct = ((matched.get("win_pct_home") or 0.5) >= 0.5) == (home_score > away_score)

                    # ── Backfill missing odds from ESPN scoreboard (same as predict) ──
                    mkt_spread = matched.get("market_spread_home")
                    ou_line = matched.get("market_ou_total")
                    odds_backfill = {}
                    try:
                        odds_list = comp.get("odds", [])
                        if odds_list:
                            odds = odds_list[0]
                            if mkt_spread is None:
                                sp = odds.get("spread")
                                if sp is None:
                                    ps = odds.get("pointSpread", {})
                                    sp_str = ps.get("home", {}).get("close", {}).get("line")
                                    if sp_str: sp = float(sp_str)
                                if sp is not None:
                                    mkt_spread = float(sp)
                                    odds_backfill["market_spread_home"] = mkt_spread
                            if ou_line is None and odds.get("overUnder") is not None:
                                ou_line = odds["overUnder"]
                                odds_backfill["market_ou_total"] = ou_line
                            # Moneylines
                            ml_block = odds.get("moneyline", {})
                            h_ml_str = ml_block.get("home", {}).get("close", {}).get("odds")
                            a_ml_str = ml_block.get("away", {}).get("close", {}).get("odds")
                            if h_ml_str and not matched.get("opening_home_ml"):
                                odds_backfill["closing_home_ml"] = int(str(h_ml_str).replace("+", ""))
                            if a_ml_str and not matched.get("opening_away_ml"):
                                odds_backfill["closing_away_ml"] = int(str(a_ml_str).replace("+", ""))
                    except Exception as oe:
                        print(f"  [cron/nba] grade odds parse error {game_id}: {oe}")

                    rl_correct = None
                    if mkt_spread is not None:
                        ats = actual_margin + mkt_spread
                        rl_correct = True if ats > 0 else (False if ats < 0 else None)
                    total = home_score + away_score
                    pred_total = (matched.get("pred_home_score") or 0) + (matched.get("pred_away_score") or 0)
                    ou_correct = None
                    if ou_line is not None and total != ou_line:
                        # Store actual result side — DailyBets compares ou_correct vs pick side
                        ou_correct = "OVER" if total > ou_line else "UNDER"
                    patch = {"actual_home_score": home_score, "actual_away_score": away_score,
                             "result_entered": True, "ml_correct": ml_correct, "rl_correct": rl_correct, "ou_correct": ou_correct,
                             **odds_backfill}

                    # ── ATS: compute pick on-the-fly if market data was just backfilled ──
                    ats_side = matched.get("ats_side")
                    ats_units = matched.get("ats_units") or 0
                    if not ats_side and mkt_spread is not None and matched.get("spread_home") is not None:
                        # Compute ATS pick now (wasn't computed at predict time due to missing odds)
                        model_margin = matched["spread_home"]
                        mkt_implied = -mkt_spread
                        disagree = abs(model_margin - mkt_implied)
                        patch["ats_disagree"] = round(disagree, 2)
                        if disagree >= 2:
                            ats_side = "HOME" if model_margin > mkt_implied else "AWAY"
                            ats_units = 3 if disagree >= 7 else (2 if disagree >= 4 else 1)
                            patch["ats_side"] = ats_side
                            patch["ats_units"] = ats_units
                            patch["ats_pick_spread"] = mkt_spread
                        else:
                            patch["ats_units"] = 0

                    # ── O/U: compute pick on-the-fly if just backfilled ──
                    if not matched.get("ou_pick") and ou_line is not None:
                        model_total = matched.get("ou_predicted_total") or pred_total
                        if model_total and model_total > 0:
                            ou_edge = model_total - ou_line
                            if abs(ou_edge) >= 4:
                                patch["ou_pick"] = "OVER" if ou_edge > 0 else "UNDER"
                                patch["ou_tier"] = 3 if abs(ou_edge) >= 10 else (2 if abs(ou_edge) >= 7 else 1)
                                patch["ou_edge"] = round(ou_edge, 1)

                    # ATS pick grading (did our specific bet win?)
                    if ats_units > 0 and ats_side and mkt_spread is not None:
                        ats_result = actual_margin + mkt_spread
                        if ats_result != 0:
                            home_covered = ats_result > 0
                            picked_home = ats_side == "HOME"
                            patch["ats_correct"] = picked_home == home_covered
                    _req.patch(f"{SUPABASE_URL}/rest/v1/nba_predictions?id=eq.{matched['id']}",
                        headers={**headers, "Content-Type": "application/json"}, json=patch, timeout=15)
                    graded += 1
                    try:
                        ha = _m(home_c.get("team", {}).get("abbreviation", ""))
                        aa = _m(away_c.get("team", {}).get("abbreviation", ""))
                        if process_completed_game(game_id, check_date, ha, aa, home_score, away_score, mkt_spread or 0):
                            stats_extracted += 1
                    except Exception as se:
                        print(f"  [cron/nba] stats error {game_id}: {se}")
            results["graded"] = graded; results["stats_extracted"] = stats_extracted

            # Recompute enrichment features for all teams after grading
            if stats_extracted > 0:
                try:
                    from nba_enrichment import recompute_all_enrichment
                    enriched = recompute_all_enrichment()
                    results["enrichment_updated"] = enriched
                except Exception as ee:
                    print(f"  [cron/nba] enrichment error: {ee}")
                    results["enrichment_error"] = str(ee)
                try:
                    from nba_enrichment import build_ref_profiles
                    ref_profiles = build_ref_profiles(min_games=10)
                    results["ref_profiles_updated"] = len(ref_profiles)
                except Exception as re:
                    print(f"  [cron/nba] ref profiles error: {re}")

        duration = _time.time() - start
        results["duration_sec"] = round(duration, 1)
        results["status"] = "complete"
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _nba_cron_lock = False


# ═══════════════════════════════════════════════════════════════
# ROUTES — NBA Player Impact Update (hoopR rolling BPM/VORP)
# Schedule: 1:00 AM EST (06:00 UTC) via GitHub Actions
# ═══════════════════════════════════════════════════════════════

@app.route("/cron/nba-player-update", methods=["GET", "POST"])
def route_nba_player_update():
    """Download latest hoopR box scores, recompute rolling player impact, upload to Supabase."""
    try:
        from nba_player_cron import run_player_update
        results = run_player_update()
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# ROUTES — MLB Daily Cron (grade completed games)
# Schedule: 12:30 AM EST (05:30 UTC) via GitHub Actions
# ═══════════════════════════════════════════════════════════════

_mlb_cron_lock = False

@app.route("/cron/mlb-daily", methods=["GET", "POST"])
def route_mlb_daily():
    """MLB daily cron — predict + grade.
    Modes: predict (create server-side predictions), grade (fill scores), auto (morning=predict, night=grade)
    """
    import time as _time, traceback
    from datetime import datetime, timezone, timedelta
    global _mlb_cron_lock
    if _mlb_cron_lock:
        return jsonify({"status": "already_running"}), 429
    _mlb_cron_lock = True
    start = _time.time()
    try:
        mode = request.args.get("mode", "auto")
        now_utc = datetime.now(timezone.utc)
        now_est = now_utc - timedelta(hours=5)
        today_est = now_est.strftime("%Y-%m-%d")
        hour_est = now_est.hour
        import requests as _req
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        results = {"mode": mode, "date": today_est, "hour_est": hour_est}

        # Auto: predict in morning, always grade
        if mode == "auto":
            mode = "predict" if 6 <= hour_est <= 14 else "grade"
            results["resolved_mode"] = mode

        # ── PREDICT: Create server-side predictions for today ──
        predicted = 0
        patched = 0
        predict_errors = []
        games_found = 0
        games_skipped = 0
        if mode in ("predict", "auto"):
            try:
                from mlb_full_predict import predict_mlb_full
                sched = _req.get(
                    f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_est}"
                    f"&hydrate=probablePitcher,teams,venue,linescore,officials", timeout=15)
                # Fetch existing predictions WITH id for PATCH support
                existing_resp = _req.get(
                    f"{SUPABASE_URL}/rest/v1/mlb_predictions?game_date=eq.{today_est}"
                    f"&select=id,game_pk,ats_units,market_ou_total,market_home_ml,market_away_ml,market_spread_home",
                    headers=headers, timeout=15)
                existing_map = {}  # game_pk → {id, has_ats, market_*}
                if existing_resp.ok:
                    for r in existing_resp.json():
                        if r.get("game_pk"):
                            existing_map[str(r["game_pk"])] = {
                                "id": r["id"],
                                "has_ats": r.get("ats_units") is not None,
                                "market_ou_total": r.get("market_ou_total"),
                                "market_home_ml": r.get("market_home_ml"),
                                "market_away_ml": r.get("market_away_ml"),
                                "market_spread_home": r.get("market_spread_home"),
                            }

                if sched.ok:
                    for dt in sched.json().get("dates", []):
                        for g in dt.get("games", []):
                            status = g.get("status", {}).get("abstractGameState", "")
                            if status in ("Final", "Live"):
                                games_skipped += 1
                                continue
                            gpk = g.get("gamePk")
                            force = request.args.get("force", "").lower() in ("true", "1")

                            # ── Smart timing: only predict games starting within 3 hours ──
                            if not force:
                                try:
                                    game_start = g.get("gameDate", "")
                                    if game_start:
                                        pass  # use datetime.fromisoformat
                                        start_dt = datetime.fromisoformat(game_start.replace("Z", "+00:00"))
                                        hours_until = (start_dt - now_utc).total_seconds() / 3600
                                        if hours_until > 3:
                                            games_skipped += 1
                                            continue
                                        if hours_until < 0:
                                            games_skipped += 1
                                            continue
                                except Exception:
                                    pass  # If parsing fails, proceed with prediction

                            # ── Starter gate: no starters = no prediction ──
                            h_sp = g.get("teams", {}).get("home", {}).get("probablePitcher", {})
                            a_sp = g.get("teams", {}).get("away", {}).get("probablePitcher", {})
                            if not h_sp.get("id") or not a_sp.get("id"):
                                if not force:
                                    print(f"  [mlb-cron] Skipping {gpk}: starters not announced")
                                    games_skipped += 1
                                    continue

                            games_found += 1
                            existing_info = existing_map.get(str(gpk))
                            # Skip if already fully predicted — unless force
                            if existing_info and existing_info.get("has_ats") and not force:
                                games_skipped += 1
                                continue
                            try:
                                predict_input = {"game_id": gpk, "game_date": today_est}
                                # Pass stored market data so predict doesn't rely on ESPN (may be stripped)
                                if existing_info:
                                    for mf in ["market_ou_total", "market_home_ml", "market_away_ml", "market_spread_home"]:
                                        if existing_info.get(mf):
                                            predict_input[mf] = existing_info[mf]
                                res = predict_mlb_full(predict_input)
                                if res and "error" not in res:
                                    margin = res.get("ml_margin", 0) or 0
                                    wp = res.get("ml_win_prob_home", 0.5) or 0.5
                                    pt = res.get("pred_total")
                                    ds = res.get("data_sources", {})
                                    total_base = pt if pt else 9.0
                                    row = {
                                        "game_date": today_est, "game_pk": gpk,
                                        "home_team": res["home_team"], "away_team": res["away_team"],
                                        "game_type": "R" if today_est >= "2026-03-27" else "S",
                                        "win_pct_home": round(wp, 4),
                                        "ml_win_prob_home": round(wp, 4),
                                        "spread_home": round(margin, 2),
                                        "confidence": "HIGH",
                                        "pred_home_runs": round(total_base / 2 + margin / 2, 2),
                                        "pred_away_runs": round(total_base / 2 - margin / 2, 2),
                                        "ou_total": round(pt, 1) if pt else 9.0,
                                        "ml_ou_pred_total": round(pt, 2) if pt else None,
                                        # Display stats from predict function
                                        "home_starter": res.get("home_starter"),
                                        "away_starter": res.get("away_starter"),
                                        "umpire": res.get("umpire"),
                                        "home_sp_fip": ds.get("home_sp_fip"),
                                        "away_sp_fip": ds.get("away_sp_fip"),
                                        "home_woba": ds.get("home_woba"),
                                        "away_woba": ds.get("away_woba"),
                                        "park_factor": ds.get("park_factor"),
                                        "home_team_era": ds.get("home_team_era"),
                                        "away_team_era": ds.get("away_team_era"),
                                        "is_dome": res.get("is_dome", False),
                                        "ml_feature_coverage": res.get("feature_coverage"),
                                    }
                                    # ── Market odds from ESPN scoreboard (match by team) ──
                                    # ESPN uses different abbreviations: CHW=CWS, WSN=WSH, AZ=ARI
                                    _ESPN_ALIAS = {"CHW":"CWS","WSN":"WSH","AZ":"ARI","CWS":"CWS","WSH":"WSH","ARI":"ARI"}
                                    try:
                                        espn_s = _req.get(
                                            f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_est.replace('-','')}",
                                            headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                                        if espn_s.ok:
                                            for ev in espn_s.json().get("events", []):
                                                comps = ev.get("competitions", [{}])[0].get("competitors", [])
                                                ev_home_raw = next((c.get("team", {}).get("abbreviation", "") for c in comps if c.get("homeAway") == "home"), "")
                                                ev_home = _ESPN_ALIAS.get(ev_home_raw.upper(), ev_home_raw.upper())
                                                if ev_home == res["home_team"].upper():
                                                    odds_list = ev.get("competitions", [{}])[0].get("odds", [])
                                                    if odds_list:
                                                        odds = odds_list[0]
                                                        if odds.get("overUnder") is not None:
                                                            row["market_ou_total"] = odds["overUnder"]
                                                        # Moneylines: odds.moneyline.home/away.close.odds
                                                        ml = odds.get("moneyline", {})
                                                        h_ml_str = ml.get("home", {}).get("close", {}).get("odds")
                                                        a_ml_str = ml.get("away", {}).get("close", {}).get("odds")
                                                        if h_ml_str:
                                                            row["market_home_ml"] = int(h_ml_str.replace("+", ""))
                                                        if a_ml_str:
                                                            row["market_away_ml"] = int(a_ml_str.replace("+", ""))
                                                        # Run line: odds.pointSpread.home/away.close.line
                                                        ps = odds.get("pointSpread", {})
                                                        h_rl = ps.get("home", {}).get("close", {}).get("line")
                                                        if h_rl:
                                                            row["run_line_home"] = float(h_rl)
                                                            row["market_spread_home"] = float(h_rl)
                                                    break
                                    except Exception as e:
                                        print(f"  [cron/mlb] ESPN odds error: {e}")

                                    # ── Preserve existing market data if ESPN didn't return new odds ──
                                    if existing_info:
                                        if not row.get("market_ou_total") and existing_info.get("market_ou_total"):
                                            row["market_ou_total"] = existing_info["market_ou_total"]
                                        if not row.get("market_home_ml") and existing_info.get("market_home_ml"):
                                            row["market_home_ml"] = existing_info["market_home_ml"]
                                        if not row.get("market_away_ml") and existing_info.get("market_away_ml"):
                                            row["market_away_ml"] = existing_info["market_away_ml"]
                                        if not row.get("market_spread_home") and existing_info.get("market_spread_home"):
                                            row["market_spread_home"] = existing_info["market_spread_home"]

                                    # ── Compute ML edge ──
                                    hml = row.get("market_home_ml")
                                    aml = row.get("market_away_ml")
                                    if hml and aml:
                                        h_imp = abs(hml) / (abs(hml) + 100) if hml < 0 else 100 / (hml + 100)
                                        a_imp = abs(aml) / (abs(aml) + 100) if aml < 0 else 100 / (aml + 100)
                                        vig_t = h_imp + a_imp
                                        h_true = h_imp / vig_t if vig_t > 0 else 0.5
                                        ml_edge = wp - h_true
                                        row["ml_edge_pct"] = round(abs(ml_edge) * 100, 2)
                                        row["ml_bet_side"] = "HOME" if ml_edge >= 0 else "AWAY"

                                    # ── ML bet signal (raw margin + ATS agreement) ──
                                    # Walk-forward validated (check highest tier first):
                                    #   2u: |margin| 2.5+ standalone → 81.4% ML (167 games)
                                    #   2u: |margin| 1.5+ AND ATS edge 1.0+ agree → 81.9% ML (171 games)
                                    #   1u: |margin| 1.0+ AND ATS edge 1.0+ agree → 80.4% ML (189 games)
                                    row["ml_bet_units"] = 0
                                    abs_margin = abs(margin)
                                    margin_side_home = margin > 0

                                    # Compute ATS edge direction agreement
                                    mkt_spread_ml = row.get("market_spread_home")
                                    ats_edge_val = 0
                                    ats_agrees = False
                                    if mkt_spread_ml is not None:
                                        mkt_implied = -float(mkt_spread_ml)
                                        ats_edge_val = abs(margin - mkt_implied)
                                        ats_side_home = margin > mkt_implied
                                        ats_agrees = (margin_side_home == ats_side_home) and ats_edge_val >= 1.0

                                    # 2u: margin 2.5+ standalone (81.4%)
                                    if abs_margin >= 2.5:
                                        row["ml_bet_units"] = 2
                                        row["ml_bet_side"] = "HOME" if margin_side_home else "AWAY"
                                    # 2u: margin 1.5+ AND ATS edge 1.0+ same direction (81.9%)
                                    elif abs_margin >= 1.5 and ats_agrees:
                                        row["ml_bet_units"] = 2
                                        row["ml_bet_side"] = "HOME" if margin_side_home else "AWAY"
                                    # 1u: margin 1.0+ AND ATS edge 1.0+ same direction (80.4%)
                                    elif abs_margin >= 1.0 and ats_agrees:
                                        row["ml_bet_units"] = 1
                                        row["ml_bet_side"] = "HOME" if margin_side_home else "AWAY"

                                    # ── ATS: v9 only — no fallback ──
                                    # If v9 sniper doesn't see enough edge, skip.
                                    # v8_fallback produced 29% ATS live — worse than coin flip.
                                    row["ats_units"] = 0
                                    row["ats_models_agree"] = res.get("models_agree")
                                    if res.get("ats_v9_units", 0) > 0:
                                        row["ats_side"] = res["ats_v9_side"]
                                        row["ats_units"] = res["ats_v9_units"]
                                        row["ats_disagree"] = res.get("ats_v9_edge", 0)
                                        row["ats_pick_spread"] = row.get("market_spread_home")
                                        row["ats_models_agree"] = res.get("ats_v9_models_agree", True)
                                        row["ats_model_version"] = "v9"

                                    # ── O/U from v3 model result ──
                                    # Always store these from predict result
                                    row["pred_total"] = round(pt, 2) if pt else None
                                    row["sp_form_combined"] = res.get("sp_form_combined")
                                    row["ou_edge"] = res.get("ou_edge")
                                    row["ou_pick"] = res.get("ou_pick")     # None clears stale picks
                                    row["ou_tier"] = res.get("ou_tier")
                                    row["ou_units"] = res.get("ou_units")
                                    row["ou_res_avg"] = res.get("ou_res_avg")
                                    row["lineup_delta_sum"] = res.get("lineup_delta_sum", 0)
                                    row["market_ou_total"] = res.get("market_ou_total") or row.get("market_ou_total")

                                    # ── Data availability flags ──
                                    row["lineup_available"] = res.get("lineup_available", False)

                                    # ── Save: PATCH existing or POST new ──
                                    if existing_info:
                                        # PATCH existing row — update with ATS/display/odds
                                        patch_row = {k: v for k, v in row.items() if k != "game_date" and k != "game_pk"}
                                        sv = _req.patch(
                                            f"{SUPABASE_URL}/rest/v1/mlb_predictions?id=eq.{existing_info['id']}",
                                            headers={**headers, "Content-Type": "application/json"},
                                            json=patch_row, timeout=15)
                                        if sv.ok:
                                            patched += 1
                                            ou_info = f" ou={row.get('ou_pick','—')}" if row.get("ou_pick") else ""
                                            print(f"  [cron/mlb] PATCHED {res['away_team']}@{res['home_team']}: m={margin:.1f} ats={row.get('ats_units', 0)}u{ou_info}")
                                    else:
                                        sv = _req.post(
                                            f"{SUPABASE_URL}/rest/v1/mlb_predictions",
                                            headers={**headers, "Content-Type": "application/json",
                                                     "Prefer": "return=representation"},
                                            json=row, timeout=15)
                                        if sv.ok:
                                            predicted += 1
                                            print(f"  [cron/mlb] Predicted {res['away_team']}@{res['home_team']}: m={margin:.1f} wp={wp:.3f} ats={row.get('ats_units', 0)}u")
                                        else:
                                            print(f"  [cron/mlb] POST failed {gpk}: {sv.status_code} {sv.text[:100]}")
                                            if len(predict_errors) < 3:
                                                predict_errors.append(f"POST {gpk}: {sv.status_code} {sv.text[:150]}")
                                else:
                                    err_msg = res.get("error", "no result") if res else "None returned"
                                    print(f"  [cron/mlb] predict_mlb_full error for {gpk}: {err_msg}")
                                    if len(predict_errors) < 3:
                                        predict_errors.append(f"{gpk}: {err_msg}")
                            except Exception as e:
                                print(f"  [cron/mlb] Predict error {gpk}: {e}")
                                if len(predict_errors) < 3:
                                    predict_errors.append(f"{gpk}: {str(e)[:150]}")
                else:
                    results["sched_status"] = sched.status_code if sched else "no_response"
            except Exception as e:
                print(f"  [cron/mlb] Predict mode error: {e}")
                results["predict_error"] = str(e)[:200]
            results["predicted"] = predicted
            results["patched"] = patched
            results["games_found"] = games_found
            results["games_skipped"] = games_skipped
            if predict_errors:
                results["predict_errors"] = predict_errors

        # ── GRADE: Fill final scores ──
        graded = 0
        for days_ago in range(0, 7):
            check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            pending_resp = _req.get(
                f"{SUPABASE_URL}/rest/v1/mlb_predictions?game_date=eq.{check_date}"
                f"&result_entered=eq.false"
                f"&select=id,game_pk,home_team,away_team,win_pct_home,spread_home,"
                f"market_spread_home,market_ou_total,ou_total,pred_home_runs,pred_away_runs,ml_ou_pred_total,"
                f"ats_units,ats_side,ml_bet_units,ml_bet_side",
                headers=headers, timeout=15)
            pending = pending_resp.json() if pending_resp.ok else []
            if not isinstance(pending, list) or not pending:
                continue
            schedule_resp = _req.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={check_date}&hydrate=linescore",
                timeout=15)
            if not schedule_resp.ok:
                continue
            schedule = schedule_resp.json()
            for date_obj in schedule.get("dates", []):
                for game in date_obj.get("games", []):
                    state = game.get("status", {}).get("abstractGameState", "")
                    if state != "Final":
                        continue
                    home_score = game.get("teams", {}).get("home", {}).get("score")
                    away_score = game.get("teams", {}).get("away", {}).get("score")
                    if home_score is None or away_score is None:
                        continue
                    game_pk = game.get("gamePk")
                    home_abbr = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                    away_abbr = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                    matched = None
                    for p in pending:
                        if p.get("game_pk") and str(p["game_pk"]) == str(game_pk):
                            matched = p; break
                        if (p.get("home_team", "").upper() == home_abbr.upper() and
                            p.get("away_team", "").upper() == away_abbr.upper()):
                            matched = p; break
                    if not matched:
                        continue
                    # ML
                    model_home = (matched.get("win_pct_home") or 0.5) >= 0.5
                    home_won = home_score > away_score
                    ml_correct = model_home == home_won if home_score != away_score else None
                    # RL (run line)
                    margin = home_score - away_score
                    mkt_spread = matched.get("market_spread_home")
                    rl_correct = None
                    if mkt_spread is not None:
                        ats_r = margin + float(mkt_spread)
                        rl_correct = True if ats_r > 0 else (False if ats_r < 0 else None)
                    else:
                        rl_correct = (margin > 1.5 if model_home else margin < -1.5) if abs(margin) > 1.5 else None
                    # O/U — prefer ML O/U model total, fall back to heuristic
                    total = home_score + away_score
                    ou_line = matched.get("market_ou_total") or matched.get("ou_total")
                    pred_total = (
                        float(matched["ml_ou_pred_total"]) if matched.get("ml_ou_pred_total")
                        else float(matched.get("pred_home_runs") or 0) + float(matched.get("pred_away_runs") or 0)
                    )
                    ou_correct = None
                    if ou_line and total != float(ou_line):
                        actual_over = total > float(ou_line)
                        # Always record the actual side — DailyBets compares ou_correct vs pick side
                        ou_correct = "OVER" if actual_over else "UNDER"
                    elif ou_line and total == float(ou_line):
                        ou_correct = "PUSH"
                    # ATS pick — compute model margin from pred runs or spread_home
                    # ATS pick grading — ONLY for games where we made a pick
                    ats_correct = None
                    if matched.get("ats_units") and matched["ats_units"] > 0 and matched.get("ats_side") and mkt_spread is not None:
                        ats_r = margin + float(mkt_spread)
                        if ats_r != 0:
                            home_covered = ats_r > 0
                            picked_home = matched["ats_side"] == "HOME"
                            ats_correct = picked_home == home_covered
                    patch = {
                        "actual_home_runs": home_score, "actual_away_runs": away_score,
                        "result_entered": True, "ml_correct": ml_correct,
                        "rl_correct": rl_correct, "ou_correct": ou_correct,
                    }
                    if ats_correct is not None:
                        patch["ats_correct"] = ats_correct
                    # ML bet grading (margin-based picks only)
                    if matched.get("ml_bet_units") and matched["ml_bet_units"] > 0 and matched.get("ml_bet_side"):
                        if home_score != away_score:
                            home_won = home_score > away_score
                            picked_home = matched["ml_bet_side"] == "HOME"
                            patch["ml_bet_correct"] = picked_home == home_won
                    _req.patch(f"{SUPABASE_URL}/rest/v1/mlb_predictions?id=eq.{matched['id']}",
                        headers={**headers, "Content-Type": "application/json"}, json=patch, timeout=15)
                    graded += 1
                    print(f"  [cron/mlb] {away_abbr}@{home_abbr}: {away_score}-{home_score} ml={'T' if ml_correct else 'F'}")
        results["graded"] = graded

        # AUDIT FIX: Refresh team rolling stats after grading new results
        if graded > 0:
            try:
                from mlb_rolling_stats import seed_all as _mlb_seed_rolling
                print("  [cron/mlb] Refreshing team rolling stats...")
                _mlb_seed_rolling()
                results["rolling_stats"] = "updated"
            except ImportError:
                results["rolling_stats"] = "skipped (module not deployed)"
            except Exception as e:
                print(f"  [cron/mlb] rolling stats error: {e}")
                results["rolling_stats"] = f"error: {str(e)[:100]}"

        # Refresh lineup rolling wOBA (pre-cache for v9 serve)
        if graded > 0:
            try:
                from mlb_lineup_cron import refresh_all_teams as _mlb_lineup_refresh
                print("  [cron/mlb] Refreshing lineup rolling wOBA...")
                lineup_result = _mlb_lineup_refresh()
                results["lineup_rolling"] = lineup_result
            except ImportError:
                results["lineup_rolling"] = "skipped (module not deployed)"
            except Exception as e:
                print(f"  [cron/mlb] lineup rolling error: {e}")
                results["lineup_rolling"] = f"error: {str(e)[:100]}"

        # ABS challenge data collection (runs after grading)
        try:
            from mlb_abs_collector import collect_date_range, build_team_stats, build_ump_stats, upload_to_supabase
            season_start = f"{now_est.year}-03-20"  # ABS data from spring training onward
            abs_games = collect_date_range(season_start, today_est)
            if abs_games:
                upload_to_supabase(build_team_stats(abs_games), build_ump_stats(abs_games))
                results["abs_data"] = f"updated ({len(abs_games)} games)"
            else:
                results["abs_data"] = "no games"
        except ImportError:
            results["abs_data"] = "skipped (module not deployed)"
        except Exception as e:
            print(f"  [cron/mlb] ABS collection error: {e}")
            results["abs_data"] = f"error: {str(e)[:100]}"

        duration = _time.time() - start
        results["duration_sec"] = round(duration, 1)
        results["status"] = "complete"
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _mlb_cron_lock = False


@app.route("/cron/mlb-rolling", methods=["GET", "POST"])
def route_mlb_rolling():
    """Manually refresh MLB team rolling stats + umpire profiles."""
    import time as _time
    start = _time.time()
    try:
        from mlb_rolling_stats import seed_all
        seed_all()
        return jsonify({"status": "complete", "duration_sec": round(_time.time() - start, 1)})
    except ImportError:
        return jsonify({"error": "mlb_rolling_stats.py not deployed"}), 500
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/cron/mlb-lineup-rolling", methods=["GET", "POST"])
def route_mlb_lineup_rolling():
    """Manually refresh MLB lineup rolling wOBA."""
    import time as _time
    start = _time.time()
    try:
        from mlb_lineup_cron import refresh_all_teams
        result = refresh_all_teams()
        result["duration_sec"] = round(_time.time() - start, 1)
        return jsonify(result)
    except ImportError:
        return jsonify({"error": "mlb_lineup_cron.py not deployed"}), 500
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/cron/ncaa-ats-record")
def route_ncaa_ats_record():
    """Live ATS record — no login required."""
    import requests as _req
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    r = _req.get(
        f"{SUPABASE_URL}/rest/v1/ncaa_predictions?ats_units=gt.0&result_entered=eq.true"
        f"&select=game_date,home_team,away_team,spread_home,market_spread_home,ats_disagree,ats_units,ats_side,ats_correct,actual_home_score,actual_away_score"
        f"&order=game_date.desc&limit=200",
        headers=headers, timeout=30
    )
    games = r.json() if r.ok else []

    record = {"1u": {"wins": 0, "losses": 0, "pushes": 0},
              "2u": {"wins": 0, "losses": 0, "pushes": 0},
              "3u": {"wins": 0, "losses": 0, "pushes": 0}}
    total_units_won = 0

    for g in games:
        tier = f"{g.get('ats_units', 0)}u"
        if tier not in record:
            continue
        if g.get("ats_correct") is True:
            record[tier]["wins"] += 1
            total_units_won += g["ats_units"] * 0.909  # -110 payout
        elif g.get("ats_correct") is False:
            record[tier]["losses"] += 1
            total_units_won -= g["ats_units"]
        else:
            record[tier]["pushes"] += 1

    for tier in record:
        t = record[tier]
        total = t["wins"] + t["losses"]
        t["ats_pct"] = round(t["wins"] / total, 4) if total > 0 else None
        t["total_graded"] = total

    return jsonify({
        "record": record,
        "total_units_profit": round(total_units_won, 2),
        "recent_picks": games[:20],
        "total_picks": len(games),
    })


# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════

@app.before_request
def _once():
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

@app.route("/backfill/nba-historical", methods=["POST"])
def route_backfill_nba():
    seasons = (request.get_json(force=True, silent=True) or {}).get("seasons", [2022, 2023, 2024, 2025])
    return jsonify({"error": "nba_backfill not available"})

@app.route("/backtest/quick/nba")
def route_quick_nba(): return jsonify(quick_backtest_nba())
@app.route("/backtest/quick/ncaa")
def route_quick_ncaa(): return jsonify(quick_backtest_ncaa())
@app.route("/backtest/quick/mlb")
def route_quick_mlb(): return jsonify(quick_backtest_mlb())
@app.route("/backtest/quick/all")
def route_quick_all():
    return jsonify({"mlb": quick_backtest_mlb(), "nba": quick_backtest_nba(), "ncaa": quick_backtest_ncaa()})

@app.route("/backtest/holdout/nba")
def route_holdout_nba():
    s = int(request.args.get("season", 2025))
    return jsonify(season_holdout_nba(test_season=s))
@app.route("/backtest/holdout/ncaa")
def route_holdout_ncaa():
    s = int(request.args.get("season", 2026))
    return jsonify(season_holdout_ncaa(test_season=s))
@app.route("/backtest/holdout/mlb")
def route_holdout_mlb():
    s = int(request.args.get("season", 2024))
    return jsonify(season_holdout_mlb(test_season=s))
@app.route("/backtest/holdout/all")
def route_holdout_all():
    return jsonify(season_holdout_all())
# force reload Sun Mar 15 10:55:32 PDT 2026
