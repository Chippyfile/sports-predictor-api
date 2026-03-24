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

# ── Import sport modules ──────────────────────────────────────
from sports.mlb import train_mlb, predict_mlb, calibrate_mlb_dispersion
from sports.nba import train_nba, predict_nba, nba_build_features
from sports.ncaa import train_ncaa, predict_ncaa, ncaa_build_features
from ncaa_full_predict import predict_ncaa_full
from nba_full_predict import predict_nba_full
from nba_game_stats import process_completed_games, backfill_game_stats
from sports.nfl import train_nfl, predict_nfl, nfl_build_features
from sports.ncaaf import train_ncaaf, predict_ncaaf, ncaaf_build_features
# REMOVED: nba_backfill cleaned from repo
from quick_backtest import quick_backtest_nba, quick_backtest_ncaa, quick_backtest_mlb
from season_holdout_backtest import season_holdout_nba, season_holdout_ncaa, season_holdout_mlb, season_holdout_all
from monte_carlo import monte_carlo
from cron import _active_sports, _log_training, _should_promote


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
            "GET  /cron/nba-daily?mode=predict|grade|auto",
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
def route_train_mlb(): return jsonify(train_mlb())

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

@app.route("/nba/backfill-stats", methods=["POST"])
def route_nba_backfill_stats():
    """Backfill nba_game_stats + nba_team_rolling from recent completed games."""
    data = request.get_json(force=True, silent=True) or {}
    days = data.get("days_back", 30)
    try:
        n = backfill_game_stats(days_back=days)
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
                        for pc in pickcenter:
                            if pc.get("homeTeamOdds", {}).get("spreadOdds"):
                                espn_spread = pc.get("spread")
                                espn_ou = pc.get("overUnder")
                                break
                        if espn_spread is None:
                            for pc in pickcenter:
                                if pc.get("spread") is not None:
                                    espn_spread = pc.get("spread")
                                    espn_ou = pc.get("overUnder")
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
                            "home_adj_em": audit.get("home_adj_em"),
                            "away_adj_em": audit.get("away_adj_em"),
                            "home_sos": audit.get("home_sos"),
                            "away_sos": audit.get("away_sos"),
                            "home_opp_fgpct": audit.get("home_opp_fgpct"),
                            "away_opp_fgpct": audit.get("away_opp_fgpct"),
                            "home_conference": audit.get("home_conference"),
                            "away_conference": audit.get("away_conference"),
                            "confidence": "HIGH",  # will be refined by frontend
                            "result_entered": False,
                        }

                        # Market spread
                        if espn_spread is not None:
                            row["market_spread_home"] = espn_spread
                        if espn_ou is not None:
                            row["market_ou_total"] = espn_ou

                        # O/U prediction (v26)
                        if pred.get("ou_predicted_total") is not None:
                            row["ou_total"] = pred["ou_predicted_total"]

                        # ATS pick (v27)
                        if row.get("spread_home") is not None and row.get("market_spread_home") is not None:
                            model_margin = row["spread_home"]
                            mkt_implied = -row["market_spread_home"]
                            disagree = abs(model_margin - mkt_implied)
                            row["ats_disagree"] = round(disagree, 2)
                            if disagree >= 4:
                                row["ats_side"] = "HOME" if model_margin > mkt_implied else "AWAY"
                                row["ats_units"] = 3 if disagree >= 10 else 2 if disagree >= 7 else 1
                                row["ats_pick_spread"] = row["market_spread_home"]
                            else:
                                row["ats_units"] = 0

                        # Upsert to Supabase
                        upsert_resp = _req.post(
                            f"{SUPABASE_URL}/rest/v1/ncaa_predictions",
                            headers={**headers, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
                            json=row,
                            timeout=30
                        )

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
            for days_ago in range(0, 3):  # check today + last 2 days
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
        today_est = now_est.strftime("%Y-%m-%d")
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
            existing = _req.get(f"{SUPABASE_URL}/rest/v1/nba_predictions?game_date=eq.{today_est}&select=game_id", headers=headers, timeout=15)
            existing_ids = {r["game_id"] for r in (existing.json() if existing.ok else [])}
            results["existing"] = len(existing_ids)
            predicted, errors = 0, 0
            for event in events:
                comp = event.get("competitions", [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed"): continue
                game_id = str(event.get("id", ""))
                if game_id in existing_ids: continue
                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home_c or not away_c: continue
                home_abbr = _m(home_c.get("team", {}).get("abbreviation", ""))
                away_abbr = _m(away_c.get("team", {}).get("abbreviation", ""))
                home_name = home_c.get("team", {}).get("displayName", "")
                away_name = away_c.get("team", {}).get("displayName", "")
                try:
                    pr = _req.post(f"http://localhost:{os.environ.get('PORT', 5000)}/predict/nba/full",
                        json={"game_id": game_id, "home_team": home_abbr, "away_team": away_abbr, "game_date": today_est}, timeout=30)
                    pred = pr.json() if pr.ok else None
                    if pred and not pred.get("error") and pred.get("ml_win_prob_home") is not None:
                        margin = pred["ml_margin"]; wp = pred["ml_win_prob_home"]
                        mkt_sp = pred.get("market_spread", 0); mkt_tot = pred.get("market_total", 0)
                        row = {"game_date": today_est, "game_id": game_id, "home_team": home_abbr, "away_team": away_abbr,
                               "home_team_name": home_name, "away_team_name": away_name,
                               "spread_home": round(margin, 1), "win_pct_home": round(wp, 4), "ml_win_prob_home": round(wp, 4),
                               "pred_home_score": pred.get("pred_home_score"), "pred_away_score": pred.get("pred_away_score"),
                               "result_entered": False}
                        if mkt_sp: row["market_spread_home"] = mkt_sp
                        if mkt_tot: row["market_ou_total"] = mkt_tot; row["ou_total"] = mkt_tot
                        if mkt_sp: row["ats_disagree"] = round(abs(margin - (-mkt_sp)), 2)
                        _req.post(f"{SUPABASE_URL}/rest/v1/nba_predictions",
                            headers={**headers, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
                            json=row, timeout=15)
                        predicted += 1
                except Exception as e:
                    errors += 1; print(f"  [cron/nba] predict error {game_id}: {e}")
            results["predicted"] = predicted; results["errors"] = errors

        elif mode == "grade":
            from nba_game_stats import process_completed_game
            graded, stats_extracted = 0, 0
            for days_ago in range(0, 3):
                check_date = (now_est - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                compact_date = check_date.replace("-", "")
                pending = _req.get(
                    f"{SUPABASE_URL}/rest/v1/nba_predictions?game_date=eq.{check_date}&result_entered=eq.false"
                    f"&select=id,game_id,home_team,away_team,win_pct_home,spread_home,market_spread_home,market_ou_total,pred_home_score,pred_away_score",
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
                    mkt_spread = matched.get("market_spread_home")
                    rl_correct = None
                    if mkt_spread is not None:
                        ats = actual_margin + mkt_spread
                        rl_correct = True if ats > 0 else (False if ats < 0 else None)
                    total = home_score + away_score
                    ou_line = matched.get("market_ou_total")
                    pred_total = (matched.get("pred_home_score") or 0) + (matched.get("pred_away_score") or 0)
                    ou_correct = None
                    if ou_line and total != ou_line:
                        ou_correct = "OVER" if (total > ou_line) == (pred_total > ou_line) else "UNDER"
                    patch = {"actual_home_score": home_score, "actual_away_score": away_score,
                             "result_entered": True, "ml_correct": ml_correct, "rl_correct": rl_correct, "ou_correct": ou_correct}
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

        duration = _time.time() - start
        results["duration_sec"] = round(duration, 1)
        results["status"] = "complete"
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        _nba_cron_lock = False


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
