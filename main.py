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
from sports.nfl import train_nfl, predict_nfl, nfl_build_features
from sports.ncaaf import train_ncaaf, predict_ncaaf, ncaaf_build_features
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
        ],
    })

@app.route("/health")
def health():
    trained = [s for s in ["mlb", "nba", "ncaa", "nfl", "ncaaf"] if load_model(s)]
    disp = load_model("mlb_dispersion")
    return jsonify({
        "status": "healthy",
        "trained_models": trained,
        "mlb_dispersion": disp if disp else "not calibrated — POST /calibrate/mlb",
        "timestamp": datetime.utcnow().isoformat(),
    })


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
    return jsonify(fn(game))

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
    bundle = load_model(sport.lower())
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
        info["dispersion"] = load_model("mlb_dispersion")
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
        bundle = load_model(sport)
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

from backtests import (
    route_backtest_mlb, route_backtest_nba, route_backtest_ncaa,
    route_model_info as _backtest_model_info,
    nba_confidence_calibration, route_backtest_current_model,
)

# Only register if they exist (graceful degradation)
import inspect
_backtest_routes = {
    "/backtest/mlb": (["GET", "POST"], "route_backtest_mlb", route_backtest_mlb),
    "/backtest/nba": (["GET", "POST"], "route_backtest_nba", route_backtest_nba),
    "/backtest/ncaa": (["POST"], "route_backtest_ncaa", route_backtest_ncaa),
    "/backtest/nba-confidence": (["GET"], "nba_confidence", nba_confidence_calibration),
    "/backtest/mlb/current-model": (["POST"], "backtest_mlb_current", route_backtest_current_model),
}
for path, (methods, name, fn) in _backtest_routes.items():
    try:
        app.add_url_rule(path, name, fn, methods=methods)
    except Exception as e:
        print(f"  [routes] Could not register {path}: {e}")


# ═══════════════════════════════════════════════════════════════
# ROUTES — NCAA Ratings
# ═══════════════════════════════════════════════════════════════

from ncaa_ratings import run_ncaa_efficiency_computation

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
    return jsonify(rows)

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
# STARTUP
# ═══════════════════════════════════════════════════════════════

@app.before_request
def _once():
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
