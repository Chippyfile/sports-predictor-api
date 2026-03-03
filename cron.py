import json, traceback, time as _time, requests
from datetime import datetime
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get, save_model, load_model

def _active_sports():
    """Return list of sports currently in-season."""
    month = datetime.utcnow().month
    active = []
    if month in [3, 4, 5, 6, 7, 8, 9, 10]:      active.append("mlb")
    if month in [10, 11, 12, 1, 2, 3, 4, 5, 6]:  active.append("nba")
    if month in [11, 12, 1, 2, 3, 4]:             active.append("ncaa")
    if month in [9, 10, 11, 12, 1, 2]:            active.append("nfl")
    if month in [8, 9, 10, 11, 12, 1]:            active.append("ncaaf")
    return active


def _log_training(sport, status, result=None, error=None, duration=0.0, trigger="cron"):
    """Write a row to the training_log table in Supabase."""
    try:
        row = {
            "trigger": trigger,
            "sport": sport,
            "status": status,
            "n_train": result.get("n_train") if result else None,
            "mae_cv": result.get("mae_cv") if result else None,
            "model_type": result.get("model_type", "") if result else None,
            "promoted": result.get("_promoted", False) if result else False,
            "promote_reason": result.get("_promote_reason", "") if result else "",
            "details": json.dumps(result) if result else "{}",
            "error_message": str(error)[:500] if error else None,
            "duration_sec": round(duration, 2),
        }
        if result and "_mae_previous" in result:
            row["mae_previous"] = result["_mae_previous"]
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        requests.post(
            f"{SUPABASE_URL}/rest/v1/training_log",
            headers=headers, json=row, timeout=10,
        )
    except Exception as e:
        print(f"  [auto-train] Failed to log training: {e}")


def _should_promote(sport, new_result):
    """Compare new model MAE against current production model."""
    current = load_model(sport)
    if not current:
        return True, "no_existing_model", None
    current_mae = current.get("mae_cv")
    new_mae = new_result.get("mae_cv")
    if current_mae is None or new_mae is None:
        return True, "missing_mae_comparison", current_mae
    improvement = current_mae - new_mae
    if improvement >= -0.01:  # promote if equal or better (0.01 noise tolerance)
        return True, f"mae_delta_{improvement:+.4f}", current_mae
    else:
        return False, f"mae_regressed_{improvement:+.4f}", current_mae


def cron_auto_train():
    """
    Daily auto-training with shadow model comparison.
    Called by Railway cron at 4 AM ET (8:00 UTC).

    For each in-season sport:
      1. Train a new model on latest Supabase data
      2. Compare OOF MAE against current production model
      3. Promote only if new model is better (or no model exists)
      4. Log everything to training_log table

    Query params:
      ?force=true       — Train all 5 sports regardless of season
      ?sports=mlb,nba   — Override which sports to train
      ?trigger=manual   — Tag the log entry (default: cron)
    """
    import traceback
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
    promotions = []
    errors = []

    for sport in sports:
        fn = train_fns.get(sport)
        if not fn:
            results[sport] = {"status": "unknown_sport"}
            continue

        sport_start = _time.time()
        try:
            print(f"\n[auto-train] Training {sport.upper()}...")
            new_result = fn()
            duration = _time.time() - sport_start

            if "error" in new_result:
                results[sport] = {
                    "status": "skipped", "reason": new_result["error"],
                    "duration_sec": round(duration, 2),
                }
                _log_training(sport, "skipped", new_result, duration=duration, trigger=trigger)
                continue

            should_promote, reason, prev_mae = _should_promote(sport, new_result)
            new_result["_promoted"] = should_promote
            new_result["_promote_reason"] = reason
            if prev_mae is not None:
                new_result["_mae_previous"] = prev_mae

            if should_promote:
                promotions.append(sport)
                status = "promoted"
                print(f"  [auto-train] {sport.upper()}: PROMOTED ({reason})")
            else:
                status = "trained_not_promoted"
                print(f"  [auto-train] {sport.upper()}: NOT promoted ({reason})")

            results[sport] = {
                "status": status,
                "n_train": new_result.get("n_train"),
                "mae_cv": new_result.get("mae_cv"),
                "mae_previous": prev_mae,
                "model_type": new_result.get("model_type", ""),
                "promote_reason": reason,
                "duration_sec": round(duration, 2),
            }
            _log_training(sport, status, new_result, duration=duration, trigger=trigger)

        except Exception as e:
            duration = _time.time() - sport_start
            tb = traceback.format_exc()
            print(f"  [auto-train] {sport.upper()} ERROR: {e}")
            results[sport] = {
                "status": "error", "error": str(e),
                "traceback": tb, "duration_sec": round(duration, 2),
            }
            errors.append(sport)
            _log_training(sport, "error", error=e, duration=duration, trigger=trigger)

    # MLB dispersion recalibration
    if "mlb" in sports and "mlb" not in errors:
        try:
            print("\n[auto-train] Calibrating MLB dispersion...")
            results["mlb_dispersion"] = calibrate_mlb_dispersion()
        except Exception as e:
            results["mlb_dispersion"] = {"error": str(e)}

    # NCAA KenPom-style efficiency ratings (nightly)
    if "ncaa" in sports:
        try:
            print("\n[auto-train] Computing NCAA efficiency ratings...")
            eff_result = run_ncaa_efficiency_computation()
            results["ncaa_efficiency"] = {
                "status": "ok",
                "teams_rated": eff_result.get("teams_rated", 0),
                "iterations": eff_result.get("iterations", 0),
                "elapsed_sec": eff_result.get("elapsed_sec", 0),
            }
            print(f"[auto-train] NCAA ratings: {eff_result.get('teams_rated', 0)} teams rated")
        except Exception as e:
            results["ncaa_efficiency"] = {"status": "error", "error": str(e)}
            print(f"[auto-train] NCAA ratings error: {e}")

    total_duration = _time.time() - start
    summary = {
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat(),
        "trigger": trigger,
        "total_duration_sec": round(total_duration, 2),
        "sports_attempted": sports,
        "sports_promoted": promotions,
        "sports_errored": errors,
        "results": results,
    }

    _log_training("all", "complete", {
        "n_train": sum(r.get("n_train", 0) or 0 for r in results.values() if isinstance(r, dict)),
        "_promoted": len(promotions) > 0,
        "_promote_reason": f"promoted:{','.join(promotions)}" if promotions else "none",
    }, duration=total_duration, trigger=trigger)

    print(f"\n[auto-train] Done in {total_duration:.1f}s. Promoted: {promotions or 'none'}")
    return jsonify(summary)


def cron_status():
    """Model freshness, last run info, and in-season detection."""
    status = {}
    for sport in ["mlb", "nba", "ncaa", "nfl", "ncaaf"]:
        model = load_model(sport)
        if model:
            trained_at = model.get("trained_at", "unknown")
            try:
                trained_dt = datetime.fromisoformat(trained_at)
                age_hours = (datetime.utcnow() - trained_dt).total_seconds() / 3600
                freshness = "fresh" if age_hours < 26 else "stale" if age_hours < 72 else "very_stale"
            except Exception:
                age_hours = None
                freshness = "unknown"
            status[sport] = {
                "trained": True, "trained_at": trained_at,
                "age_hours": round(age_hours, 1) if age_hours else None,
                "freshness": freshness,
                "n_train": model.get("n_train"),
                "mae_cv": model.get("mae_cv"),
                "model_type": model.get("model_type", ""),
            }
        else:
            status[sport] = {"trained": False, "freshness": "no_model"}

    last_log = None
    try:
        rows = sb_get("training_log", "sport=eq.all&order=run_at.desc&limit=1")
        if rows:
            last_log = {
                "run_at": rows[0].get("run_at"),
                "status": rows[0].get("status"),
                "duration_sec": rows[0].get("duration_sec"),
            }
    except Exception:
        pass

    return jsonify({
        "active_sports": _active_sports(),
        "models": status,
        "last_cron_run": last_log,
        "next_run": "Daily at 08:00 UTC (4 AM ET)",
        "timestamp": datetime.utcnow().isoformat(),
    })


def route_train_all_logged():
    """Same as /train/all but with shadow comparison + Supabase logging."""
    with app.test_request_context(
        "/cron/auto-train?force=true&trigger=manual", method="POST"
    ):
        return cron_auto_train()


# ── Debug endpoint ─────────────────────────────────────────────
def debug_train_mlb():
    import traceback
    try:
        result = train_mlb()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500


# ═══════════════════════════════════════════════════════════════
# NCAA KENPOM-STYLE EFFICIENCY RATINGS
# ═══════════════════════════════════════════════════════════════
# Iterative opponent-adjusted efficiency computation for all D1 teams.
# POST /compute/ncaa-efficiency — triggers computation (~3-5 min)
# GET  /ratings/ncaa — returns current ratings from Supabase
#
# Supabase table: ncaa_team_ratings
#   team_id TEXT PRIMARY KEY, team_name TEXT, team_abbr TEXT,
#   conference TEXT, adj_oe REAL, adj_de REAL, adj_em REAL,
#   adj_ppg REAL, adj_opp_ppg REAL, adj_tempo REAL,
#   raw_oe REAL, raw_de REAL, raw_ppg REAL, raw_opp_ppg REAL,
#   sos REAL, wins INT, losses INT, games_used INT,
#   iterations INT, rank_adj_em INT,
#   updated_at TIMESTAMPTZ DEFAULT now()

ESPN_CBB_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
