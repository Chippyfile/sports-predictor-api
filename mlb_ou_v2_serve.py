"""
mlb_ou_v2_serve.py — Serve-side MLB O/U v2 prediction
======================================================
Add these functions to sports/mlb.py (or import from here).

The v2 model uses sp_form_combined: combined pitcher deterioration signal.
At serve time, we compute this from the MLB Stats API game log endpoint.

Integration:
  1. In predict_mlb_ou(), after loading the bundle, check bundle.get("_v2_sp_form")
  2. If True, call predict_mlb_ou_v2() instead of the v1 path
  3. In mlb_full_predict.py, compute sp_form and pass it in the payload
"""
import numpy as np
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"


def fetch_pitcher_recent_form(pitcher_id, n_starts=3, timeout=8):
    """
    Fetch a pitcher's last N starts from MLB Stats API game log.
    Returns average runs allowed per start, or None if unavailable.
    
    Uses the current season game log. Each entry has:
      - stat.runs (runs allowed)
      - stat.inningsPitched
      - stat.gamesStarted (1 if started)
    """
    if not pitcher_id:
        return None
    
    try:
        # Get current season game log
        from datetime import datetime
        season = datetime.now().year
        url = (f"{MLB_API}/people/{pitcher_id}/stats"
               f"?stats=gameLog&season={season}&group=pitching")
        r = requests.get(url, timeout=timeout)
        if not r.ok:
            return None
        
        data = r.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        
        # Filter to starts only (gamesStarted > 0), most recent first
        starts = []
        for s in reversed(splits):
            stat = s.get("stat", {})
            if int(stat.get("gamesStarted", 0)) > 0:
                runs = float(stat.get("runs", 0))
                ip = float(stat.get("inningsPitched", "0").replace(".", ""))
                # inningsPitched format: "6.1" means 6⅓ innings
                # Actually MLB API returns it as a string like "6.1" = 6.333
                ip_str = str(stat.get("inningsPitched", "0"))
                if "." in ip_str:
                    whole, frac = ip_str.split(".")
                    ip = int(whole) + int(frac) / 3.0
                else:
                    ip = float(ip_str)
                
                starts.append({"runs": runs, "ip": ip})
                if len(starts) >= n_starts:
                    break
        
        if not starts:
            return None
        
        # Average runs per start (not per 9 IP — raw runs allowed)
        avg_runs = sum(s["runs"] for s in starts) / len(starts)
        return {
            "recent_avg_runs": round(avg_runs, 2),
            "n_starts": len(starts),
            "starts": starts,
        }
    except Exception as e:
        print(f"  [ou_v2] Game log fetch failed for {pitcher_id}: {e}")
        return None


def compute_sp_form_combined(home_pitcher_id, away_pitcher_id, home_fip, away_fip):
    """
    Compute sp_form_combined for a single game at serve time.
    
    sp_form_combined = (home_recent_ERA - home_season_FIP) + (away_recent_ERA - away_season_FIP)
    Positive = both pitchers trending worse than their season averages = UNDER signal.
    
    Returns: float (sp_form_combined), or 0.0 if data unavailable
    """
    sp_form = 0.0
    
    for pitcher_id, season_fip, label in [
        (home_pitcher_id, home_fip or 4.25, "home"),
        (away_pitcher_id, away_fip or 4.25, "away"),
    ]:
        form = fetch_pitcher_recent_form(pitcher_id)
        if form and form["n_starts"] >= 1:
            delta = form["recent_avg_runs"] - season_fip
            sp_form += delta
            print(f"  [ou_v2] {label} SP: last {form['n_starts']} starts avg {form['recent_avg_runs']:.1f} runs, "
                  f"FIP {season_fip:.2f}, delta={delta:+.2f}")
        else:
            print(f"  [ou_v2] {label} SP: no recent starts found, delta=0")
    
    return round(sp_form, 3)


def predict_mlb_ou_v2(game, bundle):
    """
    V2 O/U prediction using sp_form residual model.
    
    Args:
        game: dict with keys from mlb_full_predict payload:
            - market_ou_total: float (Vegas O/U line)
            - home_sp_fip, away_sp_fip: float (starter season FIP)
            - home_starter_id, away_starter_id: int (MLB player IDs)
            - temp_f: float (game-time temp, optional)
            - wind_out_flag: int (0/1, optional)
            - sp_form_combined: float (if pre-computed, skip API calls)
        bundle: loaded model bundle with _v2_sp_form flag
    
    Returns: dict with pred_total, ou_edge, ou_pick, ou_units, ou_tier, residual, etc.
    """
    market_total = float(game.get("market_ou_total", 0) or 0)
    
    # ── Compute sp_form_combined ──
    sp_form = game.get("sp_form_combined")
    if sp_form is None:
        sp_form = compute_sp_form_combined(
            game.get("home_starter_id"),
            game.get("away_starter_id"),
            game.get("home_sp_fip", 4.25),
            game.get("away_sp_fip", 4.25),
        )
    
    # ── Build feature vector (model expects 4 features even though Lasso zeroed 3) ──
    feature_vals = {
        "market_total": market_total if market_total > 0 else 9.0,
        "sp_form_combined": float(sp_form),
        "real_temp_f": float(game.get("temp_f", 72.0) or 72.0),
        "real_wind_out": int(game.get("wind_out_flag", 0) or 0),
    }
    
    feature_cols = bundle["feature_cols"]
    X = np.array([[feature_vals.get(f, 0) for f in feature_cols]])
    
    # ── Scale and predict residual ──
    X_scaled = bundle["scaler"].transform(X)
    residual = float(bundle["model_res"].predict(X_scaled)[0])
    residual -= bundle.get("bias_correction", 0.0)
    
    # ── ATS implied total ──
    ats_home = float(bundle["model_ats_home"].predict(X_scaled)[0])
    ats_away = float(bundle["model_ats_away"].predict(X_scaled)[0])
    ats_total = ats_home + ats_away
    ats_edge = ats_total - market_total if market_total > 0 else 0
    
    # ── Predicted total ──
    pred_total = (market_total + residual) if market_total > 0 else (ats_total if ats_total > 0 else 9.0)
    pred_total = max(4.0, min(18.0, pred_total))
    
    # ── Determine pick and tier using thresholds from bundle ──
    under_thresholds = bundle.get("under_thresholds", {1: -0.3, 2: -0.3, 3: -0.8})
    ats_threshold = bundle.get("ats_threshold", -0.8)
    over_thresholds = bundle.get("over_thresholds", {1: 0.5})
    
    ou_pick = None
    ou_tier = 0
    ou_units = 0
    
    if market_total > 0:
        # ── UNDER tiers (check highest first) ──
        # 3u: residual ≤ -0.8 (standalone)
        if residual <= under_thresholds.get(3, -0.8):
            ou_pick = "UNDER"
            ou_tier = 3
            ou_units = 3
        # 2u: residual ≤ -0.3 AND ATS edge ≤ -0.8 (double agreement)
        elif residual <= under_thresholds.get(2, -0.3) and ats_edge <= ats_threshold:
            ou_pick = "UNDER"
            ou_tier = 2
            ou_units = 2
        # 1u: residual ≤ -0.3 (standalone)
        elif residual <= under_thresholds.get(1, -0.3):
            ou_pick = "UNDER"
            ou_tier = 1
            ou_units = 1
        
        # ── OVER tiers ──
        if ou_pick is None:
            for tier in sorted(over_thresholds.keys(), reverse=True):
                if residual >= over_thresholds[tier]:
                    ou_pick = "OVER"
                    ou_tier = tier
                    ou_units = tier
                    break
    
    ou_edge = residual  # for display: how far from market
    
    return {
        "sport": "MLB",
        "type": "ou_v2",
        "pred_total": round(pred_total, 2),
        "market_total": round(market_total, 1) if market_total > 0 else None,
        "ou_edge": round(ou_edge, 3),
        "ou_pick": ou_pick,
        "ou_units": ou_units,
        "ou_tier": ou_tier,
        "residual": round(residual, 3),
        "sp_form_combined": round(sp_form, 3),
        "ats_total": round(ats_total, 2),
        "ats_edge": round(ats_edge, 2),
        "model_meta": {
            "model_type": bundle.get("model_type", "mlb_ou_v2"),
            "n_train": bundle.get("n_train", 0),
            "mae_cv": bundle.get("mae_cv", 0),
            "res_alpha": bundle.get("res_alpha"),
            "ats_alpha": bundle.get("ats_alpha"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# INTEGRATION PATCH for sports/mlb.py predict_mlb_ou()
# ═══════════════════════════════════════════════════════════════
#
# Add this at the TOP of predict_mlb_ou(), before any v1 logic:
#
#     bundle = load_model("mlb_ou")
#     if not bundle:
#         return {"error": "No MLB O/U model found"}
#
#     # ── V2: SP Form residual model ──
#     if bundle.get("_v2_sp_form"):
#         from mlb_ou_v2_serve import predict_mlb_ou_v2
#         return predict_mlb_ou_v2(game, bundle)
#
#     # ... existing v1 code below ...
#
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# INTEGRATION PATCH for mlb_full_predict.py predict_mlb_full()
# ═══════════════════════════════════════════════════════════════
#
# Before calling predict_mlb_ou(payload), add starter IDs to the payload:
#
#     payload["home_starter_id"] = h_starter_id    # already fetched from schedule
#     payload["away_starter_id"] = a_starter_id    # already fetched from schedule
#
# The v2 model will use these to fetch game logs and compute sp_form.
# If IDs aren't available, sp_form defaults to 0 (no signal = no pick).
#
# ═══════════════════════════════════════════════════════════════
