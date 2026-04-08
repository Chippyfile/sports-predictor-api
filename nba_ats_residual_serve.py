"""
nba_ats_residual_serve.py — Serve-time residual ATS prediction
===============================================================
Loads nba_ats_residual model from Supabase, computes player features
at serve time, runs CatBoost + Lasso, blends, returns ATS pick + units.

Called by nba_full_predict.py after margin prediction.
"""

import numpy as np
import traceback

# ── Model cache ──
_ats_bundle = None
_ats_load_error = ""


def _load_ats_model():
    """Load nba_ats_residual bundle from Supabase (cached)."""
    global _ats_bundle, _ats_load_error
    if _ats_bundle is not None:
        return _ats_bundle
    try:
        from db import load_model
        _ats_bundle = load_model("nba_ats_residual")
        if _ats_bundle and _ats_bundle.get("models"):
            names = _ats_bundle.get("model_names", [])
            feats = _ats_bundle.get("feature_sets", [])
            print(f"  [ats_residual] Loaded: {names}, features: {[len(f) for f in feats]}")
            return _ats_bundle
        else:
            _ats_load_error = "bundle empty or no models"
            return None
    except Exception as e:
        _ats_load_error = str(e)
        print(f"  [ats_residual] Load error: {e}")
        return None


def compute_player_features_serve(home_abbr, away_abbr, home_out=None, away_out=None):
    """
    Compute player-level features at serve time from nba_player_impact table.
    Returns dict of features matching the training pipeline.
    """
    feats = {}
    try:
        from db import sb_get
        
        # Fetch all players for both teams
        home_players = sb_get("nba_player_impact",
                              f"team=eq.{home_abbr}&season=eq.2026&select=player_name,bpm,bpm_weighted,vorp,margin_impact,mpg,minutes_share,ts_pct&order=mpg.desc&limit=15") or []
        away_players = sb_get("nba_player_impact",
                              f"team=eq.{away_abbr}&season=eq.2026&select=player_name,bpm,bpm_weighted,vorp,margin_impact,mpg,minutes_share,ts_pct&order=mpg.desc&limit=15") or []
        
        for side, team_players, out_names in [("home", home_players, home_out or []),
                                               ("away", away_players, away_out or [])]:
            if not team_players:
                for k in ["missing_bpm", "missing_minutes", "lineup_continuity",
                           "starter_bpm", "star_dependency", "bench_bpm",
                           "starter_form", "minutes_load"]:
                    feats[f"{side}_{k}"] = 0
                continue
            
            out_lower = [n.lower() for n in out_names]
            
            # Expected starters: top 5 by minutes
            starters = team_players[:5]
            starter_names = [p["player_name"] for p in starters]
            
            # Missing player impact
            missing_bpm = 0
            missing_min = 0
            available_starters = []
            for p in starters:
                if p["player_name"].lower() in out_lower:
                    bpm = float(p.get("bpm", 0) or 0)
                    mpg = float(p.get("mpg", 0) or 0)
                    missing_bpm += bpm * min(mpg / 36.0, 1.0)
                    missing_min += mpg
                else:
                    available_starters.append(p)
            
            feats[f"{side}_missing_bpm"] = round(missing_bpm, 3)
            feats[f"{side}_missing_minutes"] = round(missing_min, 1)
            
            # Lineup continuity
            continuity = len(available_starters) / max(len(starters), 1)
            feats[f"{side}_lineup_continuity"] = round(continuity, 2)
            
            # Starter BPM (available starters only)
            starter_bpm = sum(float(p.get("bpm", 0) or 0) for p in available_starters)
            feats[f"{side}_starter_bpm"] = round(starter_bpm, 2)
            
            # Star dependency
            if team_players:
                total_abs_bpm = sum(abs(float(p.get("bpm", 0) or 0)) for p in team_players)
                best_abs_bpm = max(abs(float(p.get("bpm", 0) or 0)) for p in team_players)
                feats[f"{side}_star_dependency"] = round(best_abs_bpm / max(total_abs_bpm, 1), 3)
            else:
                feats[f"{side}_star_dependency"] = 0
            
            # Bench depth (players 6-10 avg BPM)
            bench = [p for p in team_players if p["player_name"] not in starter_names][:5]
            feats[f"{side}_bench_bpm"] = round(
                np.mean([float(p.get("bpm", 0) or 0) for p in bench]), 3) if bench else -2.0
            
            # Starter form (placeholder — would need rolling plus_minus from table)
            feats[f"{side}_starter_form"] = 0
            
            # Minutes load (top 7 avg mpg)
            top7 = team_players[:7]
            feats[f"{side}_minutes_load"] = round(
                np.mean([float(p.get("mpg", 0) or 0) for p in top7]), 1) if top7 else 0
        
        # Diffs
        feats["missing_bpm_diff"] = round(feats.get("home_missing_bpm", 0) - feats.get("away_missing_bpm", 0), 3)
        feats["lineup_continuity_diff"] = round(feats.get("home_lineup_continuity", 1) - feats.get("away_lineup_continuity", 1), 2)
        feats["starter_bpm_diff"] = round(feats.get("home_starter_bpm", 0) - feats.get("away_starter_bpm", 0), 2)
        feats["bench_depth_diff"] = round(feats.get("home_bench_bpm", 0) - feats.get("away_bench_bpm", 0), 3)
        feats["starter_form_diff"] = 0
        
    except Exception as e:
        print(f"  [ats_residual] player features error: {e}")
    
    return feats


def predict_ats_residual(feat_df, row, home_abbr, away_abbr,
                          home_out=None, away_out=None, market_spread=0):
    """
    Run the v28 residual ATS model. Returns dict with:
      ats_side, ats_units, ats_residual_blend, ats_residual_cb, ats_residual_lasso,
      ats_models_agree, ats_pick_spread
    
    feat_df: v27 feature DataFrame (1 row) from build_v27_features()
    row: raw data dict with team stats
    market_spread: home spread (e.g., -7.0)
    """
    result = {
        "ats_side": None, "ats_units": 0, "ats_residual_blend": None,
        "ats_residual_cb": None, "ats_residual_lasso": None,
        "ats_models_agree": None, "ats_pick_spread": None,
    }
    
    if not market_spread or abs(market_spread) < 0.1:
        return result
    
    bundle = _load_ats_model()
    if not bundle:
        return result
    
    try:
        models = bundle.get("models", [])
        scalers = bundle.get("scalers", [])
        feature_sets = bundle.get("feature_sets", [])
        model_names = bundle.get("model_names", [])
        weights = bundle.get("model_weights", [0.3, 0.7])  # [Lasso, CatBoost]
        
        if len(models) < 2 or len(feature_sets) < 2:
            return result
        
        # Compute player features at serve time
        player_feats = compute_player_features_serve(
            home_abbr, away_abbr, home_out, away_out)
        
        # Build feature vectors for each model
        predictions = []
        for i, (model, scaler, feat_list, name) in enumerate(
                zip(models, scalers, feature_sets, model_names)):
            
            fv = []
            for f in feat_list:
                # Check player features first, then feat_df, then row
                if f in player_feats:
                    fv.append(float(player_feats[f]))
                elif f in feat_df.columns:
                    fv.append(float(feat_df[f].iloc[0]))
                elif f in row:
                    fv.append(float(row.get(f, 0) or 0))
                else:
                    fv.append(0.0)
            
            X = np.array([fv])
            X_s = scaler.transform(X)
            pred = float(model.predict(X_s)[0])
            predictions.append(pred)
            
            if "Lasso" in name:
                result["ats_residual_lasso"] = round(pred, 2)
            elif "CatBoost" in name:
                result["ats_residual_cb"] = round(pred, 2)
        
        # Blend (weights are [Lasso, CatBoost])
        blend = sum(p * w for p, w in zip(predictions, weights))
        result["ats_residual_blend"] = round(blend, 2)
        
        # Agreement: both models agree on direction
        if len(predictions) >= 2:
            lasso_pred = predictions[0]  # first model is Lasso
            cb_pred = predictions[1]     # second is CatBoost
            agree = (lasso_pred > 0 and cb_pred > 0) or (lasso_pred < 0 and cb_pred < 0)
            result["ats_models_agree"] = agree
        
        # Unit sizing per validated tier structure:
        # 1u: blend ≥ 3
        # 2u: blend ≥ 4 AND agree
        # 3u: blend ≥ 5 AND agree
        abs_blend = abs(blend)
        if abs_blend >= 5 and result.get("ats_models_agree"):
            units = 3
        elif abs_blend >= 4 and result.get("ats_models_agree"):
            units = 2
        elif abs_blend >= 3:
            units = 1
        else:
            units = 0
        
        result["ats_units"] = units
        if units > 0:
            result["ats_side"] = "HOME" if blend > 0 else "AWAY"
            result["ats_pick_spread"] = market_spread
        
        n_nz = sum(1 for v in fv if abs(v) > 1e-6)  # last model's coverage
        print(f"  [ats_residual] blend={blend:+.2f}, cb={cb_pred:+.2f}, lasso={lasso_pred:+.2f}, "
              f"agree={agree}, units={units}, coverage={n_nz}/{len(feat_list)}")
        
    except Exception as e:
        print(f"  [ats_residual] Prediction error: {e}")
        print(f"  {traceback.format_exc()}")
    
    return result
