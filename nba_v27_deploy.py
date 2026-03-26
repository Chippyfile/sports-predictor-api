#!/usr/bin/env python3
"""
nba_v27_deploy.py — Deployment checklist and predict function for NBA v27.

This file contains:
  1. predict_nba_v27() — drop-in replacement for predict_nba() in sports/nba.py
  2. Data sourcing requirements for nba_full_predict.py
  3. Deployment checklist

DEPLOYMENT STEPS:
  1. Run:   python3 nba_v27_train.py --deploy --validate
  2. Copy:  nba_v27_features_live.py → Railway repo root
  3. Patch: sports/nba.py predict_nba() → predict_nba_v27() (below)
  4. Patch: nba_full_predict.py to fetch enrichment + ref profiles + new game dict keys
  5. Push:  git add . && git commit -m "NBA v27: Lasso 38 features" && git push
  6. Test:  curl -X POST .../predict/nba/full -d '{"home_team_id":"...", ...}'
"""
import numpy as np
import pandas as pd
from db import load_model


def predict_nba_v27(game: dict):
    """
    NBA v27 prediction — Lasso on 38 features.

    Drop-in replacement for predict_nba() in sports/nba.py.
    Requires nba_v27_features_live.py in the import path.

    Args:
        game: Dict with all game data (stats, market, enrichment, refs).
              See DATA_REQUIREMENTS below for required keys.

    Returns:
        Dict with ml_margin, ml_win_prob_home, feature values, model meta.
    """
    bundle = load_model("nba")
    if not bundle:
        return {"error": "NBA model not trained — call /train/nba or run nba_v27_train.py --deploy"}

    # Import v27 feature builder
    from nba_v27_features_live import build_v27_features

    # Extract enrichment and ref data from game dict
    enrichment = {
        "home": game.pop("_home_enrichment", {}),
        "away": game.pop("_away_enrichment", {}),
    }
    ref_profile = game.pop("_ref_profile", {})

    # Build features
    X = build_v27_features(game, enrichment=enrichment, ref_profile=ref_profile)

    # Align columns to model's expected order
    for col in bundle["feature_cols"]:
        if col not in X.columns:
            X[col] = 0
    X = X[bundle["feature_cols"]]

    # Scale + predict
    X_s = bundle["scaler"].transform(X)
    margin = float(bundle["reg"].predict(X_s)[0])

    # Bias correction
    bias = bundle.get("bias_correction", 0.0)
    if bias:
        margin -= bias

    # Win probability (classifier + isotonic)
    win_prob = float(bundle["clf"].predict_proba(X_s)[0][1])
    isotonic = bundle.get("isotonic")
    if isotonic is not None:
        try:
            win_prob = float(isotonic.predict([win_prob])[0])
        except Exception:
            pass

    # Feature values for display
    feature_values = [
        {"feature": f, "value": round(float(X[f].iloc[0]), 4),
         "coef": round(float(bundle["reg"].coef_[i]), 4) if hasattr(bundle["reg"], "coef_") else 0}
        for i, f in enumerate(bundle["feature_cols"])
    ]
    feature_values.sort(key=lambda x: abs(x["coef"]), reverse=True)

    # Feature coverage — what % of features have non-default values
    n_nonzero = sum(1 for f in bundle["feature_cols"] if abs(float(X[f].iloc[0])) > 1e-6)
    coverage = n_nonzero / len(bundle["feature_cols"])

    return {
        "sport": "NBA",
        "model_version": "v27",
        "ml_margin": round(margin, 2),
        "ml_win_prob_home": round(win_prob, 4),
        "ml_win_prob_away": round(1 - win_prob, 4),
        "features": feature_values,
        "feature_coverage": round(coverage, 3),
        "model_meta": {
            "n_train": bundle.get("n_train"),
            "mae_cv": bundle.get("mae_cv"),
            "trained_at": bundle.get("trained_at"),
            "model_type": bundle.get("model_type"),
            "n_features": bundle.get("n_features"),
        },
    }


# ═══════════════════════════════════════════════════════════
# DATA REQUIREMENTS
# ═══════════════════════════════════════════════════════════
# These keys must be added to the game dict in nba_full_predict.py

REQUIRED_GAME_DICT_KEYS = """
# Keys already in current nba_full_predict.py game dict:
  home_ppg, away_ppg, home_opp_ppg, away_opp_ppg
  home_fgpct, away_fgpct, home_threepct, away_threepct
  home_ftpct, away_ftpct, home_steals, away_steals
  home_turnovers, away_turnovers, home_wins, home_losses
  away_wins, away_losses, home_net_rtg, away_net_rtg
  market_spread_home, market_ou_total
  home_days_rest, away_days_rest, home_opp_fgpct, away_opp_fgpct

# Keys that may need to be ADDED:
  espn_pregame_wp        — ESPN pregame predictor home win probability (0-1)
                           Source: ESPN game summary → predictor/winprobability
  crowd_pct              — attendance / venue capacity (0-1)
                           Source: ESPN game summary → attendance / venue.capacity
  home_last_result       — 1 if won, -1 if lost, 0 if unknown
  away_last_result       — same
  home_games_last_14     — count of games in last 14 days
  away_games_last_14     — same
  h2h_total_games        — head-to-head games this season
  away_team_abbr         — for public_team detection
  sharp_spread_signal    — from market/odds data (0 if unavailable)
  spread_juice_imbalance — from odds data (0 if unavailable)
  vig_uncertainty        — from odds data (0 if unavailable)
  home_moneyline         — for overround computation
  away_moneyline         — same

  # Rolling stats from nba_game_stats (last 5-10 games):
  home_roll_dreb, away_roll_dreb
  home_roll_paint_pts, away_roll_paint_pts
  home_roll_max_run, away_roll_max_run
  home_roll_fast_break, away_roll_fast_break
  home_roll_ft_trip_rate, away_roll_ft_trip_rate

  # Enrichment (passed as _home_enrichment / _away_enrichment dicts):
  lineup_value, scoring_hhi, ceiling, efg_pct, opp_efg_pct, ts_pct

  # Referee (passed as _ref_profile dict):
  home_whistle, foul_rate
"""

# ═══════════════════════════════════════════════════════════
# PATCH FOR nba_full_predict.py
# ═══════════════════════════════════════════════════════════
PATCH_NOTES = """
In nba_full_predict.py, add these data fetches after the existing parallel executor block:

    # ── v27: Fetch enrichment + ref profiles ──
    from db import sb_get

    # Team enrichment
    home_abbr = game.get("home_team_abbr", "")
    away_abbr = game.get("away_team_abbr", "")
    home_enr_rows = sb_get("nba_team_enrichment", f"team_abbr=eq.{home_abbr}&limit=1")
    away_enr_rows = sb_get("nba_team_enrichment", f"team_abbr=eq.{away_abbr}&limit=1")
    game["_home_enrichment"] = home_enr_rows[0] if home_enr_rows else {}
    game["_away_enrichment"] = away_enr_rows[0] if away_enr_rows else {}

    # Referee profile (from game officials)
    ref_name = game.get("referee_1", "")
    if ref_name:
        ref_rows = sb_get("nba_ref_profiles", f"ref_name=eq.{ref_name}&limit=1")
        game["_ref_profile"] = ref_rows[0] if ref_rows else {}
    else:
        game["_ref_profile"] = {}

    # ESPN pregame win probability
    # (may already be in game_info from _fetch_espn_game_info)
    if "espn_pregame_wp" not in game or game["espn_pregame_wp"] is None:
        game["espn_pregame_wp"] = game_info.get("predictor", {}).get("homeTeam", {}).get("gameProjection", 50) / 100

    # Rolling stats from nba_game_stats (average of last 5 games)
    for side, team_abbr in [("home", home_abbr), ("away", away_abbr)]:
        roll_rows = sb_get("nba_game_stats",
            f"team_abbr=eq.{team_abbr}&order=game_date.desc&limit=5&select=*")
        if roll_rows:
            import statistics
            for stat in ["dreb", "paint_pts", "max_run", "fast_break_pts", "ft_trip_rate"]:
                vals = [float(r.get(stat, 0) or 0) for r in roll_rows if r.get(stat)]
                game[f"{side}_roll_{stat}"] = statistics.mean(vals) if vals else 0
        else:
            for stat in ["dreb", "paint_pts", "max_run", "fast_break_pts", "ft_trip_rate"]:
                game[f"{side}_roll_{stat}"] = 0

Then replace:
    result = predict_nba(game)
With:
    from nba_v27_deploy import predict_nba_v27
    result = predict_nba_v27(game)
"""

if __name__ == "__main__":
    print("NBA v27 Deployment Guide")
    print("=" * 60)
    print("\nRequired game dict keys:")
    print(REQUIRED_GAME_DICT_KEYS)
    print("\nPatch notes for nba_full_predict.py:")
    print(PATCH_NOTES)
