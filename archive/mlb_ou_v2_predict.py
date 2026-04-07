"""
mlb_ou_v2_predict.py — Serve-side prediction for MLB O/U v2 triple agreement
=============================================================================
Drop this into predict_mlb_ou() in sports/mlb.py when v2 bundle is detected.

Detection: bundle.get("_v2_triple") == True

The bundle contains:
  - res_scaler, cls_scaler, ats_scaler (StandardScaler)
  - res_models (3), cls_models (2), ats_home_models (3), ats_away_models (3)
  - res_feature_cols, cls_feature_cols, ats_feature_cols
  - under_tiers, over_tiers (threshold dicts)
"""
import numpy as np


def predict_ou_v2(features_dict, market_ou_total, ou_bundle):
    """
    Predict O/U using triple agreement system (MLB v2).

    Args:
        features_dict: dict of feature_name → value (from build_ou_features)
        market_ou_total: float, the market O/U line
        ou_bundle: loaded model bundle from Supabase

    Returns:
        dict with keys: direction, tier, res_avg, cls_avg, ats_total, ats_edge, pred_total
    """
    if market_ou_total is None or market_ou_total < 3:
        return {"direction": None, "tier": 0, "res_avg": 0, "cls_avg": 0.5,
                "ats_total": 0, "ats_edge": 0, "pred_total": market_ou_total or 9.0}

    # Extract feature arrays
    res_feats = ou_bundle["res_feature_cols"]
    cls_feats = ou_bundle["cls_feature_cols"]
    ats_feats = ou_bundle["ats_feature_cols"]

    res_vals = np.array([[features_dict.get(f, 0) for f in res_feats]])
    cls_vals = np.array([[features_dict.get(f, 0) for f in cls_feats]])
    ats_vals = np.array([[features_dict.get(f, 0) for f in ats_feats]])

    # Scale
    res_scaled = ou_bundle["res_scaler"].transform(res_vals)
    cls_scaled = ou_bundle["cls_scaler"].transform(cls_vals)
    ats_scaled = ou_bundle["ats_scaler"].transform(ats_vals)

    # Predict residual (3 models averaged) — positive = over expected
    res_preds = [m.predict(res_scaled)[0] for m in ou_bundle["res_models"]]
    res_avg = float(np.mean(res_preds))

    # Predict P(under) (2 models averaged)
    cls_preds = [m.predict_proba(cls_scaled)[0, 1] for m in ou_bundle["cls_models"]]
    cls_avg = float(np.mean(cls_preds))

    # Predict ATS-implied total (home + away)
    home_preds = [m.predict(ats_scaled)[0] for m in ou_bundle["ats_home_models"]]
    away_preds = [m.predict(ats_scaled)[0] for m in ou_bundle["ats_away_models"]]
    ats_total = float(np.mean(home_preds) + np.mean(away_preds))
    ats_edge = ats_total - market_ou_total

    # Implied predicted total from residual
    pred_total = market_ou_total + res_avg

    # ── Determine direction and tier via triple agreement ──
    under_tiers = ou_bundle.get("under_tiers", {})
    over_tiers = ou_bundle.get("over_tiers", {})

    direction = None
    tier = 0

    # Check UNDER tiers (highest first)
    for t in sorted(under_tiers.keys(), reverse=True):
        th = under_tiers[t]
        if (res_avg <= th["res_avg"] and
            cls_avg >= th["cls_avg"] and
            ats_edge <= th["ats_edge"]):
            direction = "UNDER"
            tier = t
            break

    # If no UNDER, check OVER tiers
    if direction is None:
        for t in sorted(over_tiers.keys(), reverse=True):
            th = over_tiers[t]
            if (res_avg >= th["res_avg"] and
                cls_avg <= th["cls_avg"] and
                ats_edge >= th["ats_edge"]):
                direction = "OVER"
                tier = t
                break

    return {
        "direction": direction,
        "tier": tier,
        "res_avg": round(res_avg, 3),
        "cls_avg": round(cls_avg, 3),
        "ats_total": round(ats_total, 2),
        "ats_edge": round(ats_edge, 2),
        "pred_total": round(pred_total, 2),
    }
