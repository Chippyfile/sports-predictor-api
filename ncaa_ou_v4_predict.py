"""
NCAA O/U v4 — Serve-side prediction helper
============================================
Add this to ncaa_full_predict.py after the existing O/U prediction code.

The bundle contains:
  - res_scaler, cls_scaler, ats_scaler (StandardScaler)
  - res_models (3), cls_models (2), ats_home_models, ats_away_models
  - res_feature_cols, cls_feature_cols, ats_feature_cols
  - under_tiers, over_tiers (threshold dicts)
"""


def predict_ou_v4(features_dict, market_ou_total, ou_bundle):
    """
    Predict O/U using triple agreement system.
    
    Args:
        features_dict: dict of feature_name → value (from ncaa_build_features)
        market_ou_total: float, the market O/U line
        ou_bundle: loaded model bundle from Supabase
    
    Returns:
        dict with keys:
            direction: "UNDER" | "OVER" | None
            tier: 1-3 (units to bet) or 0
            res_avg: float (residual model average prediction)
            cls_avg: float (classifier average P(under))
            ats_total: float (ATS-implied total)
            ats_edge: float (ats_total - market_total)
            confidence: str description
    """
    import numpy as np

    if market_ou_total is None or market_ou_total < 50:
        return {"direction": None, "tier": 0, "res_avg": 0, "cls_avg": 0.5,
                "ats_total": 0, "ats_edge": 0, "confidence": "no market total"}

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

    # Predict residual (3 models averaged)
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

    # Determine direction and tier
    under_tiers = ou_bundle.get("under_tiers", {})
    over_tiers = ou_bundle.get("over_tiers", {})

    # Check UNDER tiers (highest first)
    direction = None
    tier = 0
    for t in sorted(under_tiers.keys(), reverse=True):
        thresholds = under_tiers[t]
        if (res_avg <= thresholds["res_avg"] and
            cls_avg >= thresholds["cls_avg"] and
            ats_edge <= thresholds["ats_edge"]):
            direction = "UNDER"
            tier = t
            break

    # If no UNDER, check OVER tiers (highest first)
    if direction is None:
        for t in sorted(over_tiers.keys(), reverse=True):
            thresholds = over_tiers[t]
            if (res_avg >= thresholds["res_avg"] and
                cls_avg <= thresholds["cls_avg"] and
                ats_edge >= thresholds["ats_edge"]):
                direction = "OVER"
                tier = t
                break

    # Confidence description
    if tier == 3:
        confidence = "strong"
    elif tier == 2:
        confidence = "medium"
    elif tier == 1:
        confidence = "lean"
    else:
        confidence = "no signal"

    return {
        "direction": direction,
        "tier": tier,
        "res_avg": round(res_avg, 2),
        "cls_avg": round(cls_avg, 4),
        "ats_total": round(ats_total, 1),
        "ats_edge": round(ats_edge, 1),
        "confidence": confidence,
    }
