from flask import request, jsonify
from sports.mlb import mlb_build_features,  backfill_heuristic
from sports.nba import nba_build_features
from sports.ncaa import ncaa_build_features
import numpy as np, pandas as pd, traceback, shap
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.optimize import minimize_scalar
from db import sb_get, load_model


# ─────────────────────────────────────────────────────────────
# HELPER: ATS + O/U hit rates for any graded DataFrame subset
# ─────────────────────────────────────────────────────────────
def _ats_ou_stats(subset):
    """Compute ATS and O/U hit rates for a graded DataFrame subset."""
    stats = {}
    if "rl_correct" in subset.columns:
        ats_sub = subset[subset["rl_correct"].notna()]
        stats["ats_n"] = len(ats_sub)
        stats["ats_accuracy"] = round(float(ats_sub["rl_correct"].astype(bool).mean()), 4) if len(ats_sub) > 0 else None
    else:
        stats["ats_n"] = 0
        stats["ats_accuracy"] = None
    if "ou_correct" in subset.columns:
        has_line = subset["market_ou_total"].notna() if "market_ou_total" in subset.columns else pd.Series(False, index=subset.index)
        ou_sub = subset[has_line & (subset["ou_correct"] != "PUSH")]
        if len(ou_sub) > 0:
            stats["ou_n"] = len(ou_sub)
            stats["ou_accuracy"] = round(float(ou_sub["ou_correct"].isin(["OVER", "UNDER"]).mean()), 4)
        else:
            stats["ou_n"] = 0
            stats["ou_accuracy"] = None
    else:
        stats["ou_n"] = 0
        stats["ou_accuracy"] = None
    return stats
from ml_utils import StackedRegressor, StackedClassifier

def route_model_info(sport):
    bundle = load_model(sport.lower())
    if not bundle:
        return jsonify({"error": f"{sport} model not trained yet"})
    info = {
        "sport":      sport.upper(),
        "n_train":    bundle.get("n_train"),
        "mae_cv":     bundle.get("mae_cv"),
        "trained_at": bundle.get("trained_at"),
        "features":   bundle.get("feature_cols"),
        "alpha":      bundle.get("alpha"),
    }
    if sport.lower() == "mlb":
        info["dispersion"] = load_model("mlb_dispersion")
    return jsonify(info)


# ═══════════════════════════════════════════════════════════════
# NBA CONFIDENCE & CALIBRATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def nba_confidence_calibration():
    """NBA confidence/calibration analysis — tiers based on win probability margin."""
    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null&ml_correct=not.is.null"
                  "&select=*&order=game_date.asc")
    if not rows or len(rows) < 50:
        return jsonify({"error": "Need 50+ graded NBA games", "n": len(rows) if rows else 0})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["ml_correct"] = df["ml_correct"].astype(bool)
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()

    # ── Margin-based tiers (what actually predicts accuracy) ──
    # Sport-specific thresholds calibrated from cumulative accuracy curves.
    # NBA: tighter spreads, more parity → lower thresholds than NCAA.
    NBA_HIGH = 0.15   # ≥65% win prob → HIGH
    NBA_MED  = 0.05   # ≥55% win prob → MEDIUM
    tier_results = {}
    for tier, lo, hi in [("LOW", 0, NBA_MED), ("MEDIUM", NBA_MED, NBA_HIGH), ("HIGH", NBA_HIGH, 0.5)]:
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)] if tier != "HIGH" else df[df["conf_margin"] >= lo]
        if len(subset) > 0:
            ats_ou = _ats_ou_stats(subset)
            tier_results[tier] = {
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "avg_win_pct_margin": round(float(subset["conf_margin"].mean()), 4),
                "margin_range": f"{lo:.2f}-{hi:.2f}" if tier != "HIGH" else f">={lo:.2f}",
                **ats_ou,
            }

    # ── Data quality breakdown (secondary info) ──
    dq_results = {}
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].fillna("MEDIUM")
        for dq in ["LOW", "MEDIUM", "HIGH"]:
            subset = df[df["confidence"] == dq]
            if len(subset) > 0:
                ats_ou = _ats_ou_stats(subset)
                dq_results[dq] = {"n_games": len(subset),
                                  "accuracy": round(float(subset["ml_correct"].mean()), 4),
                                  **ats_ou}

    # Accuracy by win probability margin decile
    decile_results = []
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) >= 5:
            decile_results.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "expected_accuracy": round(0.5 + (lo + hi) / 2, 4),
            })

    # Cumulative: "If I only bet on games with margin >= X, what accuracy?"
    cumulative = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        subset = df[df["conf_margin"] >= threshold]
        if len(subset) >= 5:
            cumulative.append({
                "min_margin": threshold,
                "n_games": len(subset),
                "accuracy": round(float(subset["ml_correct"].mean()), 4),
                "pct_of_total": round(len(subset) / len(df), 4),
            })

    # ATS cumulative threshold
    ats_cumulative = []
    if "rl_correct" in df.columns:
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            subset = df[df["conf_margin"] >= threshold]
            ats_valid = subset[subset["rl_correct"].notna()]
            if len(ats_valid) >= 5:
                ats_cumulative.append({
                    "min_margin": threshold,
                    "n_games": len(ats_valid),
                    "ats_accuracy": round(float(ats_valid["rl_correct"].astype(bool).mean()), 4),
                    "pct_of_total": round(len(ats_valid) / max(1, df["rl_correct"].notna().sum()), 4),
                })

    # O/U cumulative threshold
    ou_cumulative = []
    if "ou_correct" in df.columns and "market_ou_total" in df.columns:
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            subset = df[df["conf_margin"] >= threshold]
            has_line = subset["market_ou_total"].notna() & (subset["ou_correct"] != "PUSH")
            ou_sub = subset[has_line]
            if len(ou_sub) >= 5:
                ou_cumulative.append({
                    "min_margin": threshold,
                    "n_games": len(ou_sub),
                    "ou_accuracy": round(float(ou_sub["ou_correct"].isin(["OVER", "UNDER"]).mean()), 4),
                    "pct_of_total": round(len(ou_sub) / max(1, (df["market_ou_total"].notna() & (df["ou_correct"] != "PUSH")).sum()), 4),
                })

    # Brier score
    brier_overall = round(float(np.mean(
        (df["win_pct_home"].clip(0.5, 0.97) - df["ml_correct"].astype(float)) ** 2
    )), 4)

    # Suggested thresholds
    suggested_medium = suggested_high = None
    for row in cumulative:
        if row["accuracy"] >= 0.55 and suggested_medium is None:
            suggested_medium = row["min_margin"]
        if row["accuracy"] >= 0.65 and suggested_high is None:
            suggested_high = row["min_margin"]

    return jsonify({
        "total_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": brier_overall,
        "by_tier": tier_results,
        "data_quality_tiers": dq_results,
        "tier_thresholds": {"LOW": f"<{NBA_MED}", "MEDIUM": f"{NBA_MED}-{NBA_HIGH}", "HIGH": f">={NBA_HIGH}"},
        "by_margin_decile": decile_results,
        "cumulative_threshold": cumulative,
        "ats_cumulative_threshold": ats_cumulative,
        "ou_cumulative_threshold": ou_cumulative,
        "suggested_thresholds": {
            "MEDIUM_min_margin": suggested_medium,
            "HIGH_min_margin": suggested_high,
        },
    })


def nba_calibration_diagnostic():
    """NBA calibration deep dive — isotonic mapping and refit potential."""
    rows = sb_get("nba_predictions",
                  "result_entered=eq.true&actual_home_score=not.is.null"
                  "&select=win_pct_home,ml_correct,confidence,actual_home_score,actual_away_score")
    if not rows or len(rows) < 50:
        return jsonify({"error": "Need 50+ graded NBA games", "n": len(rows) if rows else 0})

    df = pd.DataFrame(rows)
    df["win_pct_home"] = pd.to_numeric(df["win_pct_home"], errors="coerce").fillna(0.5)
    df["actual_home_score"] = pd.to_numeric(df["actual_home_score"], errors="coerce")
    df["actual_away_score"] = pd.to_numeric(df["actual_away_score"], errors="coerce")
    df["home_win"] = (df["actual_home_score"] > df["actual_away_score"]).astype(int)
    df["conf_margin"] = (df["win_pct_home"] - 0.5).abs()
    df["ml_correct"] = df["ml_correct"].astype(bool)

    # Calibration by confidence margin decile
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    cal_deciles = []
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        subset = df[(df["conf_margin"] >= lo) & (df["conf_margin"] < hi)]
        if len(subset) >= 5:
            actual_acc = float(subset["ml_correct"].mean())
            expected_acc = 0.5 + (lo + hi) / 2
            gap = actual_acc - expected_acc
            cal_deciles.append({
                "margin_range": f"{lo:.2f}-{hi:.2f}",
                "n_games": len(subset),
                "actual_accuracy": round(actual_acc, 4),
                "expected_accuracy": round(expected_acc, 4),
                "gap": round(gap, 4),
                "miscalibrated": abs(gap) > 0.10,
            })

    # Current isotonic mapping
    bundle = load_model("nba")
    iso_info = {"status": "not_loaded"}
    if bundle:
        isotonic = bundle.get("isotonic")
        if isotonic is not None:
            try:
                test_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
                mapped = isotonic.predict(test_probs)
                iso_info = {
                    "status": "fitted",
                    "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
                }
            except Exception:
                iso_info = {"status": "fitted_but_error"}
        else:
            iso_info = {"status": "null"}

    # Refit potential
    refit_info = {}
    if len(df) >= 50:
        from sklearn.isotonic import IsotonicRegression as IR
        refit_iso = IR(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        refit_iso.fit(df["win_pct_home"].values, df["home_win"].values)
        test_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        mapped = refit_iso.predict(test_probs)
        refit_info = {
            "mapping": {f"{p:.2f}": round(float(m), 4) for p, m in zip(test_probs, mapped)},
            "n_samples": len(df),
        }

    # Brier
    brier = round(float(np.mean(
        (df["win_pct_home"].clip(0.5, 0.97) - df["ml_correct"].astype(float)) ** 2
    )), 4)

    # Recommendations
    recs = []
    for d in cal_deciles:
        if d["miscalibrated"]:
            direction = "OVERCONFIDENT" if d["gap"] < 0 else "UNDERCONFIDENT"
            recs.append(
                f"Margin {d['margin_range']}: Model is {direction} "
                f"(actual {d['actual_accuracy']:.1%} vs expected {d['expected_accuracy']:.1%})."
            )

    return jsonify({
        "n_graded_games": len(df),
        "overall_accuracy": round(float(df["ml_correct"].mean()), 4),
        "brier_score": brier,
        "calibration_by_decile": cal_deciles,
        "current_isotonic": iso_info,
        "refit_potential": refit_info,
        "recommendations": recs,
    })


# ═══════════════════════════════════════════════════════════════
# NBA WALK-FORWARD BACKTEST
# ═══════════════════════════════════════════════════════════════

def route_backtest_nba():
    """
    Walk-forward backtest for NBA predictions.
    Uses stacking ensemble (GBM+RF+Ridge) with isotonic calibration.
    Trains on all months BEFORE the test month, predicts the test month.
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        min_train = int(body.get("min_train", 200))

        rows = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*&order=game_date.asc")
        if not rows or len(rows) < min_train + 50:
            return jsonify({"error": f"Need {min_train + 50}+ graded games, have {len(rows) if rows else 0}"})

        df = pd.DataFrame(rows)
        for col in ["actual_home_score", "actual_away_score", "pred_home_score", "pred_away_score",
                     "win_pct_home", "spread_home", "ou_total",
                     "market_spread_home", "market_ou_total", "model_ml_home",
                     "home_net_rtg", "away_net_rtg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["game_date"] = pd.to_datetime(df["game_date"])
        df["month"] = df["game_date"].dt.to_period("M")
        y_margin = (df["actual_home_score"] - df["actual_away_score"]).to_numpy().astype(float)
        y_win = (y_margin > 0).astype(int)

        months = sorted(df["month"].unique())
        results_by_month = []
        all_predictions = []

        for i, test_month in enumerate(months):
            train_mask = df["month"] < test_month
            test_mask = df["month"] == test_month
            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < min_train or len(test_df) < 5:
                continue

            X_train = nba_build_features(train_df)
            X_test = nba_build_features(test_df)
            y_train_margin = y_margin[train_mask.to_numpy()]
            y_test_margin = y_margin[test_mask.to_numpy()]
            y_train_win = y_win[train_mask.to_numpy()]
            y_test_win = y_win[test_mask.to_numpy()]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            cv_folds = min(3, len(train_df))

            if len(train_df) >= 200:
                # Stacking ensemble (matches train_nba architecture)
                gbm = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_r = RandomForestRegressor(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

                oof_g = cross_val_predict(gbm, X_tr, y_train_margin, cv=cv_folds)
                oof_r = cross_val_predict(rf_r, X_tr, y_train_margin, cv=cv_folds)
                oof_ridge = cross_val_predict(ridge, X_tr, y_train_margin, cv=cv_folds)

                gbm.fit(X_tr, y_train_margin)
                rf_r.fit(X_tr, y_train_margin)
                ridge.fit(X_tr, y_train_margin)

                meta_X = np.column_stack([oof_g, oof_r, oof_ridge])
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                meta.fit(meta_X, y_train_margin)

                # Bias correction
                oof_meta = meta.predict(meta_X)
                bias_correction = float(np.mean(oof_meta - y_train_margin))

                test_meta_X = np.column_stack([
                    gbm.predict(X_te), rf_r.predict(X_te), ridge.predict(X_te)
                ])
                pred_margin = meta.predict(test_meta_X) - bias_correction

                # Stacked classifier
                gbm_c = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_c = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                lr_c = LogisticRegression(max_iter=1000, C=1.0)

                oof_gc = cross_val_predict(gbm_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_rc = cross_val_predict(rf_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_lc = cross_val_predict(lr_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]

                gbm_c.fit(X_tr, y_train_win)
                rf_c.fit(X_tr, y_train_win)
                lr_c.fit(X_tr, y_train_win)

                meta_clf = LogisticRegression(max_iter=1000, C=1.0)
                oof_clf_X = np.column_stack([oof_gc, oof_rc, oof_lc])
                meta_clf.fit(oof_clf_X, y_train_win)

                # Isotonic calibration on OOF stacked probs
                oof_stacked_p = meta_clf.predict_proba(oof_clf_X)[:, 1]
                iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
                iso.fit(oof_stacked_p, y_train_win)

                test_clf_X = np.column_stack([
                    gbm_c.predict_proba(X_te)[:, 1],
                    rf_c.predict_proba(X_te)[:, 1],
                    lr_c.predict_proba(X_te)[:, 1],
                ])
                raw_wp = meta_clf.predict_proba(test_clf_X)[:, 1]
                pred_wp = iso.predict(raw_wp)
            else:
                reg = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.08, subsample=0.8,
                    min_samples_leaf=15, random_state=42,
                )
                reg.fit(X_tr, y_train_margin)
                pred_margin = reg.predict(X_te)

                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.08, subsample=0.8,
                    min_samples_leaf=15, random_state=42,
                )
                clf.fit(X_tr, y_train_win)
                pred_wp = clf.predict_proba(X_te)[:, 1]

            pred_pick = (pred_wp >= 0.5).astype(int)

            accuracy = float(np.mean(pred_pick == y_test_win))
            mae_margin = float(mean_absolute_error(y_test_margin, pred_margin))
            brier = float(brier_score_loss(y_test_win, pred_wp))

            # Heuristic baseline
            heur_wp = test_df["win_pct_home"].fillna(0.5).values
            heur_pick = (heur_wp >= 0.5).astype(int)
            heur_acc = float(np.mean(heur_pick == y_test_win))
            heur_brier = float(brier_score_loss(y_test_win, heur_wp))

            results_by_month.append({
                "month": str(test_month),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "ml_accuracy": round(accuracy, 4),
                "ml_brier": round(brier, 4),
                "ml_mae_margin": round(mae_margin, 3),
                "heuristic_accuracy": round(heur_acc, 4),
                "heuristic_brier": round(heur_brier, 4),
                "home_win_rate": round(float(y_test_win.mean()), 3),
            })

            for j in range(len(test_df)):
                all_predictions.append({
                    "pred_win_prob": round(float(pred_wp[j]), 4),
                    "heur_win_prob": round(float(heur_wp[j]), 4),
                    "pred_margin": round(float(pred_margin[j]), 2),
                    "actual_margin": int(y_test_margin[j]),
                    "actual_home_win": int(y_test_win[j]),
                    "ml_correct": int(pred_pick[j] == y_test_win[j]),
                    "heur_correct": int(heur_pick[j] == y_test_win[j]),
                })

        if not results_by_month:
            return jsonify({"error": f"No months with >= {min_train} training games"})

        total = sum(r["n_test"] for r in results_by_month)
        agg = {
            "total_games_tested": total,
            "months_tested": len(results_by_month),
            "ml_overall_accuracy": round(sum(r["ml_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_brier": round(sum(r["ml_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_mae_margin": round(sum(r["ml_mae_margin"] * r["n_test"] for r in results_by_month) / total, 3),
            "heur_overall_accuracy": round(sum(r["heuristic_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "heur_overall_brier": round(sum(r["heuristic_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "baseline_home_always": round(sum(r["home_win_rate"] * r["n_test"] for r in results_by_month) / total, 4),
        }

        # Confidence tier analysis on walk-forward predictions
        conf_results = []
        all_preds_arr = np.array([(p["pred_win_prob"], p["actual_home_win"]) for p in all_predictions])
        if len(all_preds_arr) > 0:
            for t in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
                strong = (all_preds_arr[:, 0] >= t) | (all_preds_arr[:, 0] <= (1 - t))
                ns = int(strong.sum())
                if ns > 0:
                    pred_side = (all_preds_arr[strong, 0] >= 0.5).astype(int)
                    actual = all_preds_arr[strong, 1].astype(int)
                    conf_results.append({
                        "min_confidence": f"{t:.0%}",
                        "n_games": ns,
                        "accuracy": round(float(np.mean(pred_side == actual)), 4),
                    })

        return jsonify({
            "status": "backtest_complete",
            "model": "StackedEnsemble(GBM+RF+Ridge) with isotonic calibration",
            "aggregate": agg,
            "by_month": results_by_month,
            "confidence_tiers": conf_results,
            "n_predictions": len(all_predictions),
            "sample_predictions": all_predictions[:20],
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# CLV (CLOSING LINE VALUE) DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def clv_diagnostic():
    """
    Check CLV tracking pipeline health across all sports.
    Reports how many games have opening/closing lines captured,
    and basic CLV statistics for validated predictions.
    """
    results = {}
    for sport, (table, label) in SPORT_TABLES.items():
        try:
            # First query with only guaranteed columns to get graded games
            rows = sb_get(table,
                          "result_entered=eq.true&select=*&limit=5000")
            if not rows:
                results[sport] = {"status": "no_graded_games", "n": 0}
                continue

            df = pd.DataFrame(rows)
            n_total = len(df)

            # Check which CLV-relevant columns actually exist in the data
            clv_cols = {}
            for col in ["market_spread_home", "opening_spread_home", "closing_spread_home",
                        "market_ou_total", "opening_ou_total", "closing_ou_total",
                        "spread_home", "ou_total", "win_pct_home", "ml_correct"]:
                if col in df.columns:
                    non_null = pd.to_numeric(df[col], errors="coerce").dropna()
                    clv_cols[col] = int(len(non_null))
                else:
                    clv_cols[col] = 0

            # Basic ML accuracy from available data
            ml_acc = None
            if "ml_correct" in df.columns:
                ml_vals = df["ml_correct"].dropna()
                if len(ml_vals) > 0:
                    ml_acc = round(float(ml_vals.astype(float).mean()), 4)

            # Compute spread CLV if we have both model and market spreads
            spread_clv = None
            if clv_cols.get("spread_home", 0) > 0 and clv_cols.get("market_spread_home", 0) > 0:
                model_spread = pd.to_numeric(df["spread_home"], errors="coerce")
                market_spread = pd.to_numeric(df["market_spread_home"], errors="coerce")
                valid = model_spread.notna() & market_spread.notna()
                if valid.sum() >= 10:
                    # Spread CLV = how far the model's spread differs from market
                    # If model says -5.0 and market says -3.0, model is getting +2 pts
                    diff = (model_spread[valid] - market_spread[valid])
                    actual_margin = None
                    if "actual_home_score" in df.columns and "actual_away_score" in df.columns:
                        actual_margin = (
                            pd.to_numeric(df.loc[valid, "actual_home_score"], errors="coerce") -
                            pd.to_numeric(df.loc[valid, "actual_away_score"], errors="coerce")
                        )
                    spread_clv = {
                        "n_games_with_both": int(valid.sum()),
                        "avg_model_spread": round(float(model_spread[valid].mean()), 2),
                        "avg_market_spread": round(float(market_spread[valid].mean()), 2),
                        "avg_diff": round(float(diff.mean()), 3),
                        "model_closer_to_result_pct": None,
                    }
                    if actual_margin is not None:
                        model_err = (model_spread[valid] - actual_margin).abs()
                        market_err = (market_spread[valid] - actual_margin).abs()
                        both_valid = model_err.notna() & market_err.notna()
                        if both_valid.sum() >= 10:
                            model_closer = (model_err[both_valid] < market_err[both_valid]).mean()
                            spread_clv["model_closer_to_result_pct"] = round(float(model_closer), 4)
                            spread_clv["model_mae"] = round(float(model_err[both_valid].mean()), 2)
                            spread_clv["market_mae"] = round(float(market_err[both_valid].mean()), 2)

            # O/U analysis
            ou_comparison = None
            if clv_cols.get("ou_total", 0) > 0 and clv_cols.get("market_ou_total", 0) > 0:
                model_ou = pd.to_numeric(df["ou_total"], errors="coerce")
                market_ou = pd.to_numeric(df["market_ou_total"], errors="coerce")
                valid = model_ou.notna() & market_ou.notna()
                if valid.sum() >= 10:
                    actual_total = None
                    # Try to compute actual total
                    for h_col, a_col in [("actual_home_score", "actual_away_score"),
                                         ("actual_home_runs", "actual_away_runs")]:
                        if h_col in df.columns and a_col in df.columns:
                            actual_total = (
                                pd.to_numeric(df.loc[valid, h_col], errors="coerce") +
                                pd.to_numeric(df.loc[valid, a_col], errors="coerce")
                            )
                            break
                    ou_comparison = {
                        "n_games": int(valid.sum()),
                        "avg_model_total": round(float(model_ou[valid].mean()), 2),
                        "avg_market_total": round(float(market_ou[valid].mean()), 2),
                    }
                    if actual_total is not None:
                        model_ou_err = (model_ou[valid] - actual_total).abs()
                        market_ou_err = (market_ou[valid] - actual_total).abs()
                        both_valid = model_ou_err.notna() & market_ou_err.notna()
                        if both_valid.sum() >= 10:
                            ou_comparison["model_mae"] = round(float(model_ou_err[both_valid].mean()), 2)
                            ou_comparison["market_mae"] = round(float(market_ou_err[both_valid].mean()), 2)
                            ou_comparison["model_closer_pct"] = round(
                                float((model_ou_err[both_valid] < market_ou_err[both_valid]).mean()), 4
                            )

            # Pipeline health determination
            has_closing = clv_cols.get("closing_spread_home", 0) > 0
            has_market = clv_cols.get("market_spread_home", 0) > 0
            has_model = clv_cols.get("spread_home", 0) > 0
            health = ("ACTIVE" if has_closing else
                      "WIRED" if has_market and has_model else
                      "PARTIAL" if has_market or has_model else
                      "NOT_WIRED")

            results[sport] = {
                "n_graded_games": n_total,
                "ml_accuracy": ml_acc,
                "column_coverage": {k: v for k, v in clv_cols.items() if k not in ("ml_correct",)},
                "spread_analysis": spread_clv,
                "ou_analysis": ou_comparison,
                "pipeline_health": health,
            }
        except Exception as e:
            import traceback as _tb
            results[sport] = {"error": str(e), "traceback": _tb.format_exc()}

    return jsonify(results)


def spread_accuracy_analysis():
    """
    Detailed spread accuracy analysis: model vs market for each sport.
    Shows how often the model's spread is closer to the actual result.
    """
    results = {}
    for sport, (table, label) in SPORT_TABLES.items():
        try:
            rows = sb_get(table, "result_entered=eq.true&select=*&limit=5000")
            if not rows:
                results[sport] = {"n": 0, "status": "no_graded_games"}
                continue

            df = pd.DataFrame(rows)
            n_total = len(df)

            # Determine score columns
            h_score = a_score = None
            for h, a in [("actual_home_score", "actual_away_score"),
                         ("actual_home_runs", "actual_away_runs")]:
                if h in df.columns and a in df.columns:
                    h_score, a_score = h, a
                    break
            if not h_score:
                results[sport] = {"n": n_total, "error": "no_score_columns"}
                continue

            df["actual_margin"] = pd.to_numeric(df[h_score], errors="coerce") - pd.to_numeric(df[a_score], errors="coerce")
            df["actual_total"] = pd.to_numeric(df[h_score], errors="coerce") + pd.to_numeric(df[a_score], errors="coerce")

            model_spread_col = "spread_home" if "spread_home" in df.columns else None
            model_total_col = "ou_total" if "ou_total" in df.columns else None
            market_spread_col = "market_spread_home" if "market_spread_home" in df.columns else None
            market_total_col = "market_ou_total" if "market_ou_total" in df.columns else None

            analysis = {"n_graded": n_total, "label": label}

            # Model spread accuracy
            if model_spread_col:
                ms = pd.to_numeric(df[model_spread_col], errors="coerce")
                valid = ms.notna() & df["actual_margin"].notna()
                if valid.sum() >= 10:
                    errs = (ms[valid] - df.loc[valid, "actual_margin"]).abs()
                    analysis["model_spread"] = {
                        "n": int(valid.sum()),
                        "mae": round(float(errs.mean()), 2),
                        "correct_side_pct": round(float(
                            ((ms[valid] > 0) == (df.loc[valid, "actual_margin"] > 0)).mean()
                        ), 4),
                    }

            # Market spread accuracy
            if market_spread_col:
                mkt = pd.to_numeric(df[market_spread_col], errors="coerce")
                valid = mkt.notna() & df["actual_margin"].notna()
                if valid.sum() >= 10:
                    errs = (mkt[valid] - df.loc[valid, "actual_margin"]).abs()
                    analysis["market_spread"] = {
                        "n": int(valid.sum()),
                        "mae": round(float(errs.mean()), 2),
                        "correct_side_pct": round(float(
                            ((mkt[valid] > 0) == (df.loc[valid, "actual_margin"] > 0)).mean()
                        ), 4),
                    }

            # Head-to-head spread
            if model_spread_col and market_spread_col:
                ms = pd.to_numeric(df[model_spread_col], errors="coerce")
                mkt = pd.to_numeric(df[market_spread_col], errors="coerce")
                valid = ms.notna() & mkt.notna() & df["actual_margin"].notna()
                if valid.sum() >= 5:
                    me = (ms[valid] - df.loc[valid, "actual_margin"]).abs()
                    mke = (mkt[valid] - df.loc[valid, "actual_margin"]).abs()
                    analysis["h2h_spread"] = {
                        "n": int(valid.sum()),
                        "model_closer_pct": round(float((me < mke).mean()), 4),
                        "avg_model_advantage": round(float((mke - me).mean()), 2),
                    }

            # Model total accuracy + bias
            if model_total_col:
                mt = pd.to_numeric(df[model_total_col], errors="coerce")
                valid = mt.notna() & df["actual_total"].notna()
                if valid.sum() >= 10:
                    errs = (mt[valid] - df.loc[valid, "actual_total"]).abs()
                    bias = float((mt[valid] - df.loc[valid, "actual_total"]).mean())
                    analysis["model_total"] = {
                        "n": int(valid.sum()),
                        "mae": round(float(errs.mean()), 2),
                        "bias": round(bias, 2),
                        "bias_direction": "HIGH" if bias > 1 else "LOW" if bias < -1 else "OK",
                        "avg_model": round(float(mt[valid].mean()), 2),
                        "avg_actual": round(float(df.loc[valid, "actual_total"].mean()), 2),
                        "model_p10": round(float(mt[valid].quantile(0.1)), 1),
                        "model_p50": round(float(mt[valid].quantile(0.5)), 1),
                        "model_p90": round(float(mt[valid].quantile(0.9)), 1),
                    }

            # Head-to-head totals
            if model_total_col and market_total_col:
                mt = pd.to_numeric(df[model_total_col], errors="coerce")
                mkt_t = pd.to_numeric(df[market_total_col], errors="coerce")
                valid = mt.notna() & mkt_t.notna() & df["actual_total"].notna()
                if valid.sum() >= 5:
                    me = (mt[valid] - df.loc[valid, "actual_total"]).abs()
                    mke = (mkt_t[valid] - df.loc[valid, "actual_total"]).abs()
                    analysis["h2h_total"] = {
                        "n": int(valid.sum()),
                        "model_closer_pct": round(float((me < mke).mean()), 4),
                        "avg_model": round(float(mt[valid].mean()), 2),
                        "avg_market": round(float(mkt_t[valid].mean()), 2),
                        "avg_actual": round(float(df.loc[valid, "actual_total"].mean()), 2),
                        "model_mae": round(float(me.mean()), 2),
                        "market_mae": round(float(mke.mean()), 2),
                    }

            results[sport] = analysis
        except Exception as e:
            import traceback as _tb
            results[sport] = {"error": str(e)}

    return jsonify(results)

# ══════════════════════════════════════════════════════════════════
# PYTHON-SIDE HEURISTIC REPLAY (mirrors mlb.js logic)
# Uses columns that ACTUALLY EXIST in mlb_historical:
#   home_woba, away_woba, home_fip, away_fip, home_k9, away_k9,
#   home_bb9, away_bb9, park_factor, home_rest_days, away_rest_days,
#   home_travel, away_travel
# Does NOT use: home_sp_fip (null), home_bullpen_era (missing), temp_f (null)
# SEASON_CONSTANTS, DEFAULT_CONSTANTS, FIP_COEFF, HFA_RUNS defined above
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════

def route_backtest_mlb():
    """
    Walk-forward backtest with heuristic backfill.
    Body params (all optional):
      - test_seasons: list (default: [2019,2021,2022,2023,2024,2025])
      - use_heuristic: bool (default: true)
      - min_train_seasons: int (default: 3)
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        test_seasons = body.get("test_seasons", [2019, 2021, 2022, 2023, 2024, 2025])
        use_heuristic = body.get("use_heuristic", True)
        min_train = body.get("min_train_seasons", 3)

        all_rows = sb_get(
            "mlb_historical",
            "is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*&order=season.asc&limit=100000"
        )
        if not all_rows or len(all_rows) < 100:
            return jsonify({"error": "Not enough historical data"})

        all_df = pd.DataFrame(all_rows)
        for col in ["actual_home_runs", "actual_away_runs", "home_win",
                     "home_woba", "away_woba", "home_sp_fip", "away_sp_fip",
                     "home_fip", "away_fip", "home_k9", "away_k9",
                     "home_bb9", "away_bb9",
                     "park_factor", "temp_f", "wind_mph", "wind_out_flag",
                     "home_rest_days", "away_rest_days", "home_travel", "away_travel",
                     "season_weight", "season",
                     "home_sp_fip_known", "away_sp_fip_known"]:
            if col in all_df.columns:
                all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

        if use_heuristic:
            print("  Backfilling heuristic predictions...")
            all_df = backfill_heuristic(all_df)
            # Verify differentiation
            wp_std = all_df["win_pct_home"].std()
            wp_min = all_df["win_pct_home"].min()
            wp_max = all_df["win_pct_home"].max()
            print(f"  Heuristic win_pct: min={wp_min:.3f} max={wp_max:.3f} std={wp_std:.3f}")
        else:
            all_df["pred_home_runs"] = 0.0
            all_df["pred_away_runs"] = 0.0
            all_df["win_pct_home"]   = 0.5
            all_df["ou_total"]       = 0.0
            all_df["model_ml_home"]  = 0

        available_seasons = sorted(all_df["season"].dropna().astype(int).unique().tolist())
        results_by_season = []
        all_predictions = []

        for test_season in test_seasons:
            if test_season not in available_seasons:
                continue

            train_df = all_df[all_df["season"] < test_season].copy()
            test_df  = all_df[all_df["season"] == test_season].copy()

            train_seasons = sorted(train_df["season"].dropna().astype(int).unique().tolist())
            if len(train_seasons) < min_train or len(test_df) < 10:
                continue

            X_train = mlb_build_features(train_df)
            y_train_margin = (train_df["actual_home_runs"] - train_df["actual_away_runs"]).values
            y_train_win = train_df["home_win"].astype(int).values

            X_test = mlb_build_features(test_df)
            y_test_margin = (test_df["actual_home_runs"] - test_df["actual_away_runs"]).values
            y_test_win = test_df["home_win"].astype(int).values
            y_test_hr = test_df["actual_home_runs"].values
            y_test_ar = test_df["actual_away_runs"].values

            weights = train_df["season_weight"].fillna(1.0).astype(float).values if "season_weight" in train_df.columns else np.ones(len(train_df))

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            # Lighter models for Railway backtest speed (proxy timeout ~120s)
            gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, min_samples_leaf=25, random_state=42)
            rf_reg = RandomForestRegressor(n_estimators=60, max_depth=5, min_samples_leaf=20, max_features=0.7, random_state=42, n_jobs=1)
            ridge = RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0], cv=3)

            gbm.fit(X_train_s, y_train_margin, sample_weight=weights)
            rf_reg.fit(X_train_s, y_train_margin, sample_weight=weights)
            ridge.fit(X_train_s, y_train_margin, sample_weight=weights)

            meta_X = np.column_stack([gbm.predict(X_train_s), rf_reg.predict(X_train_s), ridge.predict(X_train_s)])
            meta_reg = Ridge(alpha=1.0)
            meta_reg.fit(meta_X, y_train_margin)

            # Lighter classifier: GBM + LR only (RF classifier adds time, minimal backtest accuracy delta)
            gbm_clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8, min_samples_leaf=25, random_state=42)
            lr_clf = LogisticRegression(max_iter=1000)
            gbm_clf.fit(X_train_s, y_train_win, sample_weight=weights)
            lr_clf.fit(X_train_s, y_train_win, sample_weight=weights)

            test_meta = np.column_stack([gbm.predict(X_test_s), rf_reg.predict(X_test_s), ridge.predict(X_test_s)])
            pred_margin = meta_reg.predict(test_meta)
            pred_wp = 0.6 * gbm_clf.predict_proba(X_test_s)[:, 1] + 0.4 * lr_clf.predict_proba(X_test_s)[:, 1]
            pred_pick = (pred_wp >= 0.5).astype(int)

            accuracy = float(np.mean(pred_pick == y_test_win))
            mae_margin = float(mean_absolute_error(y_test_margin, pred_margin))
            brier = float(brier_score_loss(y_test_win, pred_wp))

            # Run-level MAE
            lg_avg = float(y_test_hr.mean() + y_test_ar.mean()) / 2
            pred_hr = lg_avg + pred_margin / 2
            pred_ar = lg_avg - pred_margin / 2
            mae_home_runs = float(mean_absolute_error(y_test_hr, pred_hr))
            mae_away_runs = float(mean_absolute_error(y_test_ar, pred_ar))
            mae_total = float(mean_absolute_error(y_test_hr + y_test_ar, pred_hr + pred_ar))

            # Heuristic-only comparison
            heur_acc = heur_brier = heur_mae_margin = heur_mae_total = None
            if use_heuristic:
                heur_wp = test_df["win_pct_home"].values
                heur_pick = (heur_wp >= 0.5).astype(int)
                heur_acc = float(np.mean(heur_pick == y_test_win))
                heur_brier = float(brier_score_loss(y_test_win, np.clip(heur_wp, 0.01, 0.99)))
                heur_pred_hr = test_df["pred_home_runs"].values
                heur_pred_ar = test_df["pred_away_runs"].values
                heur_mae_margin = float(mean_absolute_error(y_test_margin, heur_pred_hr - heur_pred_ar))
                heur_mae_total = float(mean_absolute_error(y_test_hr + y_test_ar, heur_pred_hr + heur_pred_ar))

            cal_bins = []
            for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
                mask = (pred_wp >= lo) & (pred_wp < hi)
                n_bin = int(mask.sum())
                if n_bin > 0:
                    cal_bins.append({"range": f"{lo:.0%}-{hi:.0%}", "n": n_bin,
                        "predicted": round(float(pred_wp[mask].mean()), 3),
                        "actual": round(float(y_test_win[mask].mean()), 3)})

            conf_results = []
            for t in [0.52, 0.55, 0.58, 0.60, 0.65]:
                strong = (pred_wp >= t) | (pred_wp <= (1 - t))
                ns = int(strong.sum())
                if ns > 0:
                    conf_results.append({"min_confidence": f"{t:.0%}", "n_games": ns,
                        "accuracy": round(float(np.mean(pred_pick[strong] == y_test_win[strong])), 4)})

            results_by_season.append({
                "season": int(test_season), "n_train": len(train_df), "n_test": len(test_df),
                "train_seasons": train_seasons,
                "ml_accuracy": round(accuracy, 4),
                "ml_brier": round(brier, 4),
                "ml_mae_margin": round(mae_margin, 3),
                "ml_mae_home_runs": round(mae_home_runs, 3),
                "ml_mae_away_runs": round(mae_away_runs, 3),
                "ml_mae_total": round(mae_total, 3),
                "heuristic_accuracy": round(heur_acc, 4) if heur_acc is not None else None,
                "heuristic_brier": round(heur_brier, 4) if heur_brier is not None else None,
                "heuristic_mae_margin": round(heur_mae_margin, 3) if heur_mae_margin is not None else None,
                "heuristic_mae_total": round(heur_mae_total, 3) if heur_mae_total is not None else None,
                "home_win_rate": round(float(y_test_win.mean()), 3),
                "calibration": cal_bins, "confidence_tiers": conf_results,
            })

            for i in range(len(test_df)):
                all_predictions.append({
                    "season": int(test_season),
                    "home_team": str(test_df.iloc[i].get("home_team", "")),
                    "away_team": str(test_df.iloc[i].get("away_team", "")),
                    "pred_win_prob": round(float(pred_wp[i]), 4),
                    "heur_win_prob": round(float(test_df.iloc[i].get("win_pct_home", 0.5)), 4) if use_heuristic else None,
                    "heur_pred_hr": round(float(test_df.iloc[i].get("pred_home_runs", 0)), 2) if use_heuristic else None,
                    "heur_pred_ar": round(float(test_df.iloc[i].get("pred_away_runs", 0)), 2) if use_heuristic else None,
                    "pred_margin": round(float(pred_margin[i]), 2),
                    "actual_margin": int(y_test_margin[i]),
                    "actual_home_win": int(y_test_win[i]),
                    "correct": int(pred_pick[i] == y_test_win[i]),
                })

        if results_by_season:
            total = sum(r["n_test"] for r in results_by_season)
            agg = {
                "total_games_tested": total,
                "seasons_tested": len(results_by_season),
                "heuristic_backfill": use_heuristic,
                "ml_overall_accuracy": round(sum(r["ml_accuracy"] * r["n_test"] for r in results_by_season) / total, 4),
                "ml_overall_brier": round(sum(r["ml_brier"] * r["n_test"] for r in results_by_season) / total, 4),
                "ml_overall_mae_margin": round(sum(r["ml_mae_margin"] * r["n_test"] for r in results_by_season) / total, 3),
                "baseline_home_always": round(sum(r["home_win_rate"] * r["n_test"] for r in results_by_season) / total, 4),
            }
            if use_heuristic:
                agg["heur_overall_accuracy"] = round(sum((r["heuristic_accuracy"] or 0) * r["n_test"] for r in results_by_season) / total, 4)
                agg["heur_overall_brier"] = round(sum((r["heuristic_brier"] or 0) * r["n_test"] for r in results_by_season) / total, 4)
                agg["heur_overall_mae_margin"] = round(sum((r["heuristic_mae_margin"] or 0) * r["n_test"] for r in results_by_season) / total, 3)
                agg["heur_overall_mae_total"] = round(sum((r["heuristic_mae_total"] or 0) * r["n_test"] for r in results_by_season) / total, 3)
        else:
            agg = {"error": "No seasons tested"}

        return jsonify({
            "status": "backtest_complete", "aggregate": agg,
            "by_season": results_by_season, "n_predictions": len(all_predictions),
            "sample_predictions": all_predictions[:20],
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


def route_backtest_current_model():
    """Test the CURRENT production model against a season. Body: { "season": 2024, "use_heuristic": true }"""
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        test_season = int(body.get("season", 2024))
        use_heuristic = body.get("use_heuristic", True)

        bundle = load_model("mlb")
        if not bundle:
            return jsonify({"error": "MLB model not trained"})

        test_rows = sb_get("mlb_historical", f"season=eq.{test_season}&is_outlier_season=eq.0&actual_home_runs=not.is.null&select=*")
        if not test_rows or len(test_rows) < 10:
            return jsonify({"error": f"Not enough data for season {test_season}"})

        test_df = pd.DataFrame(test_rows)
        for col in ["actual_home_runs","actual_away_runs","home_win","home_woba","away_woba",
                     "home_sp_fip","away_sp_fip","home_fip","away_fip","home_k9","away_k9",
                     "home_bb9","away_bb9","park_factor","temp_f","wind_mph","wind_out_flag",
                     "home_rest_days","away_rest_days","home_travel","away_travel"]:
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        if use_heuristic:
            test_df = backfill_heuristic(test_df)
        else:
            test_df["pred_home_runs"] = 0.0
            test_df["pred_away_runs"] = 0.0
            test_df["win_pct_home"]   = 0.5
            test_df["ou_total"]       = 0.0
            test_df["model_ml_home"]  = 0

        X_test = mlb_build_features(test_df)
        y_margin = (test_df["actual_home_runs"] - test_df["actual_away_runs"]).values
        y_win = test_df["home_win"].astype(int).values

        X_s = bundle["scaler"].transform(X_test[bundle["feature_cols"]])
        pred_margin = bundle["reg"].predict(X_s)
        pred_wp = bundle["clf"].predict_proba(X_s)[:, 1]
        pred_pick = (pred_wp >= 0.5).astype(int)

        accuracy = round(float(np.mean(pred_pick == y_win)), 4)
        mae = round(float(mean_absolute_error(y_margin, pred_margin)), 3)
        brier = round(float(brier_score_loss(y_win, pred_wp)), 4)

        heur_acc = heur_mae = None
        if use_heuristic:
            heur_wp = test_df["win_pct_home"].values
            heur_pick = (heur_wp >= 0.5).astype(int)
            heur_acc = round(float(np.mean(heur_pick == y_win)), 4)
            heur_mae = round(float(mean_absolute_error(y_margin, test_df["pred_home_runs"].values - test_df["pred_away_runs"].values)), 3)

        monthly = {}
        if "game_date" in test_df.columns:
            for i in range(len(test_df)):
                m = str(test_df.iloc[i].get("game_date", ""))[:7]
                if not m: continue
                if m not in monthly: monthly[m] = {"n": 0, "correct": 0, "errs": []}
                monthly[m]["n"] += 1
                monthly[m]["correct"] += int(pred_pick[i] == y_win[i])
                monthly[m]["errs"].append(abs(float(pred_margin[i] - y_margin[i])))

        monthly_results = [{"month": m, "n": v["n"], "accuracy": round(v["correct"]/v["n"], 4),
                           "mae": round(float(np.mean(v["errs"])), 3)} for m, v in sorted(monthly.items())]

        return jsonify({
            "status": "current_model_backtest", "test_season": test_season,
            "heuristic_backfill": use_heuristic,
            "n_test": len(test_df), "model_trained_on": bundle.get("n_train", 0),
            "ml_accuracy": accuracy, "ml_brier": brier, "ml_mae_margin": mae,
            "heuristic_accuracy": heur_acc, "heuristic_mae_margin": heur_mae,
            "home_win_rate": round(float(y_win.mean()), 3),
            "monthly": monthly_results,
            "note": "In-sample test. Use /backtest/mlb for unbiased walk-forward.",
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# NCAA BACKTEST (v17: R1-R8 fixes — ElasticNet, bias correction, isotonic)
# ═══════════════════════════════════════════════════════════════

def route_backtest_ncaa():
    """
    Walk-forward backtest for NCAAB predictions.
    v17: Uses ElasticNet, bias correction, isotonic calibration.
    Body: { "min_train": 200 }
    """
    import traceback
    try:
        body = request.get_json(force=True, silent=True) or {}
        min_train = int(body.get("min_train", 200))

        rows = sb_get("ncaa_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*&order=game_date.asc")
        if not rows or len(rows) < min_train + 50:
            return jsonify({"error": f"Need {min_train + 50}+ graded games, have {len(rows) if rows else 0}"})

        df = pd.DataFrame(rows)
        for col in ["actual_home_score", "actual_away_score", "pred_home_score", "pred_away_score",
                     "home_adj_em", "away_adj_em", "win_pct_home", "spread_home",
                     "market_spread_home", "market_ou_total", "ou_total", "model_ml_home"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["game_date"] = pd.to_datetime(df["game_date"])
        df["month"] = df["game_date"].dt.to_period("M")
        y_margin = (df["actual_home_score"] - df["actual_away_score"]).to_numpy().astype(float)
        y_win = (y_margin > 0).astype(int)

        months = sorted(df["month"].unique())
        results_by_month = []
        all_predictions = []

        for i, test_month in enumerate(months):
            train_mask = df["month"] < test_month
            test_mask = df["month"] == test_month
            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < min_train or len(test_df) < 5:
                continue

            X_train = ncaa_build_features(train_df)
            X_test = ncaa_build_features(test_df)
            y_train_margin = y_margin[train_mask.to_numpy()]
            y_test_margin = y_margin[test_mask.to_numpy()]
            y_train_win = y_win[train_mask.to_numpy()]
            y_test_win = y_win[test_mask.to_numpy()]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            cv_folds = min(3, len(train_df))

            if len(train_df) >= 200:
                # R7: ElasticNet replaces Ridge
                gbm = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_r = RandomForestRegressor(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9],
                                    alphas=[0.01, 0.1, 1.0],
                                    cv=cv_folds, random_state=42)

                oof_g = cross_val_predict(gbm, X_tr, y_train_margin, cv=cv_folds)
                oof_r = cross_val_predict(rf_r, X_tr, y_train_margin, cv=cv_folds)
                oof_e = cross_val_predict(enet, X_tr, y_train_margin, cv=cv_folds)

                gbm.fit(X_tr, y_train_margin)
                rf_r.fit(X_tr, y_train_margin)
                enet.fit(X_tr, y_train_margin)

                meta_X = np.column_stack([oof_g, oof_r, oof_e])
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                meta.fit(meta_X, y_train_margin)

                # R6: Compute bias correction from OOF residuals
                oof_meta = meta.predict(meta_X)
                bias_correction = float(np.mean(oof_meta - y_train_margin))

                test_meta_X = np.column_stack([
                    gbm.predict(X_te), rf_r.predict(X_te), enet.predict(X_te)
                ])
                pred_margin = meta.predict(test_meta_X) - bias_correction  # R6: apply

                # Stacked classifier
                gbm_c = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.06, subsample=0.8,
                    min_samples_leaf=20, random_state=42,
                )
                rf_c = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    min_samples_leaf=15, max_features=0.7,
                    random_state=42, n_jobs=1,
                )
                lr_c = LogisticRegression(max_iter=1000, C=1.0)

                oof_gc = cross_val_predict(gbm_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_rc = cross_val_predict(rf_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]
                oof_lc = cross_val_predict(lr_c, X_tr, y_train_win, cv=cv_folds, method="predict_proba")[:, 1]

                gbm_c.fit(X_tr, y_train_win)
                rf_c.fit(X_tr, y_train_win)
                lr_c.fit(X_tr, y_train_win)

                meta_clf = LogisticRegression(max_iter=1000, C=1.0)
                oof_clf_X = np.column_stack([oof_gc, oof_rc, oof_lc])
                meta_clf.fit(oof_clf_X, y_train_win)

                # R4: Isotonic calibration on OOF stacked probs
                oof_stacked_p = meta_clf.predict_proba(oof_clf_X)[:, 1]
                iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
                iso.fit(oof_stacked_p, y_train_win)

                test_clf_X = np.column_stack([
                    gbm_c.predict_proba(X_te)[:, 1],
                    rf_c.predict_proba(X_te)[:, 1],
                    lr_c.predict_proba(X_te)[:, 1],
                ])
                raw_wp = meta_clf.predict_proba(test_clf_X)[:, 1]
                pred_wp = iso.predict(raw_wp)  # R4: isotonic calibrated
            else:
                reg = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.08, subsample=0.8,
                    min_samples_leaf=15, random_state=42,
                )
                reg.fit(X_tr, y_train_margin)
                pred_margin = reg.predict(X_te)

                clf = CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000), cv=cv_folds
                )
                clf.fit(X_tr, y_train_win)
                pred_wp = clf.predict_proba(X_te)[:, 1]

            pred_pick = (pred_wp >= 0.5).astype(int)

            accuracy = float(np.mean(pred_pick == y_test_win))
            mae_margin = float(mean_absolute_error(y_test_margin, pred_margin))
            brier = float(brier_score_loss(y_test_win, pred_wp))

            # Heuristic baseline (the v16 formula predictions stored in win_pct_home)
            heur_wp = test_df["win_pct_home"].fillna(0.5).values
            heur_pick = (heur_wp >= 0.5).astype(int)
            heur_acc = float(np.mean(heur_pick == y_test_win))
            heur_brier = float(brier_score_loss(y_test_win, heur_wp))

            results_by_month.append({
                "month": str(test_month),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "ml_accuracy": round(accuracy, 4),
                "ml_brier": round(brier, 4),
                "ml_mae_margin": round(mae_margin, 3),
                "heuristic_accuracy": round(heur_acc, 4),
                "heuristic_brier": round(heur_brier, 4),
                "home_win_rate": round(float(y_test_win.mean()), 3),
            })

            for j in range(len(test_df)):
                all_predictions.append({
                    "month": str(test_month),
                    "pred_win_prob": round(float(pred_wp[j]), 4),
                    "heur_win_prob": round(float(heur_wp[j]), 4),
                    "pred_margin": round(float(pred_margin[j]), 2),
                    "actual_margin": int(y_test_margin[j]),
                    "actual_home_win": int(y_test_win[j]),
                    "ml_correct": int(pred_pick[j] == y_test_win[j]),
                    "heur_correct": int(heur_pick[j] == y_test_win[j]),
                })

        if not results_by_month:
            return jsonify({"error": f"No months with >= {min_train} training games"})

        total = sum(r["n_test"] for r in results_by_month)
        agg = {
            "total_games_tested": total,
            "months_tested": len(results_by_month),
            "ml_overall_accuracy": round(sum(r["ml_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_brier": round(sum(r["ml_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "ml_overall_mae_margin": round(sum(r["ml_mae_margin"] * r["n_test"] for r in results_by_month) / total, 3),
            "heur_overall_accuracy": round(sum(r["heuristic_accuracy"] * r["n_test"] for r in results_by_month) / total, 4),
            "heur_overall_brier": round(sum(r["heuristic_brier"] * r["n_test"] for r in results_by_month) / total, 4),
            "baseline_home_always": round(sum(r["home_win_rate"] * r["n_test"] for r in results_by_month) / total, 4),
        }

        # Confidence tier analysis
        conf_results = []
        all_preds_arr = np.array([(p["pred_win_prob"], p["actual_home_win"]) for p in all_predictions])
        if len(all_preds_arr) > 0:
            for t in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
                strong = (all_preds_arr[:, 0] >= t) | (all_preds_arr[:, 0] <= (1 - t))
                ns = int(strong.sum())
                if ns > 0:
                    pred_side = (all_preds_arr[strong, 0] >= 0.5).astype(int)
                    actual = all_preds_arr[strong, 1].astype(int)
                    conf_results.append({
                        "min_confidence": f"{t:.0%}",
                        "n_games": ns,
                        "accuracy": round(float(np.mean(pred_side == actual)), 4),
                    })

        # F9: Empirical sigma calibration
        sigma_calibration = None
        if len(all_predictions) >= 100:
            from scipy.optimize import minimize_scalar
            spreads = np.array([p["pred_margin"] for p in all_predictions])
            actuals = np.array([p["actual_home_win"] for p in all_predictions])
            def brier_for_sigma(sigma):
                probs = 1 / (1 + np.power(10, -spreads / sigma))
                return np.mean((probs - actuals) ** 2)
            result = minimize_scalar(brier_for_sigma, bounds=(7.0, 16.0), method="bounded")
            sigma_calibration = {
                "optimal_sigma": round(result.x, 2),
                "brier_at_optimal": round(result.fun, 5),
                "brier_at_11": round(brier_for_sigma(11.0), 5),
                "n_games": len(all_predictions),
                "recommendation": f"Set SIGMA = {result.x:.1f} in ncaaUtils.js ncaaPredictGame()"
            }

        return jsonify({
            "status": "backtest_complete",
            "aggregate": agg,
            "by_month": results_by_month,
            "confidence_tiers": conf_results,
            "n_predictions": len(all_predictions),
            "sample_predictions": all_predictions[:20],
            "sigma_calibration": sigma_calibration,
        })
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# AUTO-TRAINING SYSTEM
# ═══════════════════════════════════════════════════════════════
# POST /cron/auto-train  — Called daily by Railway cron at 4 AM ET
# GET  /cron/status      — Model freshness & last training run
# POST /train/all-logged — Manual retrain with shadow comparison + logging
# ═══════════════════════════════════════════════════════════════
