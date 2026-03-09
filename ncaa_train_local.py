#!/usr/bin/env python3
"""
ncaa_train_local.py — Train NCAA model LOCALLY with full dataset
═════════════════════════════════════════════════════════════════
Bypasses Railway memory limits. Pulls all 64K+ games from Supabase,
trains the stacking ensemble, and uploads the model bundle directly
to Supabase model_store. Railway loads it on next prediction request.

Run:
  cd ~/Desktop/sports-predictor-api
  SUPABASE_ANON_KEY="..." python3 ncaa_train_local.py
"""
import os, sys, json, time, requests, io, base64, warnings
import numpy as np, pandas as pd, joblib
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── ML imports ──
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, brier_score_loss
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed. pip install xgboost")
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("WARNING: CatBoost not installed. pip install catboost")

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# SUPABASE HELPERS
# ═══════════════════════════════════════════════════════════════
def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=60)
        if not r.ok:
            print(f"  Error: {r.text[:200]}")
            break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data


def upload_model(name, bundle):
    """Upload trained model to Supabase model_store (same as Railway's save_model)."""
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    metadata = {
        "trained_at": bundle.get("trained_at"),
        "mae_cv": bundle.get("mae_cv"),
        "n_train": bundle.get("n_train"),
        "model_type": bundle.get("model_type", ""),
        "size_bytes": len(buf.getvalue()),
        "trained_locally": True,
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/model_store",
        headers=headers,
        json={"name": name, "data": b64, "metadata": metadata,
              "updated_at": datetime.utcnow().isoformat()},
        timeout=120,
    )
    if resp.ok:
        print(f"  ✅ Model '{name}' uploaded to Supabase ({len(buf.getvalue())/1024:.0f} KB)")
        return True
    else:
        print(f"  ❌ Upload failed: {resp.status_code} {resp.text[:200]}")
        return False


# ═══════════════════════════════════════════════════════════════
# IMPORT ncaa_build_features FROM LOCAL REPO
# ═══════════════════════════════════════════════════════════════
# Add repo to path so we can import sports/ncaa.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic, _ncaa_season_weight
    print("  Imported ncaa_build_features from sports/ncaa.py")
except ImportError as e:
    print(f"  ERROR importing: {e}")
    print("  Make sure you're running from ~/Desktop/sports-predictor-api/")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# STACKING ENSEMBLE WRAPPERS (same as Railway)
# ═══════════════════════════════════════════════════════════════
class StackedRegressor:
    def __init__(self, base_models, meta_model, base_scalers=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_scalers = base_scalers or [None] * len(base_models)
    def predict(self, X):
        base_preds = []
        for model, scaler in zip(self.base_models, self.base_scalers):
            X_in = scaler.transform(X) if scaler else X
            base_preds.append(model.predict(X_in))
        stacked = np.column_stack(base_preds)
        return self.meta_model.predict(stacked)

class StackedClassifier:
    def __init__(self, base_models, meta_model, base_scalers=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_scalers = base_scalers or [None] * len(base_models)
    def predict_proba(self, X):
        base_preds = []
        for model, scaler in zip(self.base_models, self.base_scalers):
            X_in = scaler.transform(X) if scaler else X
            base_preds.append(model.predict(X_in))
        stacked = np.column_stack(base_preds)
        return self.meta_model.predict_proba(stacked)


# ═══════════════════════════════════════════════════════════════
# TIME-SERIES OOF
# ═══════════════════════════════════════════════════════════════
def time_series_oof(model, X, y, n_splits=5, scaler=None):
    """Generate out-of-fold predictions using time-series splits."""
    n = len(X)
    oof = np.zeros(n)
    fold_size = n // (n_splits + 1)

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_start >= n: break

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[val_start:val_end]

        if scaler:
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
        else:
            X_train_s, X_val_s = X_train, X_val

        import copy
        m = copy.deepcopy(model)
        m.fit(X_train_s, y_train)
        oof[val_start:val_end] = m.predict(X_val_s)

    return oof


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  NCAA LOCAL TRAINING — Full 64K Dataset")
    print("=" * 70)

    # ── 1. Load ALL historical data ──
    print("\n  Loading ncaa_historical...")
    t0 = time.time()
    hist_rows = sb_get("ncaa_historical",
                       "actual_home_score=not.is.null&select=*&order=season.asc")
    print(f"  Loaded {len(hist_rows)} rows in {time.time()-t0:.0f}s")

    if not hist_rows:
        print("ERROR: No data"); return

    hist_df = pd.DataFrame(hist_rows)

    # Numeric conversion
    numeric_cols = [
        "actual_home_score", "actual_away_score", "home_win",
        "home_adj_em", "away_adj_em", "home_adj_oe", "away_adj_oe",
        "home_adj_de", "away_adj_de", "home_ppg", "away_ppg",
        "home_opp_ppg", "away_opp_ppg", "home_tempo", "away_tempo",
        "home_record_wins", "away_record_wins",
        "home_record_losses", "away_record_losses",
        "home_rank", "away_rank", "season",
    ]
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors="coerce")

    # Season weight
    current_year = datetime.utcnow().year
    hist_df["season_weight"] = hist_df["season"].apply(
        lambda s: 1.0 if (current_year - s) <= 0 else
                  0.9 if (current_year - s) == 1 else
                  0.75 if (current_year - s) == 2 else
                  0.6 if (current_year - s) == 3 else 0.5
    )

    # Column alignment
    if "home_record_wins" in hist_df.columns and "home_wins" not in hist_df.columns:
        hist_df["home_wins"] = hist_df["home_record_wins"]
    if "away_record_wins" in hist_df.columns and "away_wins" not in hist_df.columns:
        hist_df["away_wins"] = hist_df["away_record_wins"]
    if "home_record_losses" in hist_df.columns and "home_losses" not in hist_df.columns:
        hist_df["home_losses"] = hist_df["home_record_losses"]
    if "away_record_losses" in hist_df.columns and "away_losses" not in hist_df.columns:
        hist_df["away_losses"] = hist_df["away_record_losses"]

    # ── 2. Heuristic backfill ──
    print("  Running heuristic backfill...")
    hist_df = _ncaa_backfill_heuristic(hist_df)

    # ── 3. Build features ──
    print("  Building features...")
    df = hist_df.copy()
    df["actual_home_score"] = pd.to_numeric(df["actual_home_score"], errors="coerce")
    df["actual_away_score"] = pd.to_numeric(df["actual_away_score"], errors="coerce")
    df = df.dropna(subset=["actual_home_score", "actual_away_score"])

    X = ncaa_build_features(df)
    y_margin = df["actual_home_score"].values - df["actual_away_score"].values
    y_win = (y_margin > 0).astype(int)

    n = len(X)
    print(f"  Training set: {n} games × {X.shape[1]} features")

    if n < 200:
        print("ERROR: Not enough data"); return

    # ── 4. Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sample weights
    weights = df["season_weight"].values if "season_weight" in df.columns else np.ones(n)

    # ── 5. Train base learners + OOF ──
    print("\n  Training base learners...")
    cv_folds = 10

    # XGBoost
    if HAS_XGB:
        print("    XGBoost...", end=" ", flush=True)
        xgb = XGBRegressor(n_estimators=160, max_depth=5, learning_rate=0.06,
                           random_state=42, tree_method="hist")
        oof_xgb = time_series_oof(xgb, X_scaled, y_margin, n_splits=cv_folds)
        xgb.fit(X_scaled, y_margin, sample_weight=weights)
        print(f"MAE: {mean_absolute_error(y_margin, oof_xgb):.3f}")

    # CatBoost
    if HAS_CAT:
        print("    CatBoost...", end=" ", flush=True)
        cat = CatBoostRegressor(n_estimators=160, depth=5, learning_rate=0.06,
                                random_seed=42, verbose=0)
        oof_cat = time_series_oof(cat, X_scaled, y_margin, n_splits=cv_folds)
        cat.fit(X_scaled, y_margin, sample_weight=weights)
        print(f"MAE: {mean_absolute_error(y_margin, oof_cat):.3f}")

    # RandomForest
    print("    RandomForest...", end=" ", flush=True)
    rf = RandomForestRegressor(n_estimators=160, max_depth=10, random_state=42, n_jobs=-1)
    oof_rf = time_series_oof(rf, X_scaled, y_margin, n_splits=cv_folds)
    rf.fit(X_scaled, y_margin, sample_weight=weights)
    print(f"MAE: {mean_absolute_error(y_margin, oof_rf):.3f}")

    # ── 6. Stack ──
    print("\n  Stacking...")
    base_models = []
    base_scalers = []
    oof_list = []

    if HAS_XGB:
        base_models.append(xgb); base_scalers.append(None); oof_list.append(oof_xgb)
    if HAS_CAT:
        base_models.append(cat); base_scalers.append(None); oof_list.append(oof_cat)
    base_models.append(rf); base_scalers.append(None); oof_list.append(oof_rf)

    oof_stacked = np.column_stack(oof_list)

    # Meta-learner
    meta_reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_reg.fit(oof_stacked, y_margin)
    meta_weights = list(meta_reg.coef_.round(4))
    print(f"  Meta weights: {meta_weights}")

    # Stacked predictions
    stacked_preds = meta_reg.predict(oof_stacked)
    stacked_mae = mean_absolute_error(y_margin, stacked_preds)
    print(f"  Stacked OOF MAE: {stacked_mae:.3f}")

    # ── 7. Classifier ──
    print("  Training classifier...")
    meta_clf = LogisticRegression(max_iter=2000)
    oof_stacked_clf = oof_stacked.copy()
    meta_clf.fit(oof_stacked_clf, y_win)

    # ── 8. Bias correction ──
    residuals = y_margin - stacked_preds
    bias = float(np.mean(residuals))
    print(f"  Bias correction: {bias:.3f}")

    # ── 9. Isotonic calibration ──
    print("  Isotonic calibration...")
    oof_probs = meta_clf.predict_proba(oof_stacked_clf)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(oof_probs, y_win)

    # ── 10. SHAP explainer ──
    print("  Building SHAP explainer...")
    import shap
    # Use the first base model for SHAP
    if HAS_XGB:
        explainer = shap.TreeExplainer(xgb)
    elif HAS_CAT:
        explainer = shap.TreeExplainer(cat)
    else:
        explainer = shap.TreeExplainer(rf)

    # ── 11. Build bundle ──
    reg = StackedRegressor(base_models, meta_reg, base_scalers)
    clf = StackedClassifier(base_models, meta_clf, base_scalers)

    bundle = {
        "scaler": scaler,
        "reg": reg,
        "clf": clf,
        "explainer": explainer,
        "feature_cols": list(X.columns),
        "n_train": n,
        "n_historical": n,
        "n_current": 0,
        "mae_cv": round(stacked_mae, 3),
        "model_type": "StackedEnsemble_LOCAL_FULL",
        "trained_at": datetime.utcnow().isoformat(),
        "bias_correction": round(bias, 3),
        "isotonic": isotonic,
        "meta_weights": meta_weights,
    }

    # ── 12. Upload to Supabase ──
    print(f"\n  Uploading model to Supabase...")
    success = upload_model("ncaa", bundle)

    # ── 13. Summary ──
    print(f"\n{'='*70}")
    print(f"  LOCAL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Games:         {n}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Stacked MAE:   {stacked_mae:.3f}")
    print(f"  Bias:          {bias:.3f}")
    print(f"  Meta weights:  {meta_weights}")
    print(f"  Model type:    StackedEnsemble_LOCAL_FULL")
    print(f"  Upload:        {'✅ Success' if success else '❌ Failed'}")
    print(f"")
    print(f"  Railway will auto-load from Supabase on next prediction.")
    print(f"  To verify: curl $RAILWAY_API/model-info/ncaa")
    print(f"  To backtest: curl -X POST $RAILWAY_API/backtest/ncaa ...")
    print(f"{'='*70}")

    # Also save locally
    joblib.dump(bundle, "ncaa_model_local.pkl")
    print(f"  Local backup: ncaa_model_local.pkl")


if __name__ == "__main__":
    main()
