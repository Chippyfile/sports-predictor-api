#!/usr/bin/env python3
"""
ncaa_retrain_spread.py — NCAA Spread Model: Lasso + LightGBM
=============================================================
Architecture: Lasso(0.1) + LightGBM(300) → simple average
Features: 43 (Lasso L1 selection from 148 — validated to beat full set)
Data filters: No 2020/2021, no games before Nov 10 of any season

Usage:
    python3 ncaa_retrain_spread.py              # Train + evaluate
    python3 ncaa_retrain_spread.py --upload     # Train + upload to Supabase as 'ncaa'
    python3 ncaa_retrain_spread.py --refresh    # Pull fresh data first
"""
import sys, os, time, warnings, copy
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib, io, base64, requests
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
from lightgbm import LGBMRegressor
import shap

from ml_utils import StackedRegressor, StackedClassifier
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')

SEED = 42
N_FOLDS = 30

# 43 features — validated: Lasso-kept set beats full 148 on MAE and ATS
# Phase 3 validation: MAE 8.643 (vs 8.650), ATS@7 89.7% (vs 88.0%)
FEATURES_43 = [
    # Dominant (|coef| > 0.4)
    "mkt_spread",              # -5.9648
    "player_rating_diff",      # +0.8672
    "ref_home_whistle",        # +0.6003
    "weakest_starter_diff",    # +0.5002
    "crowd_shock_diff",        # +0.4633
    "lineup_stability_diff",   # +0.4629
    "lineup_changes_diff",     # -0.4554
    "adj_oe_diff",             # +0.4115
    # Strong (|coef| 0.1-0.4)
    "hca_pts",                 # +0.2476
    "blowout_asym_diff",       # +0.2454
    "threepct_diff",           # -0.2304
    "pit_sos_diff",            # +0.1911
    "orb_pct_diff",            # +0.1856
    "blocks_diff",             # +0.1789
    "drb_pct_diff",            # +0.1472
    "opp_to_rate_diff",        # +0.1412
    "elo_diff",                # +0.1387
    "is_early",                # +0.1336
    "spread_regime",           # +0.1275
    "assist_rate_diff",        # +0.1268
    "opp_ppg_diff",            # -0.1180
    "opp_suppression_diff",    # +0.1155
    "roll_ats_margin_gated",   # -0.1090
    "has_ats_data",            # -0.1002
    # Moderate (|coef| 0.05-0.1)
    "tempo_avg",               # +0.0912
    "form_x_familiarity",      # +0.0797
    "to_conversion_diff",      # -0.0745
    "conf_strength_diff",      # +0.0739
    "roll_rotation_diff",      # +0.0714
    "roll_dominance_diff",     # +0.0677
    "importance",              # -0.0614
    "twopt_diff",              # +0.0571
    "roll_ats_diff_gated",     # -0.0552
    "overreaction_diff",       # -0.0551
    "three_rate_diff",         # +0.0547
    # Weak but kept (|coef| < 0.05)
    "ppp_diff",                # +0.0470
    "to_margin_diff",          # +0.0449
    "momentum_halflife_diff",  # -0.0379
    "starter_experience_diff", # +0.0269
    "style_familiarity",       # -0.0189
    "fatigue_x_quality",       # +0.0169
    "ato_diff",                # +0.0066
    "consistency_x_spread",    # +0.0055
]


def time_series_oof(models_dict, X_s, y, n_splits):
    """Walk-forward OOF for a simple-average ensemble."""
    n = len(X_s)
    oof = np.full(n, np.nan)
    fold_size = n // (n_splits + 1)
    min_train = fold_size * 2

    for fold in range(n_splits):
        ts = min_train + fold * fold_size
        te = min(ts + fold_size, n)
        if ts >= n: break

        preds = []
        for name, builder in models_dict.items():
            m = builder()
            m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)

    return oof


def ats_analysis(oof, y, spreads):
    """ATS accuracy at various thresholds."""
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.1)
    p = oof[valid]; a = y[valid]; s = spreads[valid]
    mae = float(np.mean(np.abs(p - a)))
    edge = p - (-s); margin = a + s
    np_ = margin != 0; cor = np.sign(edge) == np.sign(margin)

    print(f"\n  Walk-forward MAE: {mae:.3f}")
    print(f"  Games with spreads: {valid.sum()}")
    print(f"\n  {'Edge':>6s} {'Games':>7s} {'Acc':>6s} {'ROI':>7s}")
    print("  " + "-" * 30)

    results = {}
    for t in [0, 2, 4, 6, 7, 8, 10, 12]:
        m = (np.abs(edge) >= t) & np_; n = m.sum()
        if n < 20: continue
        acc = float(cor[m].mean())
        roi = round((acc * 1.909 - 1) * 100, 1)
        tag = "YES" if acc > 0.524 else "no"
        print(f"  {t:>5d}+ {n:>7d} {acc:>5.1%} {roi:>+6.1f}%  {tag}")
        results[t] = {"acc": acc, "n": n, "roi": roi}

    return mae, results


# ══════════════════════════════════════════════════════════
# LOAD + FILTER DATA
# ══════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA SPREAD MODEL — Lasso + LightGBM (43 features)")
print("=" * 70)

upload = "--upload" in sys.argv
refresh = "--refresh" in sys.argv

print("\n  Loading data...")
t0 = time.time()

if refresh:
    df = dump()
else:
    df = load_cached()
    if df is None: df = dump()

df = df[df["actual_home_score"].notna()].copy()

# ── Data filters ──
n_before = len(df)
df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
n_covid = n_before - len(df)
if n_covid > 0:
    print(f"  Dropped {n_covid} games from 2020/2021")

# Drop games before Nov 10 of any season
n_before = len(df)
df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
nov10_mask = ~((df["game_date_dt"].dt.month < 11) |
               ((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10)))
# Also keep March/April (tournament) — month >= 11 OR month <= 4
season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
df = df[season_mask & early_mask].copy()
n_early = n_before - len(df)
if n_early > 0:
    print(f"  Dropped {n_early} games before Nov 10 / off-season")
df = df.drop(columns=["game_date_dt"], errors="ignore")

print(f"  {len(df)} games after filters ({time.time()-t0:.0f}s)")

# ESPN odds fallback
if "espn_spread" in df.columns:
    espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
    mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
    df.loc[fill, "market_spread_home"] = espn_s[fill]
if "espn_over_under" in df.columns:
    espn_ou = pd.to_numeric(df["espn_over_under"], errors="coerce")
    mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
    fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
    df.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

# Quality filter
_qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
_keep = _qmat.mean(axis=1) >= 0.8
if "referee_1" in df.columns:
    _has_ref = df["referee_1"].notna() & (df["referee_1"] != "")
    _keep = _keep & _has_ref
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score","actual_away_score","home_adj_em","away_adj_em",
            "home_adj_oe","away_adj_oe","home_adj_de","away_adj_de",
            "home_ppg","away_ppg","home_opp_ppg","away_opp_ppg",
            "home_tempo","away_tempo","home_rank","away_rank","season",
            "home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (datetime.utcnow().year - s) <= 0 else 0.9 if (datetime.utcnow().year - s) == 1 else
    0.75 if (datetime.utcnow().year - s) == 2 else 0.6 if (datetime.utcnow().year - s) == 3 else 0.5)

print("  Heuristic backfill...")
df = _ncaa_backfill_heuristic(df)
print("  Computing crowd_shock + h2h/conf/form...")
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)

try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError: pass

try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
    print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
except FileNotFoundError:
    print("  referee_profiles.json not found")

df = df.dropna(subset=["actual_home_score","actual_away_score"])
print("  Building features...")
X_full = ncaa_build_features(df)

# Select the 43 validated features
available = [f for f in FEATURES_43 if f in X_full.columns]
missing = [f for f in FEATURES_43 if f not in X_full.columns]
if missing:
    print(f"  ⚠️ Missing features: {missing}")
X = X_full[available]
feature_cols = available

y_margin = df["actual_home_score"].values - df["actual_away_score"].values
y_win = (y_margin > 0).astype(int)
spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
weights = df["season_weight"].values
n = len(X)

print(f"  {n} games × {len(feature_cols)} features")
print(f"  Seasons: {sorted(df['season'].unique())}")

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════

LEARNERS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}

print(f"\n  {N_FOLDS}-fold walk-forward (Lasso + LightGBM avg)...")
oof = time_series_oof(LEARNERS, X_s, y_margin, N_FOLDS)
mae, ats_results = ats_analysis(oof, y_margin, spreads)

# ══════════════════════════════════════════════════════════
# PRODUCTION TRAINING
# ══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  PRODUCTION TRAINING")
print(f"{'='*70}")

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_s, y_margin)
lasso_mae = mean_absolute_error(y_margin, lasso.predict(X_s))
n_kept = np.sum(np.abs(lasso.coef_) > 1e-6)
print(f"  Lasso in-sample MAE: {lasso_mae:.3f} ({n_kept}/{len(feature_cols)} features kept)")

lgbm = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                      subsample=0.8, verbose=-1, random_state=SEED)
lgbm.fit(X_s, y_margin)
lgbm_mae = mean_absolute_error(y_margin, lgbm.predict(X_s))
print(f"  LightGBM in-sample MAE: {lgbm_mae:.3f}")

avg_pred = (lasso.predict(X_s) + lgbm.predict(X_s)) / 2
avg_mae = mean_absolute_error(y_margin, avg_pred)
print(f"  Ensemble in-sample MAE: {avg_mae:.3f}")

bias = float(np.mean(y_margin[~np.isnan(oof)] - oof[~np.isnan(oof)]))

# SHAP from LightGBM (tree-based explainer)
print("  Building SHAP explainer...")
explainer = shap.TreeExplainer(lgbm)

# Wrap for Railway pickle compatibility
from sklearn.linear_model import Ridge as _Ridge
_meta = _Ridge(alpha=0.01, fit_intercept=False)
_oof_valid = oof[~np.isnan(oof)]
_meta.fit(_oof_valid.reshape(-1, 1), y_margin[~np.isnan(oof)])
print(f"  Meta weight: {_meta.coef_[0]:.4f}")
reg = StackedRegressor([lasso, lgbm], _meta)

# ── Classifier (for win probability) ──
print("  Training classifier...")
from catboost import CatBoostClassifier
cat_c = CatBoostClassifier(n_estimators=650, depth=4, learning_rate=0.10, random_seed=SEED, verbose=0)

oof_probs = np.zeros(n)
fold_size = n // (N_FOLDS + 1)
for i in range(N_FOLDS):
    te = fold_size * (i + 2); vs = te; ve = min(te + fold_size, n)
    if vs >= n: break
    cc = copy.deepcopy(cat_c)
    cc.fit(X_s[:te], y_win[:te], sample_weight=weights[:te])
    oof_probs[vs:ve] = cc.predict_proba(X_s[vs:ve])[:, 1]

valid_mask = oof_probs != 0
isotonic = IsotonicRegression(out_of_bounds="clip")
isotonic.fit(oof_probs[valid_mask], y_win[valid_mask])

brier_raw = brier_score_loss(y_win[valid_mask], oof_probs[valid_mask])
brier_cal = brier_score_loss(y_win[valid_mask], isotonic.predict(oof_probs[valid_mask]))
print(f"  Brier (raw): {brier_raw:.4f}  Brier (cal): {brier_cal:.4f}")

cat_c.fit(X_s, y_win, sample_weight=weights)
_meta_clf = LogisticRegression(max_iter=2000)
_meta_clf.fit(oof_probs[valid_mask].reshape(-1, 1), y_win[valid_mask])
clf = StackedClassifier([cat_c], _meta_clf)

# ══════════════════════════════════════════════════════════
# BUNDLE + SAVE
# ══════════════════════════════════════════════════════════

ats7 = ats_results.get(7, {})
ats10 = ats_results.get(10, {})

bundle = {
    "scaler": scaler, "reg": reg, "clf": clf, "explainer": explainer,
    "feature_cols": feature_cols, "n_train": n,
    "mae_cv": round(mae, 3),
    "model_type": "Lasso_LGBM_v30",
    "architecture": "Lasso_0.1 + LightGBM_300 (simple avg, 43 features)",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 3),
    "isotonic": isotonic,
    "meta_weights": list(_meta.coef_.round(4)),
    "cv_ats_7_acc": ats7.get("acc", 0),
    "cv_ats_7_n": ats7.get("n", 0),
    "dropped_duplicates": None,
    "feature_selection": "Lasso L1 kept 43/148 — validated: MAE 8.643 beats 148f MAE 8.650",
    "data_filters": "no 2020/2021, no games before Nov 10",
}

print(f"\n  Verifying class paths...")
print(f"    reg: {type(bundle['reg']).__module__}.{type(bundle['reg']).__name__}")
print(f"    clf: {type(bundle['clf']).__module__}.{type(bundle['clf']).__name__}")

print("  Compressing...")
buf = io.BytesIO()
joblib.dump(bundle, buf, compress=3)
compressed = buf.getvalue()
print(f"  Size: {len(compressed)//1024} KB")

# Save locally
joblib.dump(bundle, "ncaa_model_local.pkl", compress=3)

if upload:
    print("  Uploading to Supabase as 'ncaa'...")
    b64 = base64.b64encode(compressed).decode('ascii')
    resp = requests.post(
        f'{SUPABASE_URL}/rest/v1/model_store',
        headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
                 'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
        json={'name': 'ncaa', 'data': b64,
              'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'],
                           'n_train': n, 'model_type': bundle['model_type'],
                           'architecture': bundle['architecture'],
                           'size_bytes': len(compressed)},
              'updated_at': datetime.now(timezone.utc).isoformat()},
        timeout=300)
    if resp.ok:
        print(f'  ✅ Upload successful ({len(compressed)//1024} KB)')
    else:
        print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')
else:
    print(f"  To upload: python3 ncaa_retrain_spread.py --upload")

print(f"\n{'='*70}")
print(f"  NCAA SPREAD MODEL COMPLETE")
print(f"  Architecture: Lasso + LightGBM (simple avg)")
print(f"  Features: {len(feature_cols)}")
print(f"  Games: {n}")
print(f"  Walk-forward MAE: {mae:.3f}")
print(f"  ATS 7+: {ats7.get('acc',0):.1%} ({ats7.get('n',0)} picks)")
print(f"  Bias: {bias:+.3f}")
print(f"{'='*70}")
