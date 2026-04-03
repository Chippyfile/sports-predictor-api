#!/usr/bin/env python3
"""
ncaa_retrain_ou_v4.py — NCAA O/U Triple Agreement System
==========================================================
Architecture:
  3 residual models (predict market error) — Lasso + LGBM + CatBoost (37 tight features)
  2 classifier models (predict P(under)) — LogReg + LGBM (13 tight features)
  ATS score models (predict home + away scores) — Lasso + LGBM ensemble (44 ATS features)

Signal: All three averaged signals must agree → UNDER or OVER

UNDER tiers:
  1u: res_avg ≤ -0.5, cls_avg ≥ 54%, ats_edge ≤ -1  (58.6%, +11.8% ROI)
  2u: res_avg ≤ -1.0, cls_avg ≥ 54%, ats_edge ≤ -4  (61.3%, +16.9% ROI)
  3u: res_avg ≤ -1.5, cls_avg ≥ 58%, ats_edge ≤ -5  (65.7%, +25.5% ROI)

OVER tiers:
  1u: res_avg ≥ +2.0, cls_avg ≤ 42%, ats_edge ≥ +3  (60.0%, +14.5% ROI)
  2u: res_avg ≥ +2.0, cls_avg ≤ 42%, ats_edge ≥ +6  (62.4%, +19.1% ROI)

Validated:
  - Season stability: profitable 8/9 seasons UNDER, 8/8 OVER
  - True 2026 holdout: 63-80% accuracy (better than walk-forward)
  - Ref-independent: 57.6% without top 5 refs
  - Clean data: real O/U only, zero missing features, non-push

Usage:
    python3 ncaa_retrain_ou_v4.py              # Train + evaluate
    python3 ncaa_retrain_ou_v4.py --upload     # Train + upload to Supabase
    python3 ncaa_retrain_ou_v4.py --refresh    # Pull fresh data first
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib, io, base64, requests
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
SEED = 42
N_FOLDS = 30
CURRENT_YEAR = 2026

# ATS features (same as production ATS model)
from ncaa_final_retrain import FEATURES_NEW as ATS_FEATURES, MODELS as ATS_MODELS

# Thresholds (validated via walk-forward + holdout)
UNDER_TIERS = {
    1: {"res_avg": -0.5, "cls_avg": 0.54, "ats_edge": -1},
    2: {"res_avg": -1.0, "cls_avg": 0.54, "ats_edge": -4},
    3: {"res_avg": -1.5, "cls_avg": 0.58, "ats_edge": -5},
}
OVER_TIERS = {
    1: {"res_avg": 2.0, "cls_avg": 0.42, "ats_edge": 3},
    2: {"res_avg": 2.0, "cls_avg": 0.42, "ats_edge": 6},
}


def compute_rolling_hca_col(df, window=20):
    df = df.sort_values("game_date").reset_index(drop=True)
    margins = df["actual_home_score"].values - df["actual_away_score"].values
    h_ids = df["home_team_id"].astype(str).values
    a_ids = df["away_team_id"].astype(str).values
    from collections import defaultdict
    home_margins = defaultdict(list); away_margins = defaultdict(list)
    rolling_hca = np.full(len(df), 6.6)
    for i in range(len(df)):
        hid, aid = h_ids[i], a_ids[i]
        h_hist = home_margins[hid][-window:]; a_hist = away_margins[hid][-window:]
        if len(h_hist) >= 5 and len(a_hist) >= 5:
            rolling_hca[i] = float(np.mean(h_hist) - np.mean(a_hist)) / 2
        home_margins[hid].append(margins[i]); away_margins[aid].append(-margins[i])
    df["rolling_hca"] = rolling_hca
    print(f"  Rolling HCA: mean={df['rolling_hca'].mean():+.2f}, coverage={(rolling_hca!=6.6).sum()/len(df)*100:.1f}%")
    return df


def select_features(X_s, y_res, y_under, w, all_feats):
    """Select tight features for residual and classifier."""
    # Residual: Lasso α=0.1 → ~37 features
    lasso = Lasso(alpha=0.1, max_iter=5000)
    lasso.fit(X_s, y_res, sample_weight=w)
    res_feats = [f for f, c in zip(all_feats, lasso.coef_) if abs(c) > 0.001]

    # Classifier: LogReg C=0.01 → ~13 features
    lr = LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=5000, random_state=SEED)
    lr.fit(X_s, y_under, sample_weight=w)
    cls_feats = [f for f, c in zip(all_feats, lr.coef_[0]) if abs(c) > 0.001]

    return res_feats, cls_feats


def ou_eval(res_avg, cls_avg, ats_edge, y_under, label=""):
    """Evaluate O/U triple agreement system."""
    yo = 1 - y_under
    valid = ~np.isnan(res_avg) & ~np.isnan(cls_avg) & ~np.isnan(ats_edge)

    print(f"\n  {label} ({valid.sum()} games):")
    print(f"  {'Dir':>6s} {'Tier':>5s} {'Res':>8s} {'Cls':>8s} {'ATS':>8s} {'Picks':>7s} {'Acc':>6s} {'ROI':>7s}")
    print(f"  {'-'*55}")

    for tier, t in UNDER_TIERS.items():
        mask = valid & (res_avg <= t["res_avg"]) & (cls_avg >= t["cls_avg"]) & (ats_edge <= t["ats_edge"])
        n = mask.sum()
        if n < 10: continue
        acc = y_under[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"  {'UNDER':>6s} {tier:>5d}u {t['res_avg']:>+8.1f} {t['cls_avg']:>7.0%} {t['ats_edge']:>+8d} {n:>7d} {acc:>5.1%} {roi:>+6.1f}%")

    for tier, t in OVER_TIERS.items():
        mask = valid & (res_avg >= t["res_avg"]) & (cls_avg <= t["cls_avg"]) & (ats_edge >= t["ats_edge"])
        n = mask.sum()
        if n < 10: continue
        acc = yo[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"  {'OVER':>6s} {tier:>5d}u {t['res_avg']:>+8.1f} {t['cls_avg']:>7.0%} {t['ats_edge']:>+8d} {n:>7d} {acc:>5.1%} {roi:>+6.1f}%")


# ══════════════════════════════════════════════════════════
# LOAD + FILTER DATA (clean: real O/U, refs, complete features)
# ══════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA O/U v4 — Triple Agreement System")
print("=" * 70)

upload = "--upload" in sys.argv
t0 = time.time()

if "--refresh" in sys.argv:
    df = dump()
else:
    df = load_cached()
    if df is None or len(df) == 0:
        df = dump()

df = df[df["actual_home_score"].notna()].copy()

# Force numeric
for col in ["market_spread_home","market_ou_total","espn_spread","espn_over_under",
            "dk_spread_close","dk_total_close","odds_api_spread_close","odds_api_total_close",
            "actual_home_score","actual_away_score","season"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["season"] = df["season"].fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
m = df["game_date_dt"].dt.month
df = df[((m >= 11) | (m <= 4)) & ~((m == 11) & (df["game_date_dt"].dt.day < 10))].copy()

# Cascade O/U + spread backfill
for sc, tc in [("espn_over_under","market_ou_total"),("dk_total_close","market_ou_total"),("odds_api_total_close","market_ou_total")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce")
        cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0: df.loc[fill, tc] = src[fill]
for sc, tc in [("espn_spread","market_spread_home"),("dk_spread_close","market_spread_home"),("odds_api_spread_close","market_spread_home")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce")
        cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0: df.loc[fill, tc] = src[fill]

# Filter: real O/U total (50-250)
mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
n_before = len(df)
df = df[mkt_ou.notna() & (mkt_ou > 50) & (mkt_ou < 250)].copy().reset_index(drop=True)
print(f"  Filter (real O/U): {n_before} → {len(df)}")

# Filter: refs
if "referee_1" in df.columns:
    n_before = len(df)
    df = df[df["referee_1"].notna() & (df["referee_1"] != "") & (df["referee_1"] != "None")].copy().reset_index(drop=True)
    print(f"  Filter (refs): {n_before} → {len(df)}")

for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

seasons = pd.to_numeric(df["season"], errors="coerce").values
weights = np.array([{2026:1.0,2025:0.9,2024:0.75,2023:0.6}.get(s, 0.5) for s in seasons])

print("  Heuristic backfill + features...")
df = _ncaa_backfill_heuristic(df)
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
except FileNotFoundError: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])
df = compute_rolling_hca_col(df)
df["travel_advantage"] = 0

# Build features
X_full = ncaa_build_features(df)
X_full["rolling_hca"] = df["rolling_hca"].values
X_full["travel_advantage"] = 0
raw_em = df["home_adj_em"].fillna(0).values - df["away_adj_em"].fillna(0).values
nm = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool).values
X_full["neutral_em_diff"] = raw_em - np.where(nm, 0, df["rolling_hca"].values)

y_home = df["actual_home_score"].values
y_away = df["actual_away_score"].values
y_total = y_home + y_away
mkt_total = pd.to_numeric(df["market_ou_total"], errors="coerce").fillna(0).values
push = y_total == mkt_total
y_residual = y_total - mkt_total
y_under = (y_total < mkt_total).astype(float)

all_feats = [c for c in X_full.columns if X_full[c].notna().mean() > 0.1]
ats_feats = [f for f in ATS_FEATURES if f in X_full.columns]

# Filter: complete features + non-push
complete = X_full[all_feats].isna().sum(axis=1) == 0
keep = complete & ~push
idx = np.where(keep)[0]

X = X_full.iloc[idx][all_feats].reset_index(drop=True)
X_ats = X_full.iloc[idx][ats_feats].reset_index(drop=True)
yr = y_residual[idx]; yu = y_under[idx]
yh = y_home[idx]; ya = y_away[idx]; yt = y_total[idx]
mt = mkt_total[idx]; w = weights[idx]
n = len(idx)

print(f"  {n} clean games (real O/U, refs, complete features, non-push)")
print(f"  {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════
print(f"\n  Feature selection (tight)...")
X_s = StandardScaler().fit_transform(X)
res_feats, cls_feats = select_features(X_s, yr, yu, w, all_feats)
print(f"  Residual: {len(res_feats)} features")
print(f"  Classifier: {len(cls_feats)} features")
print(f"  ATS: {len(ats_feats)} features")

# ══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════
print(f"\n  Walk-forward ({N_FOLDS} folds)...")
fs = n // (N_FOLDS + 1); min_t = fs * 2
oof_res = np.full((3, n), np.nan)
oof_cls = np.full((2, n), np.nan)
oof_ats = np.full(n, np.nan)

for fold in range(N_FOLDS):
    ts = min_t + fold * fs; te = min(ts + fs, n)
    if ts >= n: break

    # 3 residual models
    Xr = StandardScaler().fit_transform(X[res_feats])
    r1 = Lasso(alpha=0.1, max_iter=5000); r1.fit(Xr[:ts], yr[:ts])
    oof_res[0, ts:te] = r1.predict(Xr[ts:te])
    r2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    r2.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts])
    oof_res[1, ts:te] = r2.predict(Xr[ts:te])
    r3 = CatBoostRegressor(n_estimators=300, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)
    r3.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts])
    oof_res[2, ts:te] = r3.predict(Xr[ts:te])

    # 2 classifier models
    Xc = StandardScaler().fit_transform(X[cls_feats])
    c1 = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
    c1.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts])
    oof_cls[0, ts:te] = c1.predict_proba(Xc[ts:te])[:, 1]
    c2 = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    c2.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts])
    oof_cls[1, ts:te] = c2.predict_proba(Xc[ts:te])[:, 1]

    # ATS home + away score models
    Xa = StandardScaler().fit_transform(X_ats)
    hp = []; ap = []
    for name, builder in ATS_MODELS.items():
        mh = builder()
        try: mh.fit(Xa[:ts], yh[:ts], sample_weight=w[:ts])
        except TypeError: mh.fit(Xa[:ts], yh[:ts])
        hp.append(mh.predict(Xa[ts:te]))
        ma = builder()
        try: ma.fit(Xa[:ts], ya[:ts], sample_weight=w[:ts])
        except TypeError: ma.fit(Xa[:ts], ya[:ts])
        ap.append(ma.predict(Xa[ts:te]))
    oof_ats[ts:te] = np.mean(hp, axis=0) + np.mean(ap, axis=0)

    if (fold + 1) % 10 == 0: print(f"    Fold {fold+1}/{N_FOLDS}")

valid = ~np.isnan(oof_res[0])
res_avg = np.mean(oof_res, axis=0)
cls_avg = np.mean(oof_cls, axis=0)
ats_edge = oof_ats - mt

res_mae = np.mean(np.abs(res_avg[valid] - yr[valid]))
ats_mae = np.mean(np.abs(oof_ats[valid] - yt[valid]))
mkt_mae = np.mean(np.abs(mt[valid] - yt[valid]))
print(f"\n  Residual avg MAE: {res_mae:.2f}")
print(f"  ATS total MAE: {ats_mae:.2f}")
print(f"  Market total MAE: {mkt_mae:.2f}")

ou_eval(res_avg, cls_avg, ats_edge, yu, "Walk-forward")

# ══════════════════════════════════════════════════════════
# PRODUCTION TRAINING (full data)
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  PRODUCTION TRAINING")
print(f"{'='*70}")

# Scalers
res_scaler = StandardScaler().fit(X[res_feats])
cls_scaler = StandardScaler().fit(X[cls_feats])
ats_scaler = StandardScaler().fit(X_ats)

Xr = res_scaler.transform(X[res_feats])
Xc = cls_scaler.transform(X[cls_feats])
Xa = ats_scaler.transform(X_ats)

# Train residual models
print("  Training 3 residual models...")
res_models = []
r1 = Lasso(alpha=0.1, max_iter=5000); r1.fit(Xr, yr); res_models.append(r1)
r2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
r2.fit(Xr, yr, sample_weight=w); res_models.append(r2)
r3 = CatBoostRegressor(n_estimators=300, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)
r3.fit(Xr, yr, sample_weight=w); res_models.append(r3)
res_pred = np.mean([m.predict(Xr) for m in res_models], axis=0)
print(f"    Residual ensemble MAE: {mean_absolute_error(yr, res_pred):.3f}")

# Train classifier models
print("  Training 2 classifier models...")
cls_models = []
c1 = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
c1.fit(Xc, yu, sample_weight=w); cls_models.append(c1)
c2 = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
c2.fit(Xc, yu, sample_weight=w); cls_models.append(c2)
cls_pred = np.mean([m.predict_proba(Xc)[:, 1] for m in cls_models], axis=0)
print(f"    Classifier avg P(under): {cls_pred.mean():.3f}")

# Train ATS score models (home + away separately)
print("  Training ATS home + away score models...")
ats_home_models = []
ats_away_models = []
for name, builder in ATS_MODELS.items():
    mh = builder()
    try: mh.fit(Xa, yh, sample_weight=w)
    except TypeError: mh.fit(Xa, yh)
    ats_home_models.append(mh)
    ma = builder()
    try: ma.fit(Xa, ya, sample_weight=w)
    except TypeError: ma.fit(Xa, ya)
    ats_away_models.append(ma)
ats_total_pred = np.mean([m.predict(Xa) for m in ats_home_models], axis=0) + \
                 np.mean([m.predict(Xa) for m in ats_away_models], axis=0)
print(f"    ATS total MAE: {mean_absolute_error(yt, ats_total_pred):.3f}")

# ══════════════════════════════════════════════════════════
# BUNDLE + SAVE
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  BUNDLING")
print(f"{'='*70}")

bundle = {
    # Scalers
    "res_scaler": res_scaler,
    "cls_scaler": cls_scaler,
    "ats_scaler": ats_scaler,

    # Models
    "res_models": res_models,       # 3 residual models
    "cls_models": cls_models,       # 2 classifier models
    "ats_home_models": ats_home_models,
    "ats_away_models": ats_away_models,

    # Feature lists
    "res_feature_cols": res_feats,
    "cls_feature_cols": cls_feats,
    "ats_feature_cols": ats_feats,

    # Thresholds
    "under_tiers": UNDER_TIERS,
    "over_tiers": OVER_TIERS,

    # Metadata
    "n_train": n,
    "model_type": "OU_v4_triple_agreement",
    "architecture": "3res(Lasso+LGBM+Cat) + 2cls(LR+LGBM) + ATS(home+away)",
    "res_mae_cv": round(res_mae, 4),
    "ats_total_mae_cv": round(ats_mae, 2),
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "data_filters": "real_ou(50-250), refs, complete_features, non_push",
    "feature_counts": {"residual": len(res_feats), "classifier": len(cls_feats), "ats": len(ats_feats)},
}

local_path = "ncaa_ou_v4.pkl"
joblib.dump(bundle, local_path, compress=3)
size_kb = os.path.getsize(local_path) / 1024
print(f"  Saved: {local_path} ({size_kb:.0f} KB)")

if upload:
    print("  Uploading to Supabase as 'ncaa_ou'...")
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/model_store",
        headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
        json={"name": "ncaa_ou", "data": b64,
              "metadata": {"trained_at": bundle["trained_at"], "model_type": bundle["model_type"],
                           "n_train": n, "res_mae_cv": bundle["res_mae_cv"],
                           "feature_counts": bundle["feature_counts"],
                           "size_bytes": len(buf.getvalue())},
              "updated_at": datetime.now(timezone.utc).isoformat()},
        timeout=300)
    if resp.ok:
        print(f"  ✅ Upload successful ({len(buf.getvalue())//1024} KB)")
    else:
        print(f"  ❌ Failed: {resp.status_code} {resp.text[:300]}")
else:
    print(f"  To upload: python3 ncaa_retrain_ou_v4.py --upload")

print(f"\n{'='*70}")
print(f"  NCAA O/U v4 COMPLETE")
print(f"  Architecture: {bundle['architecture']}")
print(f"  Features: res={len(res_feats)}, cls={len(cls_feats)}, ats={len(ats_feats)}")
print(f"  Games: {n}")
print(f"  Residual MAE: {res_mae:.3f}")
print(f"{'='*70}")
