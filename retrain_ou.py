"""
RETRAIN O/U MODEL v2 — Data-driven feature selection
======================================================
Feature list rebuilt from three_target_eliminate.py results.

Changes from v1 (20 hand-picked features):
  REMOVED (hurt O/U per three-target analysis):
    - starter_experience_diff  (ΔOU +0.010, removing HELPS)
    - opp_fta_rate_diff        (ΔOU +0.006, removing HELPS)
    - lineup_stability_diff    (ΔOU +0.009, removing HELPS)
    - ref_foul_rate            (ΔOU +0.010, removing HELPS)
    - info_gain_diff           (ΔOU +0.014, removing HELPS)

  ADDED (essential for O/U per three-target analysis):
    - margin_trend_diff        (ΔOU -0.028, removing HURTS)
    - roll_dominance_diff      (ΔOU -0.024, removing HURTS)
    - run_vulnerability_diff   (ΔOU -0.020, removing HURTS)
    - roll_lead_change_avg     (ΔOU -0.018, removing HURTS)
    - fta_rate_diff            (ΔOU -0.016, removing HURTS)
    - pyth_residual_diff       (ΔOU -0.015, removing HURTS)
    - steal_foul_diff          (ΔOU -0.015, removing HURTS)
    - streak_diff              (ΔOU -0.015, removing HURTS)
    - matchup_orb              (ΔOU -0.011, removing HURTS) [already in v1]
    - opp_ppg_diff             (ΔOU -0.014, removing HURTS)
    - ceiling_diff             (ΔOU -0.013, removing HURTS)
    - block_foul_diff          (ΔOU -0.012, removing HURTS)
    - is_early                 (ΔOU -0.011, removing HURTS)
    - fgpct_diff               (ΔOU -0.010, removing HURTS)
    - games_since_blowout_diff (ΔOU -0.017, removing HURTS)
    - market_wp_edge           (ΔOU -0.012, removing HURTS)
    - assist_rate_diff         (ΔOU -0.014, removing HURTS) [already in v1]
    - fatigue_x_quality        (ΔOU -0.020, removing HURTS)
    - concentration_diff       (ΔOU -0.011, removing HURTS)
    - is_revenge_game          (ΔOU -0.012, removing HURTS)

Architecture: CatBoost(d=3, 650i) + MLP(256-128-64) → ElasticNet

Usage:
    python retrain_ou.py                # train locally
    python retrain_ou.py --upload       # upload to Supabase as 'ncaa_ou'
    python retrain_ou.py --refresh      # pull fresh data first
"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))

import numpy as np, pandas as pd, joblib, io, base64, requests, warnings, time, copy
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import shap
warnings.filterwarnings("ignore")

from ml_utils import StackedRegressor
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ['SUPABASE_ANON_KEY']

# ═══════════════════════════════════════════════════════════════
# O/U FEATURES — data-driven from three_target_eliminate.py
# All features where removing HURTS O/U MAE by > 0.005
# ═══════════════════════════════════════════════════════════════
OU_FEATURES = [
    # ── Dominant (ΔOU < -0.02) ──
    'mkt_total',                  # -0.681 (by far #1)
    'ref_ou_bias',                # -0.038
    'margin_trend_diff',          # -0.028
    'roll_dominance_diff',        # -0.024
    'fatigue_x_quality',          # -0.020
    'run_vulnerability_diff',     # -0.020
    # ── Strong (ΔOU -0.01 to -0.02) ──
    'roll_lead_change_avg',       # -0.018
    'games_since_blowout_diff',   # -0.017
    'fta_rate_diff',              # -0.016
    'pyth_residual_diff',         # -0.015
    'steal_foul_diff',            # -0.015
    'matchup_efg',                # -0.015
    'streak_diff',                # -0.015
    'opp_ppg_diff',               # -0.014
    'assist_rate_diff',           # -0.014
    'to_margin_diff',             # -0.013
    'ceiling_diff',               # -0.013
    'market_wp_edge',             # -0.012
    'block_foul_diff',            # -0.012
    'is_revenge_game',            # -0.012
    'hca_pts',                    # -0.012
    'is_early',                   # -0.011
    'starter_balance_diff',       # -0.011
    'concentration_diff',         # -0.011
    'matchup_orb',                # -0.011
    'fgpct_diff',                 # -0.010
    'roll_ats_margin_gated',      # -0.010
    # ── Moderate (ΔOU -0.005 to -0.010) ──
    'score_kurtosis_diff',        # -0.009
    'win_pct_diff',               # -0.009
    'matchup_to',                 # -0.008
    'venue_advantage',            # -0.008
    'threepct_diff',              # -0.008
    'scoring_entropy_diff',       # -0.008
    'roll_hhi_diff',              # -0.008
    'adj_de_diff',                # -0.007
    'momentum_halflife_diff',     # -0.007
    'blowout_asym_diff',          # -0.007
    'def_improvement_diff',       # -0.006
    'dow_effect_diff',            # -0.006
    'three_divergence_diff',      # -0.006
    'after_loss_either',          # -0.005
]

CAT_DEPTH = 3
CAT_ITERS = 650
N_FOLDS = 30


def time_series_oof(model, X, y, n_splits):
    n = len(X)
    oof = np.full(n, np.nan)
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        te = fold_size * (i + 2); vs = te; ve = min(te + fold_size, n)
        if vs >= n: break
        m = copy.deepcopy(model)
        m.fit(X[:te], y[:te])
        oof[vs:ve] = m.predict(X[vs:ve])
    return oof


def walk_forward_ou_test(y_total, oof, mkt_total, min_edges=[2, 3, 4, 5, 6, 8]):
    """Walk-forward O/U accuracy by edge threshold."""
    valid = ~np.isnan(oof) & (mkt_total > 0)
    print(f"\n  O/U walk-forward ({valid.sum()} games with market total):")
    for edge in min_edges:
        over = (oof[valid] - mkt_total[valid]) >= edge
        under = (mkt_total[valid] - oof[valid]) >= edge
        has_pick = over | under
        if has_pick.sum() < 20:
            continue
        actual_total = y_total[valid][has_pick]
        line = mkt_total[valid][has_pick]
        actual_over = actual_total > line
        actual_under = actual_total < line
        push = actual_total == line
        correct = (over[has_pick] & actual_over) | (under[has_pick] & actual_under)
        decided = ~push
        if decided.sum() == 0:
            continue
        acc = correct[decided].mean()
        roi = (acc * 1.91 - 1) * 100
        n_over = (over[has_pick] & decided).sum()
        n_under = (under[has_pick] & decided).sum()
        # Directional split
        over_correct = (over[has_pick] & actual_over & decided).sum()
        under_correct = (under[has_pick] & actual_under & decided).sum()
        over_acc = over_correct / n_over if n_over > 0 else 0
        under_acc = under_correct / n_under if n_under > 0 else 0
        print(f"    edge≥{edge}: {acc:.1%} ({decided.sum()}g, {roi:+.1f}% ROI) "
              f"| OVER {over_acc:.0%} ({n_over}g) UNDER {under_acc:.0%} ({n_under}g)")


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  RETRAIN O/U MODEL v2 (data-driven features)")
print("=" * 70)

print(f"\n  Loading data...")
t0 = time.time()

if "--refresh" in sys.argv:
    print("  --refresh: Pulling fresh data from Supabase...")
    df = dump()
else:
    df = load_cached()
    if df is None:
        print("  No local cache found. Pulling from Supabase...")
        df = dump()

df = df[df["actual_home_score"].notna()].copy()
df = df[df["season"] != 2021].copy()
print(f"  Loaded {len(df)} rows in {time.time()-t0:.0f}s")

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
_quality_cols = ["home_adj_em", "away_adj_em", "home_ppg", "away_ppg", "market_spread_home", "market_ou_total"]
_qcols = [c for c in _quality_cols if c in df.columns]
_qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em", "away_adj_em", "market_spread_home", "market_ou_total"] else True) for c in _qcols})
_row_q = _qmat.mean(axis=1)
_keep = _row_q >= 0.8
if "referee_1" in df.columns:
    _has_ref = df["referee_1"].notna() & (df["referee_1"] != "")
    _keep = _keep & _has_ref
df = df.loc[_keep].reset_index(drop=True)

for col in ["actual_home_score", "actual_away_score", "home_adj_em", "away_adj_em",
            "home_adj_oe", "away_adj_oe", "home_adj_de", "away_adj_de",
            "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
            "home_tempo", "away_tempo", "home_rank", "away_rank", "season",
            "home_record_wins", "away_record_wins", "home_record_losses", "away_record_losses"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "home_record_wins" in df.columns and "home_wins" not in df.columns:
    df["home_wins"] = df["home_record_wins"]
if "away_record_wins" in df.columns and "away_wins" not in df.columns:
    df["away_wins"] = df["away_record_wins"]
if "home_record_losses" in df.columns and "home_losses" not in df.columns:
    df["home_losses"] = df["home_record_losses"]
if "away_record_losses" in df.columns and "away_losses" not in df.columns:
    df["away_losses"] = df["away_record_losses"]

current_year = datetime.utcnow().year
df["season_weight"] = df["season"].apply(
    lambda s: 1.0 if (current_year - s) <= 0 else 0.9 if (current_year - s) == 1 else
    0.75 if (current_year - s) == 2 else 0.6 if (current_year - s) == 3 else 0.5)

print("  Heuristic backfill...")
df = _ncaa_backfill_heuristic(df)

print("  Computing crowd_shock + h2h/conf/form...")
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)

try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except ImportError:
    pass

try:
    import json as _json
    with open("referee_profiles.json") as _rf:
        ncaa_build_features._ref_profiles = _json.load(_rf)
except FileNotFoundError:
    pass

df = df.dropna(subset=["actual_home_score", "actual_away_score"])
y_total = df["actual_home_score"].values + df["actual_away_score"].values
weights = df["season_weight"].values
n = len(df)

# Build full feature matrix, then select O/U features
print("  Building features...")
X_full = ncaa_build_features(df)

# Verify all O/U features exist
missing = [f for f in OU_FEATURES if f not in X_full.columns]
if missing:
    print(f"  ⚠️ Missing O/U features: {missing}")
    OU_FEATURES = [f for f in OU_FEATURES if f in X_full.columns]

X = X_full[OU_FEATURES]
mkt_total = X_full["mkt_total"].values if "mkt_total" in X_full.columns else np.zeros(n)

print(f"  {n} games × {len(OU_FEATURES)} O/U features")

# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# ── CatBoost ──
print(f"\n  CatBoost O/U (d={CAT_DEPTH}, {CAT_ITERS}i, {N_FOLDS}-fold WF)...", end=" ", flush=True)
cat = CatBoostRegressor(n_estimators=CAT_ITERS, depth=CAT_DEPTH, learning_rate=0.10, random_seed=42, verbose=0)
oof_cat = time_series_oof(cat, X_s, y_total, n_splits=N_FOLDS)
valid_cat = ~np.isnan(oof_cat)
mae_cat = mean_absolute_error(y_total[valid_cat], oof_cat[valid_cat])
print(f"MAE: {mae_cat:.3f}")

# ── MLP ──
print(f"  MLP O/U (256-128-64, {N_FOLDS}-fold WF)...", end=" ", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                   learning_rate='adaptive', random_state=42)
oof_mlp = time_series_oof(mlp, X_s, y_total, n_splits=N_FOLDS)
valid_mlp = ~np.isnan(oof_mlp)
mae_mlp = mean_absolute_error(y_total[valid_mlp], oof_mlp[valid_mlp])
print(f"MAE: {mae_mlp:.3f}")

# ── Stack with ElasticNet ──
valid_both = valid_cat & valid_mlp
meta_X = np.column_stack([oof_cat[valid_both], oof_mlp[valid_both]])
meta = ElasticNet(alpha=0.01, l1_ratio=0.5)
meta.fit(meta_X, y_total[valid_both])
oof_stack = meta.predict(meta_X)
mae_stack = mean_absolute_error(y_total[valid_both], oof_stack)
print(f"  Stacked MAE: {mae_stack:.3f} (Cat={meta.coef_[0]:.3f}, MLP={meta.coef_[1]:.3f})")

# Bias correction
bias = float(np.mean(y_total[valid_both] - oof_stack))
print(f"  Bias correction: {bias:+.3f}")

# Walk-forward O/U accuracy
walk_forward_ou_test(y_total, oof_cat, mkt_total)

# ── Compare to v1 (20 features) ──
print(f"\n  Comparison vs v1 (20 features):")
V1_FEATURES = [
    'mkt_total', 'tempo_avg', 'ref_ou_bias', 'starter_experience_diff',
    'opp_fta_rate_diff', 'lineup_stability_diff', 'venue_advantage',
    'eff_vol_diff', 'hca_pts', 'starter_balance_diff', 'lineup_changes_diff',
    'roll_ats_margin_gated', 'to_margin_diff', 'ref_foul_rate', 'matchup_efg',
    'info_gain_diff', 'roll_top3_share_diff', 'assist_rate_diff',
    'mkt_spread_vs_model', 'matchup_orb',
]
v1_cols = [f for f in V1_FEATURES if f in X_full.columns]
X_v1 = X_full[v1_cols]
scaler_v1 = StandardScaler()
X_v1_s = scaler_v1.fit_transform(X_v1)
cat_v1 = CatBoostRegressor(n_estimators=CAT_ITERS, depth=CAT_DEPTH, learning_rate=0.10, random_seed=42, verbose=0)
oof_v1 = time_series_oof(cat_v1, X_v1_s, y_total, n_splits=N_FOLDS)
valid_v1 = ~np.isnan(oof_v1)
mae_v1 = mean_absolute_error(y_total[valid_v1], oof_v1[valid_v1])
print(f"    v1 (20 feat): MAE {mae_v1:.3f}")
print(f"    v2 ({len(OU_FEATURES)} feat): MAE {mae_cat:.3f}")
print(f"    Δ: {mae_v1 - mae_cat:+.3f} ({'BETTER' if mae_cat < mae_v1 else 'WORSE'})")

walk_forward_ou_test(y_total, oof_v1, mkt_total)

# ══════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════

print(f"\n  Refitting CatBoost on full data for SHAP...")
cat.fit(X_s, y_total, sample_weight=weights)
explainer = shap.TreeExplainer(cat)
sample_idx = np.random.RandomState(42).choice(n, min(3000, n), replace=False)
shap_vals = explainer.shap_values(X_s[sample_idx])
mean_shap = np.abs(shap_vals).mean(axis=0)

print(f"\n  O/U SHAP Top 20:")
shap_df = pd.DataFrame({"feature": OU_FEATURES, "shap": mean_shap}).sort_values("shap", ascending=False)
for i, row in shap_df.head(20).iterrows():
    print(f"    {row['feature']:<35s} {row['shap']:.4f}")

# ══════════════════════════════════════════════════════════════
# PACKAGE & UPLOAD
# ══════════════════════════════════════════════════════════════

# Refit MLP and stack on full data
print(f"\n  Refitting MLP + stack on full data...")
mlp.fit(X_s, y_total)
reg = StackedRegressor([cat, mlp], meta)

bundle = {
    "scaler": scaler,
    "reg": reg,
    "explainer": explainer,
    "feature_cols": OU_FEATURES,
    "ou_feature_cols": OU_FEATURES,  # explicit key for the O/U endpoint
    "n_train": n,
    "mae_cv": round(mae_stack, 3),
    "model_type": f"CatBoost_MLP_OU_v2_{len(OU_FEATURES)}feat",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "bias_correction": round(bias, 3),
    "meta_weights": list(meta.coef_.round(4)),
    "target": "total_points",
}

print(f"\n  Bundle: {bundle['model_type']}, MAE={mae_stack:.3f}, {n} games")

if "--upload" in sys.argv:
    print("  Compressing...")
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    compressed = buf.getvalue()
    print(f"  Size: {len(compressed)/1024:.0f} KB")

    b64 = base64.b64encode(compressed).decode('ascii')
    print("  Uploading to Supabase as 'ncaa_ou'...")
    resp = requests.post(
        f'{SUPABASE_URL}/rest/v1/model_store',
        headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
                 'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates'},
        json={'name': 'ncaa_ou', 'data': b64,
              'metadata': {'trained_at': bundle['trained_at'], 'mae_cv': bundle['mae_cv'],
                           'n_train': n, 'model_type': bundle['model_type'],
                           'size_bytes': len(compressed), 'n_features': len(OU_FEATURES)},
              'updated_at': datetime.now(timezone.utc).isoformat()},
        timeout=300)
    if resp.ok:
        print(f'  ✅ Upload successful ({len(compressed)/1024:.0f} KB)')
    else:
        print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')
else:
    print("  (use --upload to push to Supabase)")

# Save locally
joblib.dump(bundle, "ncaa_ou_model_v2.pkl", compress=3)
print(f"\n  Done. n_train={n}, MAE={mae_stack:.3f}, features={len(OU_FEATURES)}")
