#!/usr/bin/env python3
"""
ncaa_retrain_ou_v5.py — NCAA O/U v5 Asymmetric Agreement System
================================================================
Changes from v4:
  1. New O/U-specific features: tempo interactions, sum features, season phase
  2. Asymmetric tiers: OVER drops classifier requirement, UNDER keeps triple agreement
  3. UNDER 3u tightened (res<-2.0 vs -1.5)
  4. OVER 2u raised (res>3.5, ats>8 — 59.9%, +14.4% ROI)
  5. Recent weights (1/1/.5/.25/.1) validated for O/U

UNDER tiers (triple agreement: residual + classifier + ATS):
  1u: res ≤ -0.5, cls ≥ 54%, ats ≤ -1  (60.9%, +16.3% ROI, 745 picks)
  2u: res ≤ -1.0, cls ≥ 54%, ats ≤ -4  (63.5%, +21.3% ROI, 362 picks)
  3u: res ≤ -2.0, cls ≥ 58%, ats ≤ -5  (66.1%, +26.2% ROI, 62 picks)

OVER tiers (asymmetric: residual + ATS, no classifier for 1u/2u):
  1u: res ≥ 3.0, ats ≥ 3 (no cls)     (56.3%, +7.6% ROI, 1269 picks)
  2u: res ≥ 3.5, ats ≥ 8 (no cls)     (59.9%, +14.4% ROI, 277 picks)
  3u: res ≥ 2.0, cls ≤ 42%, ats ≥ 6   (58.1%, +10.8% ROI, 186 picks)

Architecture (unchanged):
  3 residual models (Lasso + LGBM + CatBoost) — tight features
  2 classifier models (LogReg + LGBM) — tight features
  ATS score models (Lasso + LGBM) — 44 ATS features

Usage:
    python3 ncaa_retrain_ou_v5.py              # Train + evaluate
    python3 ncaa_retrain_ou_v5.py --upload     # Train + upload to Supabase
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
from collections import defaultdict

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
SEED = 42
N_FOLDS = 30

# ATS features (inlined)
ATS_FEATURES = [
    "mkt_spread","player_rating_diff","ref_home_whistle","weakest_starter_diff",
    "crowd_shock_diff","lineup_stability_diff","lineup_changes_diff","adj_oe_diff",
    "rolling_hca","blowout_asym_diff","threepct_diff","pit_sos_diff","orb_pct_diff",
    "blocks_diff","drb_pct_diff","opp_to_rate_diff","elo_diff","is_early",
    "spread_regime","assist_rate_diff","opp_ppg_diff","opp_suppression_diff",
    "roll_ats_margin_gated","has_ats_data","tempo_avg","form_x_familiarity",
    "to_conversion_diff","conf_strength_diff","roll_rotation_diff","roll_dominance_diff",
    "importance","twopt_diff","roll_ats_diff_gated","overreaction_diff",
    "three_rate_diff","ppp_diff","to_margin_diff","momentum_halflife_diff",
    "starter_experience_diff","style_familiarity","fatigue_x_quality","ato_diff",
    "consistency_x_spread","travel_advantage",
]
ATS_MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}

# ── ASYMMETRIC TIERS (validated via walk-forward sweep, Apr 2026) ──
# UNDER: triple agreement (residual + classifier + ATS)
UNDER_TIERS = {
    1: {"res_avg": -0.5, "cls_avg": 0.54, "ats_edge": -1},
    2: {"res_avg": -1.0, "cls_avg": 0.54, "ats_edge": -4},
    3: {"res_avg": -2.0, "cls_avg": 0.58, "ats_edge": -5},   # tighter than v4 (was -1.5)
}
# OVER: residual + ATS only (no classifier for 1u/2u), classifier for 3u premium
OVER_TIERS = {
    1: {"res_avg": 3.0, "ats_edge": 3},                         # no classifier
    2: {"res_avg": 3.5, "ats_edge": 8},                         # no classifier, raised thresholds
    3: {"res_avg": 2.0, "cls_avg": 0.42, "ats_edge": 6},        # premium with classifier
}


def compute_rolling_hca_col(df, window=20):
    df = df.sort_values("game_date").reset_index(drop=True)
    margins = df["actual_home_score"].values - df["actual_away_score"].values
    h_ids = df["home_team_id"].astype(str).values
    a_ids = df["away_team_id"].astype(str).values
    home_margins = defaultdict(list)
    away_margins = defaultdict(list)
    rolling_hca = np.full(len(df), 6.6)
    for i in range(len(df)):
        hid, aid = h_ids[i], a_ids[i]
        h_hist = home_margins[hid][-window:]
        a_hist = away_margins[hid][-window:]
        if len(h_hist) >= 5 and len(a_hist) >= 5:
            rolling_hca[i] = float(np.mean(h_hist) - np.mean(a_hist)) / 2
        home_margins[hid].append(margins[i])
        away_margins[aid].append(-margins[i])
    df["rolling_hca"] = rolling_hca
    print(f"  Rolling HCA: mean={df['rolling_hca'].mean():+.2f}, coverage={(rolling_hca != 6.6).sum() / len(df) * 100:.1f}%")
    return df


def add_ou_features(X_full, df):
    """Add O/U-specific features (sums + interactions + context)."""
    n_added = 0

    # ── Tempo interactions (where pace-up/pace-down edge lives) ──
    h_tempo = pd.to_numeric(df.get("home_tempo", pd.Series(dtype=float)), errors="coerce").fillna(68).values
    a_tempo = pd.to_numeric(df.get("away_tempo", pd.Series(dtype=float)), errors="coerce").fillna(68).values
    tempo_avg = (h_tempo + a_tempo) / 2.0

    X_full["tempo_min"] = np.minimum(h_tempo, a_tempo)           # pace-down drag
    X_full["tempo_max"] = np.maximum(h_tempo, a_tempo)           # pace-up pull
    X_full["tempo_product"] = h_tempo * a_tempo / (68.0 ** 2)    # amplification
    X_full["pace_mismatch"] = np.abs(h_tempo - a_tempo)          # disagreement
    n_added += 4

    # ── Efficiency sums (total scoring power, not relative) ──
    h_oe = pd.to_numeric(df.get("home_adj_oe", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    a_oe = pd.to_numeric(df.get("away_adj_oe", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    h_de = pd.to_numeric(df.get("home_adj_de", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    a_de = pd.to_numeric(df.get("away_adj_de", pd.Series(dtype=float)), errors="coerce").fillna(100).values

    X_full["oe_sum"] = h_oe + a_oe              # combined offensive power
    X_full["de_sum"] = h_de + a_de              # combined defensive weakness (high = weak D)
    X_full["efficiency_total"] = (h_oe - h_de) + (a_oe - a_de)  # net efficiency
    X_full["pace_x_oe_sum"] = tempo_avg * (h_oe + a_oe) / 200.0  # fast + good offense
    X_full["pace_x_de_sum"] = tempo_avg * (h_de + a_de) / 200.0  # fast + bad defense
    n_added += 5

    # ── Scoring sums ──
    h_ppg = pd.to_numeric(df.get("home_ppg", pd.Series(dtype=float)), errors="coerce").fillna(74).values
    a_ppg = pd.to_numeric(df.get("away_ppg", pd.Series(dtype=float)), errors="coerce").fillna(74).values
    h_opp = pd.to_numeric(df.get("home_opp_ppg", pd.Series(dtype=float)), errors="coerce").fillna(72).values
    a_opp = pd.to_numeric(df.get("away_opp_ppg", pd.Series(dtype=float)), errors="coerce").fillna(72).values

    X_full["ppg_sum"] = h_ppg + a_ppg
    X_full["opp_ppg_sum"] = h_opp + a_opp
    X_full["scoring_environment"] = (h_ppg + a_ppg + h_opp + a_opp) / 4  # overall scoring context
    n_added += 3

    # ── Shooting sums ──
    h_3pct = pd.to_numeric(df.get("home_threepct", pd.Series(dtype=float)), errors="coerce").fillna(0.34).values
    a_3pct = pd.to_numeric(df.get("away_threepct", pd.Series(dtype=float)), errors="coerce").fillna(0.34).values
    h_fgpct = pd.to_numeric(df.get("home_fgpct", pd.Series(dtype=float)), errors="coerce").fillna(0.44).values
    a_fgpct = pd.to_numeric(df.get("away_fgpct", pd.Series(dtype=float)), errors="coerce").fillna(0.44).values

    X_full["threepct_sum"] = h_3pct + a_3pct       # combined 3PT shooting
    X_full["fgpct_sum"] = h_fgpct + a_fgpct         # combined FG%
    n_added += 2

    # ── Rebounding + turnover sums ──
    h_orb = pd.to_numeric(df.get("home_orb_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.28).values
    a_orb = pd.to_numeric(df.get("away_orb_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.28).values

    X_full["orb_pct_sum"] = h_orb + a_orb  # more ORBs = more possessions = higher total
    n_added += 1

    # ── Season phase (tighter defense late season → unders) ──
    gd = pd.to_datetime(df.get("game_date", "2026-01-01"), errors="coerce")
    # Days until March 15 (approx tournament start)
    march_15 = gd.apply(lambda d: pd.Timestamp(d.year, 3, 15) if pd.notna(d) else pd.NaT)
    days_to_march = (march_15 - gd).dt.days.fillna(0).values
    X_full["days_to_march"] = np.clip(days_to_march, -30, 150)  # negative = post-tournament

    # Is it tournament time? (March 10+)
    is_tourney_time = ((gd.dt.month == 3) & (gd.dt.day >= 10)) | (gd.dt.month == 4)
    X_full["is_tourney_time"] = is_tourney_time.astype(int).values
    n_added += 2

    print(f"  Added {n_added} O/U-specific features")
    return X_full


def ou_eval(res_avg, cls_avg, ats_edge, y_under, label=""):
    """Evaluate O/U with asymmetric tiers."""
    yo = 1 - y_under
    valid = ~np.isnan(res_avg) & ~np.isnan(cls_avg) & ~np.isnan(ats_edge)

    print(f"\n  {label} ({valid.sum()} games):")
    print(f"  {'Dir':>6s} {'Tier':>5s} {'Criteria':>40s} {'Picks':>7s} {'Acc':>6s} {'ROI':>7s}")
    print(f"  {'-' * 75}")

    for tier, t in UNDER_TIERS.items():
        mask = valid & (res_avg <= t["res_avg"]) & (cls_avg >= t["cls_avg"]) & (ats_edge <= t["ats_edge"])
        nn = mask.sum()
        if nn < 10:
            continue
        acc = y_under[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        crit = f"res≤{t['res_avg']:+.1f}, cls≥{t['cls_avg']:.0%}, ats≤{t['ats_edge']:+d}"
        print(f"  {'UNDER':>6s} {tier:>5d}u {crit:>40s} {nn:>7d} {acc:>5.1%} {roi:>+6.1f}%")

    for tier, t in OVER_TIERS.items():
        if "cls_avg" in t:
            mask = valid & (res_avg >= t["res_avg"]) & (cls_avg <= t["cls_avg"]) & (ats_edge >= t["ats_edge"])
            crit = f"res≥{t['res_avg']:+.1f}, cls≤{t['cls_avg']:.0%}, ats≥{t['ats_edge']:+d}"
        else:
            mask = valid & (res_avg >= t["res_avg"]) & (ats_edge >= t["ats_edge"])
            crit = f"res≥{t['res_avg']:+.1f}, ats≥{t['ats_edge']:+d} (no cls)"
        nn = mask.sum()
        if nn < 10:
            continue
        acc = yo[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"  {'OVER':>6s} {tier:>5d}u {crit:>40s} {nn:>7d} {acc:>5.1%} {roi:>+6.1f}%")


# ══════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════

print("=" * 70)
print("  NCAA O/U v5 — Asymmetric Agreement System")
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

for col in ["market_spread_home", "market_ou_total", "espn_spread", "espn_over_under",
            "dk_spread_close", "dk_total_close", "odds_api_spread_close", "odds_api_total_close",
            "actual_home_score", "actual_away_score", "season"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["season"] = df["season"].fillna(0).astype(int)
df = df[~df["season"].isin([2020, 2021])].copy()
df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
m = df["game_date_dt"].dt.month
df = df[((m >= 11) | (m <= 4)) & ~((m == 11) & (df["game_date_dt"].dt.day < 10))].copy()

# Cascade O/U + spread backfill
for sc, tc in [("espn_over_under", "market_ou_total"), ("dk_total_close", "market_ou_total"),
               ("odds_api_total_close", "market_ou_total")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce")
        cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0:
            df.loc[fill, tc] = src[fill]
for sc, tc in [("espn_spread", "market_spread_home"), ("dk_spread_close", "market_spread_home"),
               ("odds_api_spread_close", "market_spread_home")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce")
        cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0:
            df.loc[fill, tc] = src[fill]

mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
n_before = len(df)
df = df[mkt_ou.notna() & (mkt_ou > 50) & (mkt_ou < 250)].copy().reset_index(drop=True)
print(f"  Filter (real O/U): {n_before} → {len(df)}")

if "referee_1" in df.columns:
    n_before = len(df)
    df = df[df["referee_1"].notna() & (df["referee_1"] != "") & (df["referee_1"] != "None")].copy().reset_index(drop=True)
    print(f"  Filter (refs): {n_before} → {len(df)}")

for s, d in [("home_record_wins", "home_wins"), ("away_record_wins", "away_wins"),
             ("home_record_losses", "home_losses"), ("away_record_losses", "away_losses")]:
    if s in df.columns and d not in df.columns:
        df[d] = df[s]

seasons = pd.to_numeric(df["season"], errors="coerce").values
# Recent weights (validated for O/U, Apr 2026 sweep)
weights = np.array([{2026: 1.0, 2025: 1.0, 2024: 0.5, 2023: 0.25}.get(s, 0.1) for s in seasons])

print("  Heuristic backfill + features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except Exception:
    pass
try:
    import json
    with open("referee_profiles.json") as f:
        ncaa_build_features._ref_profiles = json.load(f)
except Exception:
    pass

df = df.dropna(subset=["actual_home_score", "actual_away_score"])
df = compute_rolling_hca_col(df)
df["travel_advantage"] = 0

# Build standard features
X_full = ncaa_build_features(df)
X_full["rolling_hca"] = df["rolling_hca"].values
X_full["travel_advantage"] = 0
raw_em = df["home_adj_em"].fillna(0).values - df["away_adj_em"].fillna(0).values
nm = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool).values
X_full["neutral_em_diff"] = raw_em - np.where(nm, 0, df["rolling_hca"].values)

# ── Add O/U-specific features ──
X_full = add_ou_features(X_full, df)

y_home = df["actual_home_score"].values
y_away = df["actual_away_score"].values
y_total = y_home + y_away
mkt_total = pd.to_numeric(df["market_ou_total"], errors="coerce").fillna(0).values
push = y_total == mkt_total
y_residual = y_total - mkt_total
y_under = (y_total < mkt_total).astype(float)

all_feats = [c for c in X_full.columns if X_full[c].notna().mean() > 0.1]
ats_feats = [f for f in ATS_FEATURES if f in X_full.columns]

complete = X_full[all_feats].isna().sum(axis=1) == 0
keep = complete & ~push
idx = np.where(keep)[0]

X = X_full.iloc[idx][all_feats].reset_index(drop=True)
X_ats = X_full.iloc[idx][ats_feats].reset_index(drop=True)
yr = y_residual[idx]
yu = y_under[idx]
yh = y_home[idx]
ya = y_away[idx]
yt = y_total[idx]
mt = mkt_total[idx]
w = weights[idx]
s_arr = seasons[idx]
n = len(idx)

print(f"  {n} clean games (real O/U, refs, complete features, non-push)")
print(f"  {time.time() - t0:.0f}s")

# ══════════════════════════════════════════════════════════
#  FEATURE SELECTION (tight)
# ══════════════════════════════════════════════════════════
print(f"\n  Feature selection (tight)...")
X_s = StandardScaler().fit_transform(X)

# Residual: Lasso α=0.1
lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_s, yr, sample_weight=w)
res_feats = [f for f, c in zip(all_feats, lasso.coef_) if abs(c) > 0.001]

# Classifier: LogReg C=0.01 (L1)
lr = LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=5000, random_state=SEED)
lr.fit(X_s, yu, sample_weight=w)
cls_feats = [f for f, c in zip(all_feats, lr.coef_[0]) if abs(c) > 0.001]

print(f"  Residual: {len(res_feats)} features")
print(f"  Classifier: {len(cls_feats)} features")
print(f"  ATS: {len(ats_feats)} features")

# Show new features that were selected
new_ou_feats = ["tempo_min", "tempo_max", "tempo_product", "pace_mismatch",
                "oe_sum", "de_sum", "efficiency_total", "pace_x_oe_sum", "pace_x_de_sum",
                "ppg_sum", "opp_ppg_sum", "scoring_environment",
                "threepct_sum", "fgpct_sum", "orb_pct_sum",
                "days_to_march", "is_tourney_time"]

res_new = [f for f in new_ou_feats if f in res_feats]
cls_new = [f for f in new_ou_feats if f in cls_feats]
print(f"\n  New O/U features selected:")
print(f"    Residual: {res_new if res_new else 'none'}")
print(f"    Classifier: {cls_new if cls_new else 'none'}")

# Show Lasso coefficients for new features
print(f"\n  Lasso coefficients (new features):")
for f, c in sorted(zip(all_feats, lasso.coef_), key=lambda x: -abs(x[1])):
    if f in new_ou_feats and abs(c) > 0.001:
        print(f"    {f:>25s}: {c:+.4f}")

# ══════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════
print(f"\n  Walk-forward ({N_FOLDS} folds)...")
fs = n // (N_FOLDS + 1)
min_t = fs * 2

oof_res = np.full((3, n), np.nan)
oof_cls = np.full((2, n), np.nan)
oof_ats = np.full(n, np.nan)

for fold in range(N_FOLDS):
    ts = min_t + fold * fs
    te = min(ts + fs, n)
    if ts >= n:
        break

    # 3 residual models
    Xr = StandardScaler().fit_transform(X[res_feats])
    r1 = Lasso(alpha=0.1, max_iter=5000)
    r1.fit(Xr[:ts], yr[:ts])
    oof_res[0, ts:te] = r1.predict(Xr[ts:te])

    r2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                        subsample=0.8, verbose=-1, random_state=SEED)
    r2.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts])
    oof_res[1, ts:te] = r2.predict(Xr[ts:te])

    r3 = CatBoostRegressor(n_estimators=300, depth=3, learning_rate=0.05,
                            random_seed=SEED, verbose=0)
    r3.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts])
    oof_res[2, ts:te] = r3.predict(Xr[ts:te])

    # 2 classifier models
    Xc = StandardScaler().fit_transform(X[cls_feats])
    c1 = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
    c1.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts])
    oof_cls[0, ts:te] = c1.predict_proba(Xc[ts:te])[:, 1]

    c2 = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                          subsample=0.8, verbose=-1, random_state=SEED)
    c2.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts])
    oof_cls[1, ts:te] = c2.predict_proba(Xc[ts:te])[:, 1]

    # ATS home + away score models
    Xa = StandardScaler().fit_transform(X_ats)
    hp = []
    ap = []
    for name, builder in ATS_MODELS.items():
        mh = builder()
        try:
            mh.fit(Xa[:ts], yh[:ts], sample_weight=w[:ts])
        except TypeError:
            mh.fit(Xa[:ts], yh[:ts])
        hp.append(mh.predict(Xa[ts:te]))
        ma = builder()
        try:
            ma.fit(Xa[:ts], ya[:ts], sample_weight=w[:ts])
        except TypeError:
            ma.fit(Xa[:ts], ya[:ts])
        ap.append(ma.predict(Xa[ts:te]))
    oof_ats[ts:te] = np.mean(hp, axis=0) + np.mean(ap, axis=0)

    if (fold + 1) % 10 == 0:
        print(f"    Fold {fold + 1}/{N_FOLDS}")

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

# ── Compare v4 vs v5 tiers ──
V4_UNDER = {
    1: {"res_avg": -0.5, "cls_avg": 0.54, "ats_edge": -1},
    2: {"res_avg": -1.0, "cls_avg": 0.54, "ats_edge": -4},
    3: {"res_avg": -1.5, "cls_avg": 0.58, "ats_edge": -5},
}
V4_OVER = {
    1: {"res_avg": 2.0, "cls_avg": 0.42, "ats_edge": 3},
    2: {"res_avg": 2.0, "cls_avg": 0.42, "ats_edge": 6},
}

# v4 eval
yo = 1 - yu
print(f"\n{'=' * 75}")
print(f"  V4 TIERS (old, for comparison)")
print(f"{'=' * 75}")
print(f"  {'Dir':>6s} {'Tier':>5s} {'Picks':>7s} {'Acc':>7s} {'ROI':>8s}")
print(f"  {'-' * 40}")
for tier, t in V4_UNDER.items():
    mask = valid & (res_avg <= t["res_avg"]) & (cls_avg >= t["cls_avg"]) & (ats_edge <= t["ats_edge"])
    nn = mask.sum()
    if nn < 5: continue
    acc = yu[mask].mean()
    roi = (acc * 1.909 - 1) * 100
    print(f"  {'UNDER':>6s} {tier:>5d}u {nn:>7d} {acc:>6.1%} {roi:>+7.1f}%")
for tier, t in V4_OVER.items():
    mask = valid & (res_avg >= t["res_avg"]) & (cls_avg <= t["cls_avg"]) & (ats_edge >= t["ats_edge"])
    nn = mask.sum()
    if nn < 5: continue
    acc = yo[mask].mean()
    roi = (acc * 1.909 - 1) * 100
    print(f"  {'OVER':>6s} {tier:>5d}u {nn:>7d} {acc:>6.1%} {roi:>+7.1f}%")

# v5 eval
ou_eval(res_avg, cls_avg, ats_edge, yu, "V5 TIERS (new)")

# ── Per-season v5 ──
print(f"\n{'=' * 75}")
print(f"  V5 PER-SEASON BREAKDOWN")
print(f"{'=' * 75}")
for s in sorted(set(s_arr[valid])):
    s_mask = valid & (s_arr == s)
    if s_mask.sum() < 50:
        continue
    print(f"\n  Season {s} ({s_mask.sum()} games):")
    for tier, t in UNDER_TIERS.items():
        mask = s_mask & (res_avg <= t["res_avg"]) & (cls_avg >= t["cls_avg"]) & (ats_edge <= t["ats_edge"])
        nn = mask.sum()
        if nn < 5: continue
        acc = yu[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"    UNDER {tier}u: {acc:.1%} on {nn} picks (ROI {roi:+.1f}%)")
    for tier, t in OVER_TIERS.items():
        if "cls_avg" in t:
            mask = s_mask & (res_avg >= t["res_avg"]) & (cls_avg <= t["cls_avg"]) & (ats_edge >= t["ats_edge"])
        else:
            mask = s_mask & (res_avg >= t["res_avg"]) & (ats_edge >= t["ats_edge"])
        nn = mask.sum()
        if nn < 5: continue
        acc = yo[mask].mean()
        roi = (acc * 1.909 - 1) * 100
        print(f"    OVER {tier}u: {acc:.1%} on {nn} picks (ROI {roi:+.1f}%)")

# ══════════════════════════════════════════════════════════
#  PRODUCTION TRAINING
# ══════════════════════════════════════════════════════════
if upload:
    print(f"\n{'=' * 70}")
    print(f"  PRODUCTION TRAINING")
    print(f"{'=' * 70}")

    res_scaler = StandardScaler().fit(X[res_feats])
    cls_scaler = StandardScaler().fit(X[cls_feats])
    ats_scaler = StandardScaler().fit(X_ats)

    Xr = res_scaler.transform(X[res_feats])
    Xc = cls_scaler.transform(X[cls_feats])
    Xa = ats_scaler.transform(X_ats)

    print("  Training 3 residual models...")
    res_models = []
    r1 = Lasso(alpha=0.1, max_iter=5000); r1.fit(Xr, yr); res_models.append(r1)
    r2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    r2.fit(Xr, yr, sample_weight=w); res_models.append(r2)
    r3 = CatBoostRegressor(n_estimators=300, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)
    r3.fit(Xr, yr, sample_weight=w); res_models.append(r3)
    res_pred = np.mean([m.predict(Xr) for m in res_models], axis=0)
    print(f"    Residual ensemble MAE: {mean_absolute_error(yr, res_pred):.3f}")

    print("  Training 2 classifier models...")
    cls_models = []
    c1 = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
    c1.fit(Xc, yu, sample_weight=w); cls_models.append(c1)
    c2 = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    c2.fit(Xc, yu, sample_weight=w); cls_models.append(c2)
    cls_pred = np.mean([m.predict_proba(Xc)[:, 1] for m in cls_models], axis=0)
    print(f"    Classifier avg P(under): {cls_pred.mean():.3f}")

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

    bundle = {
        "res_scaler": res_scaler, "cls_scaler": cls_scaler, "ats_scaler": ats_scaler,
        "res_models": res_models, "cls_models": cls_models,
        "ats_home_models": ats_home_models, "ats_away_models": ats_away_models,
        "res_feature_cols": res_feats, "cls_feature_cols": cls_feats, "ats_feature_cols": ats_feats,
        "under_tiers": UNDER_TIERS, "over_tiers": OVER_TIERS,
        "n_train": n, "model_type": "OU_v5_asymmetric",
        "architecture": "3res(Lasso+LGBM+Cat) + 2cls(LR+LGBM) + ATS(home+away)",
        "res_mae_cv": round(res_mae, 4), "ats_total_mae_cv": round(ats_mae, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_counts": {"residual": len(res_feats), "classifier": len(cls_feats), "ats": len(ats_feats)},
    }

    local_path = "ncaa_ou_v5.pkl"
    joblib.dump(bundle, local_path, compress=3)
    size_kb = os.path.getsize(local_path) / 1024
    print(f"\n  Saved: {local_path} ({size_kb:.0f} KB)")

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
                           "feature_counts": bundle["feature_counts"]},
              "updated_at": datetime.now(timezone.utc).isoformat()},
        timeout=300)
    if resp.ok:
        print(f"  ✅ Upload successful ({len(buf.getvalue()) // 1024} KB)")
    else:
        print(f"  ❌ Failed: {resp.status_code} {resp.text[:300]}")

print(f"\n{'=' * 70}")
print(f"  NCAA O/U v5 COMPLETE")
print(f"  Features: res={len(res_feats)}, cls={len(cls_feats)}, ats={len(ats_feats)}")
print(f"  Games: {n}")
print(f"  Residual MAE: {res_mae:.3f}")
print(f"{'=' * 70}")
