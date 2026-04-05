#!/usr/bin/env python3
"""
Quick test: OVER 2u threshold candidates on v5 walk-forward data.
Reuses exact v5 pipeline, just tests specific threshold combos.
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor
from collections import defaultdict

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42; N_FOLDS = 30

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


def compute_rolling_hca_col(df, window=20):
    df = df.sort_values("game_date").reset_index(drop=True)
    margins = df["actual_home_score"].values - df["actual_away_score"].values
    h_ids = df["home_team_id"].astype(str).values
    a_ids = df["away_team_id"].astype(str).values
    home_margins = defaultdict(list); away_margins = defaultdict(list)
    rolling_hca = np.full(len(df), 6.6)
    for i in range(len(df)):
        hid, aid = h_ids[i], a_ids[i]
        h_hist = home_margins[hid][-window:]; a_hist = away_margins[hid][-window:]
        if len(h_hist) >= 5 and len(a_hist) >= 5:
            rolling_hca[i] = float(np.mean(h_hist) - np.mean(a_hist)) / 2
        home_margins[hid].append(margins[i]); away_margins[aid].append(-margins[i])
    df["rolling_hca"] = rolling_hca
    return df


def add_ou_features(X_full, df):
    h_tempo = pd.to_numeric(df.get("home_tempo", pd.Series(dtype=float)), errors="coerce").fillna(68).values
    a_tempo = pd.to_numeric(df.get("away_tempo", pd.Series(dtype=float)), errors="coerce").fillna(68).values
    h_oe = pd.to_numeric(df.get("home_adj_oe", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    a_oe = pd.to_numeric(df.get("away_adj_oe", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    h_de = pd.to_numeric(df.get("home_adj_de", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    a_de = pd.to_numeric(df.get("away_adj_de", pd.Series(dtype=float)), errors="coerce").fillna(100).values
    tempo_avg = (h_tempo + a_tempo) / 2.0
    h_ppg = pd.to_numeric(df.get("home_ppg", pd.Series(dtype=float)), errors="coerce").fillna(74).values
    a_ppg = pd.to_numeric(df.get("away_ppg", pd.Series(dtype=float)), errors="coerce").fillna(74).values
    h_opp = pd.to_numeric(df.get("home_opp_ppg", pd.Series(dtype=float)), errors="coerce").fillna(72).values
    a_opp = pd.to_numeric(df.get("away_opp_ppg", pd.Series(dtype=float)), errors="coerce").fillna(72).values
    h_3pct = pd.to_numeric(df.get("home_threepct", pd.Series(dtype=float)), errors="coerce").fillna(0.34).values
    a_3pct = pd.to_numeric(df.get("away_threepct", pd.Series(dtype=float)), errors="coerce").fillna(0.34).values
    h_fgpct = pd.to_numeric(df.get("home_fgpct", pd.Series(dtype=float)), errors="coerce").fillna(0.44).values
    a_fgpct = pd.to_numeric(df.get("away_fgpct", pd.Series(dtype=float)), errors="coerce").fillna(0.44).values
    h_orb = pd.to_numeric(df.get("home_orb_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.28).values
    a_orb = pd.to_numeric(df.get("away_orb_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.28).values
    gd = pd.to_datetime(df.get("game_date", "2026-01-01"), errors="coerce")

    X_full["tempo_min"] = np.minimum(h_tempo, a_tempo)
    X_full["tempo_max"] = np.maximum(h_tempo, a_tempo)
    X_full["tempo_product"] = h_tempo * a_tempo / (68.0 ** 2)
    X_full["pace_mismatch"] = np.abs(h_tempo - a_tempo)
    X_full["oe_sum"] = h_oe + a_oe
    X_full["de_sum"] = h_de + a_de
    X_full["efficiency_total"] = (h_oe - h_de) + (a_oe - a_de)
    X_full["pace_x_oe_sum"] = tempo_avg * (h_oe + a_oe) / 200.0
    X_full["pace_x_de_sum"] = tempo_avg * (h_de + a_de) / 200.0
    X_full["ppg_sum"] = h_ppg + a_ppg
    X_full["opp_ppg_sum"] = h_opp + a_opp
    X_full["scoring_environment"] = (h_ppg + a_ppg + h_opp + a_opp) / 4
    X_full["threepct_sum"] = h_3pct + a_3pct
    X_full["fgpct_sum"] = h_fgpct + a_fgpct
    X_full["orb_pct_sum"] = h_orb + a_orb
    march_15 = gd.apply(lambda d: pd.Timestamp(d.year, 3, 15) if pd.notna(d) else pd.NaT)
    X_full["days_to_march"] = np.clip((march_15 - gd).dt.days.fillna(0).values, -30, 150)
    X_full["is_tourney_time"] = (((gd.dt.month == 3) & (gd.dt.day >= 10)) | (gd.dt.month == 4)).astype(int).values
    return X_full


# ══════════════════════════════════════════════════════════
print("=" * 70)
print("  OVER 2u THRESHOLD TEST (v5 walk-forward)")
print("=" * 70)

df = load_cached()
if df is None: print("No cache"); sys.exit(1)
df = df[df["actual_home_score"].notna()].copy()
for col in ["market_spread_home","market_ou_total","espn_spread","espn_over_under",
            "dk_spread_close","dk_total_close","odds_api_spread_close","odds_api_total_close",
            "actual_home_score","actual_away_score","season"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
df["season"] = df["season"].fillna(0).astype(int)
df = df[~df["season"].isin([2020,2021])].copy()
gdt = pd.to_datetime(df["game_date"], errors="coerce")
m = gdt.dt.month
df = df[((m >= 11) | (m <= 4)) & ~((m == 11) & (gdt.dt.day < 10))].copy()
for sc, tc in [("espn_over_under","market_ou_total"),("dk_total_close","market_ou_total"),("odds_api_total_close","market_ou_total")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce"); cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0: df.loc[fill, tc] = src[fill]
for sc, tc in [("espn_spread","market_spread_home"),("dk_spread_close","market_spread_home"),("odds_api_spread_close","market_spread_home")]:
    if sc in df.columns:
        src = pd.to_numeric(df[sc], errors="coerce"); cur = pd.to_numeric(df.get(tc, pd.Series(dtype=float)), errors="coerce")
        fill = (cur.isna() | (cur == 0)) & src.notna() & (src != 0)
        if fill.sum() > 0: df.loc[fill, tc] = src[fill]

mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
df = df[mkt_ou.notna() & (mkt_ou > 50) & (mkt_ou < 250)].copy().reset_index(drop=True)
if "referee_1" in df.columns:
    df = df[df["referee_1"].notna() & (df["referee_1"] != "") & (df["referee_1"] != "None")].copy().reset_index(drop=True)
for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),
             ("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
    if s in df.columns and d not in df.columns: df[d] = df[s]

seasons = pd.to_numeric(df["season"], errors="coerce").values
weights = np.array([{2026:1.0,2025:1.0,2024:0.5,2023:0.25}.get(s, 0.1) for s in seasons])

print("  Building features...")
df = _ncaa_backfill_heuristic(df)
df = compute_crowd_shock(df, n_games=5)
df = compute_missing_features(df)
try:
    from training_data_fixes import apply_training_fixes
    df = apply_training_fixes(df)
except Exception: pass
try:
    import json
    with open("referee_profiles.json") as f: ncaa_build_features._ref_profiles = json.load(f)
except Exception: pass

df = df.dropna(subset=["actual_home_score","actual_away_score"])
df = compute_rolling_hca_col(df)
df["travel_advantage"] = 0
X_full = ncaa_build_features(df)
X_full["rolling_hca"] = df["rolling_hca"].values
X_full["travel_advantage"] = 0
raw_em = df["home_adj_em"].fillna(0).values - df["away_adj_em"].fillna(0).values
nm = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool).values
X_full["neutral_em_diff"] = raw_em - np.where(nm, 0, df["rolling_hca"].values)
X_full = add_ou_features(X_full, df)

y_home = df["actual_home_score"].values; y_away = df["actual_away_score"].values
y_total = y_home + y_away
mkt_total = pd.to_numeric(df["market_ou_total"], errors="coerce").fillna(0).values
push = y_total == mkt_total; y_residual = y_total - mkt_total
y_under = (y_total < mkt_total).astype(float)

all_feats = [c for c in X_full.columns if X_full[c].notna().mean() > 0.1]
ats_feats = [f for f in ATS_FEATURES if f in X_full.columns]
complete = X_full[all_feats].isna().sum(axis=1) == 0
idx = np.where(complete & ~push)[0]

X = X_full.iloc[idx][all_feats].reset_index(drop=True)
X_ats = X_full.iloc[idx][ats_feats].reset_index(drop=True)
yr = y_residual[idx]; yu = y_under[idx]; yh = y_home[idx]; ya = y_away[idx]
yt = y_total[idx]; mt = mkt_total[idx]; w = weights[idx]; s_arr = seasons[idx]
n = len(idx)
print(f"  {n} games")

# Feature selection
X_s = StandardScaler().fit_transform(X)
lasso = Lasso(alpha=0.1, max_iter=5000); lasso.fit(X_s, yr, sample_weight=w)
res_feats = [f for f, c in zip(all_feats, lasso.coef_) if abs(c) > 0.001]
lr = LogisticRegression(C=0.01, penalty='l1', solver='saga', max_iter=5000, random_state=SEED)
lr.fit(X_s, yu, sample_weight=w)
cls_feats = [f for f, c in zip(all_feats, lr.coef_[0]) if abs(c) > 0.001]
print(f"  Residual: {len(res_feats)} | Classifier: {len(cls_feats)} | ATS: {len(ats_feats)}")

# Walk-forward
print(f"\n  Walk-forward ({N_FOLDS} folds)...")
fs = n // (N_FOLDS + 1); min_t = fs * 2
oof_res = np.full((3, n), np.nan); oof_cls = np.full((2, n), np.nan); oof_ats = np.full(n, np.nan)

for fold in range(N_FOLDS):
    ts = min_t + fold * fs; te = min(ts + fs, n)
    if ts >= n: break
    Xr = StandardScaler().fit_transform(X[res_feats])
    r1 = Lasso(alpha=0.1, max_iter=5000); r1.fit(Xr[:ts], yr[:ts]); oof_res[0, ts:te] = r1.predict(Xr[ts:te])
    r2 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    r2.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts]); oof_res[1, ts:te] = r2.predict(Xr[ts:te])
    r3 = CatBoostRegressor(n_estimators=300, depth=3, learning_rate=0.05, random_seed=SEED, verbose=0)
    r3.fit(Xr[:ts], yr[:ts], sample_weight=w[:ts]); oof_res[2, ts:te] = r3.predict(Xr[ts:te])
    Xc = StandardScaler().fit_transform(X[cls_feats])
    c1 = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED)
    c1.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts]); oof_cls[0, ts:te] = c1.predict_proba(Xc[ts:te])[:, 1]
    c2 = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
    c2.fit(Xc[:ts], yu[:ts], sample_weight=w[:ts]); oof_cls[1, ts:te] = c2.predict_proba(Xc[ts:te])[:, 1]
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
res_avg = np.mean(oof_res, axis=0)[valid]
cls_avg = np.mean(oof_cls, axis=0)[valid]
ats_edge = (oof_ats - mt)[valid]
yo = (1 - yu)[valid]
s_v = s_arr[valid]
n_eval = valid.sum()
print(f"  {n_eval} games evaluated")

# ══════════════════════════════════════════════════════════
#  TEST ALL OVER 2u CANDIDATES
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 75}")
print(f"  OVER 2u CANDIDATES (no classifier)")
print(f"{'=' * 75}")

candidates = [
    ("res>3.0, ats>6 (current)", 3.0, 6),
    ("res>3.0, ats>8",           3.0, 8),
    ("res>3.0, ats>10",          3.0, 10),
    ("res>4.0, ats>5",           4.0, 5),
    ("res>4.0, ats>6",           4.0, 6),
    ("res>4.0, ats>8",           4.0, 8),
    ("res>5.0, ats>5",           5.0, 5),
    ("res>5.0, ats>6",           5.0, 6),
    ("res>3.5, ats>7",           3.5, 7),
    ("res>3.5, ats>8",           3.5, 8),
]

print(f"\n  {'Config':>25s} {'Acc':>7s} {'Picks':>7s} {'ROI':>8s} {'Prof Seasons':>13s}")
print(f"  {'-' * 65}")

for label, res_th, ats_th in candidates:
    mask = (res_avg > res_th) & (ats_edge > ats_th)
    nn = mask.sum()
    if nn < 10:
        print(f"  {label:>25s} {'—':>7s} {nn:>7d} {'—':>8s} {'—':>13s}")
        continue
    acc = yo[mask].mean()
    roi = (acc * 1.909 - 1) * 100

    # Count profitable seasons
    prof = 0; total_s = 0
    for s in sorted(set(s_v)):
        sm = mask & (s_v == s)
        if sm.sum() >= 5:
            total_s += 1
            s_acc = yo[sm].mean()
            if s_acc > 0.524:  # breakeven at -110
                prof += 1

    print(f"  {label:>25s} {acc:>6.1%} {nn:>7d} {roi:>+7.1f}% {prof}/{total_s}")

# ══════════════════════════════════════════════════════════
#  PER-SEASON DETAIL FOR TOP CANDIDATES
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 75}")
print(f"  PER-SEASON DETAIL")
print(f"{'=' * 75}")

for label, res_th, ats_th in [("res>3.0,ats>6", 3.0, 6),
                               ("res>3.0,ats>8", 3.0, 8),
                               ("res>4.0,ats>6", 4.0, 6),
                               ("res>3.5,ats>8", 3.5, 8)]:
    mask = (res_avg > res_th) & (ats_edge > ats_th)
    print(f"\n  {label}:")
    for s in sorted(set(s_v)):
        sm = mask & (s_v == s)
        nn = sm.sum()
        if nn < 3: continue
        acc = yo[sm].mean()
        roi = (acc * 1.909 - 1) * 100
        tag = " ✅" if roi > 0 else " ❌"
        print(f"    {s}: {acc:.1%} on {nn} picks (ROI {roi:+.1f}%){tag}")

# ══════════════════════════════════════════════════════════
#  ALSO TEST UNDER 3u REINTRODUCTION
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 75}")
print(f"  UNDER 3u CANDIDATES (should we add it back?)")
print(f"{'=' * 75}")

un3_candidates = [
    ("v4: res<-1.5,cls>58%,ats<-5", -1.5, 0.58, -5),
    ("tighter: res<-2.0,cls>54%,ats<-5", -2.0, 0.54, -5),
    ("tighter: res<-2.0,cls>58%,ats<-5", -2.0, 0.58, -5),
    ("tighter: res<-1.5,cls>54%,ats<-6", -1.5, 0.54, -6),
    ("tighter: res<-2.0,cls>54%,ats<-6", -2.0, 0.54, -6),
]

print(f"\n  {'Config':>40s} {'Acc':>7s} {'Picks':>7s} {'ROI':>8s}")
print(f"  {'-' * 65}")

for label, res_th, cls_th, ats_th in un3_candidates:
    mask = (res_avg < res_th) & (cls_avg > cls_th) & (ats_edge < ats_th)
    nn = mask.sum()
    if nn < 5:
        print(f"  {label:>40s} {'—':>7s} {nn:>7d}")
        continue
    acc = yu[valid][mask].mean()
    roi = (acc * 1.909 - 1) * 100
    print(f"  {label:>40s} {acc:>6.1%} {nn:>7d} {roi:>+7.1f}%")
