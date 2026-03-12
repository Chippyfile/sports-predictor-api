#!/usr/bin/env python3
"""
apply_sweep_winner.py — Patch sports/ncaa.py with sweep-optimized config
═══════════════════════════════════════════════════════════════════════════
Three changes:
  1. Add LightGBM import
  2. ESPN/DraftKings odds fallback in _ncaa_merge_historical
  3. Update train_ncaa to XGB+CAT+LGBM @ e=175, d=7, lr=0.1

Run from sports-predictor-api root:
    python3 apply_sweep_winner.py --dry-run
    python3 apply_sweep_winner.py
"""
import os, sys, argparse

NCAA_PY = "sports/ncaa.py"


def apply_patches(txt):
    changes = []

    # ── 1. Add LightGBM import ──
    if "LGBMRegressor" not in txt:
        old = "try:\n    from catboost import CatBoostRegressor\n    HAS_CAT = True\nexcept ImportError:\n    HAS_CAT = False"
        new = """try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False"""
        if old in txt:
            txt = txt.replace(old, new)
            changes.append("Added LightGBM import (HAS_LGBM)")
        else:
            print("  WARNING: Could not find CatBoost import block")
    else:
        changes.append("LightGBM import already present")

    # ── 2. ESPN/DraftKings odds fallback ──
    marker = '    return combined, weights.values, n_hist'
    if "ESPN odds fallback" not in txt and marker in txt:
        fallback = '''
    # ── ESPN/DraftKings odds fallback (sweep-validated: 4% → 59% market coverage) ──
    if "espn_spread" in combined.columns:
        espn_s = pd.to_numeric(combined["espn_spread"], errors="coerce")
        if "market_spread_home" not in combined.columns:
            combined["market_spread_home"] = np.nan
        mkt_s = pd.to_numeric(combined["market_spread_home"], errors="coerce")
        fill_mask = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
        combined.loc[fill_mask, "market_spread_home"] = espn_s[fill_mask]
        n_filled = int(fill_mask.sum())
        if n_filled > 0:
            print(f"  ESPN odds fallback: {n_filled} spreads filled from DraftKings")

    if "espn_over_under" in combined.columns:
        espn_ou = pd.to_numeric(combined["espn_over_under"], errors="coerce")
        if "market_ou_total" not in combined.columns:
            combined["market_ou_total"] = np.nan
        mkt_ou = pd.to_numeric(combined["market_ou_total"], errors="coerce")
        fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & espn_ou.notna()
        combined.loc[fill_ou, "market_ou_total"] = espn_ou[fill_ou]

'''
        txt = txt.replace(marker, fallback + marker)
        changes.append("Added ESPN/DraftKings odds fallback in _ncaa_merge_historical")
    elif "ESPN odds fallback" in txt:
        changes.append("ESPN fallback already present")

    # ── 3. Update model config: XGB+CAT+RF → XGB+CAT+LGBM, e=175, d=7, lr=0.1 ──

    # 3a. Update estimators 160 → 175
    if "n_estimators=160" in txt:
        txt = txt.replace("n_estimators=160", "n_estimators=175")
        txt = txt.replace("iterations=160", "iterations=175")
        changes.append("Estimators: 160 → 175")

    # 3b. Update XGB depth and lr
    old_xgb = "max_depth=4, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=20"
    new_xgb = "max_depth=7, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=20"
    if old_xgb in txt:
        txt = txt.replace(old_xgb, new_xgb, 1)
        changes.append("XGB: depth 4→7, lr 0.06→0.10")

    # 3c. Update CAT depth and lr
    old_cat = "depth=4, learning_rate=0.06, subsample=0.8, min_data_in_leaf=20"
    new_cat = "depth=7, learning_rate=0.10, subsample=0.8, min_data_in_leaf=20"
    if old_cat in txt:
        txt = txt.replace(old_cat, new_cat, 1)
        changes.append("CAT: depth 4→7, lr 0.06→0.10")

    # 3d. Replace RF with LGBM in ensemble
    old_rf = """            rf_reg = RandomForestRegressor(
                n_estimators=175, max_depth=6,
                min_samples_leaf=15, max_features=0.7,
                random_state=42, n_jobs=1,  # Railway: single core; set -1 for local
            )"""
    new_lgbm = """            # SWEEP WINNER: LGBM replaces RF (faster + better MAE)
            if HAS_LGBM:
                lgbm_reg = LGBMRegressor(
                    n_estimators=175, max_depth=7, learning_rate=0.10,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                    random_state=42, verbosity=-1,
                )"""
    if old_rf in txt:
        txt = txt.replace(old_rf, new_lgbm)
        changes.append("Replaced RF with LGBM")

    # 3e. Update ensemble_parts
    if "ensemble_parts.append('RF')" in txt:
        txt = txt.replace("ensemble_parts.append('RF')", "if HAS_LGBM: ensemble_parts.append('LGBM')")
        changes.append("Ensemble parts: RF → LGBM")

    # 3f. Update reg_models
    if 'reg_models = {"rf": rf_reg}' in txt:
        txt = txt.replace(
            'reg_models = {"rf": rf_reg}',
            'reg_models = {}\n            if HAS_LGBM: reg_models["lgbm"] = lgbm_reg'
        )
        changes.append("reg_models: rf → lgbm")

    # 3g. Update print label
    old_label = "160 est"
    new_label = "e=175 d=7 lr=0.1"
    if old_label in txt:
        txt = txt.replace(old_label, new_label)
        changes.append("Updated training log label")

    return txt, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(NCAA_PY):
        print(f"ERROR: {NCAA_PY} not found. Run from sports-predictor-api root.")
        sys.exit(1)

    print("=" * 70)
    print("  APPLY SWEEP WINNER → sports/ncaa.py")
    print("  XGB+CAT+LGBM, e=175, d=7, lr=0.1, ridge")
    print("=" * 70)

    with open(NCAA_PY, 'r') as f:
        original = f.read()

    txt, changes = apply_patches(original)

    print(f"\n  Changes ({len(changes)}):")
    for c in changes:
        print(f"    ✓ {c}")

    if args.dry_run:
        print(f"\n  DRY RUN — no files written.")
        return

    with open(NCAA_PY + ".backup", 'w') as f:
        f.write(original)
    print(f"\n  Backup: {NCAA_PY}.backup")

    with open(NCAA_PY, 'w') as f:
        f.write(txt)
    print(f"  Patched: {NCAA_PY}")

    print(f"\n  NEXT STEPS:")
    print(f"  1. Review: diff {NCAA_PY}.backup {NCAA_PY}")
    print(f"  2. Retrain locally (update cv_folds=50 in retrain script):")
    print(f"     python3 retrain_and_upload.py")
    print(f"  3. Deploy:")
    print(f"     git add . && git commit -m 'Sweep: XGB+CAT+LGBM e=175 d=7 lr=0.1' && git push")
    print(f"  4. Backtest:")
    print(f"     curl -X POST $RAILWAY_API/backtest/ncaa")


if __name__ == "__main__":
    main()
