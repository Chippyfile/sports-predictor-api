"""
regrade_win_pct.py — Regrade all ncaa_predictions with ML isotonic-calibrated probabilities.

Replaces sigma-based win_pct_home (Brier ~0.174) with StackedClassifier + isotonic (Brier ~0.111).
Also updates spread_home with ML margin prediction.

Usage:
    python3 regrade_win_pct.py --dry-run   # Show what would change, don't write
    python3 regrade_win_pct.py             # Regrade all predictions
"""

import sys, os, time, requests, json
import numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '.')
from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
MODEL_PATH = 'ncaa_model_local.pkl'


def sb_get_all(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, timeout=60)
        if not r.ok:
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data

def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print(f"  REGRADE WIN_PCT_HOME — {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    # Load model
    print(f"\n  Loading model from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    reg = bundle["reg"]
    isotonic = bundle.get("isotonic")
    bias = bundle.get("bias_correction", 0.0)
    print(f"  Model: {len(feature_cols)} features, MAE={bundle.get('mae_cv')}")

    # Load referee profiles
    try:
        with open("referee_profiles.json") as f:
            ncaa_build_features._ref_profiles = json.load(f)
        print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
    except FileNotFoundError:
        print("  referee_profiles.json not found - ref features zero")

    # Load full historical data (has all 383 columns)
    print(f"\n  Loading ncaa_historical from parquet cache...")
    hist = pd.read_parquet("ncaa_training_data.parquet")
    print(f"  Historical: {len(hist)} rows × {len(hist.columns)} cols")

    # Fetch prediction IDs and game_ids
    print(f"  Fetching ncaa_predictions...")
    pred_rows = sb_get_all("ncaa_predictions", "select=id,game_id,win_pct_home")
    print(f"  Predictions: {len(pred_rows)}")

    pred_df = pd.DataFrame(pred_rows)
    pred_df["win_pct_home"] = pd.to_numeric(pred_df["win_pct_home"], errors="coerce").fillna(0.5)

    # Match predictions to historical rows by game_id
    hist["game_id"] = hist["game_id"].astype(str)
    pred_df["game_id"] = pred_df["game_id"].astype(str)
    matched = pred_df.merge(hist, on="game_id", how="inner", suffixes=("_pred", "_hist"))
    print(f"  Matched: {len(matched)}/{len(pred_df)} predictions to historical data")

    if len(matched) == 0:
        print("  ❌ No matches found. Check game_id format.")
        return

    # Build features from historical data (which has all columns)
    hist_matched = hist[hist["game_id"].isin(matched["game_id"])].copy()
    hist_matched = hist_matched.drop_duplicates(subset="game_id", keep="last")

    for col in ["actual_home_score", "actual_away_score", "home_adj_em", "away_adj_em",
                "home_ppg", "away_ppg", "home_opp_ppg", "away_opp_ppg",
                "home_tempo", "away_tempo", "home_rank", "away_rank", "season"]:
        if col in hist_matched.columns:
            hist_matched[col] = pd.to_numeric(hist_matched[col], errors="coerce")

    if "home_record_wins" in hist_matched.columns and "home_wins" not in hist_matched.columns:
        hist_matched["home_wins"] = hist_matched["home_record_wins"]
    if "away_record_wins" in hist_matched.columns and "away_wins" not in hist_matched.columns:
        hist_matched["away_wins"] = hist_matched["away_record_wins"]
    if "home_record_losses" in hist_matched.columns and "home_losses" not in hist_matched.columns:
        hist_matched["home_losses"] = hist_matched["home_record_losses"]
    if "away_record_losses" in hist_matched.columns and "away_losses" not in hist_matched.columns:
        hist_matched["away_losses"] = hist_matched["away_record_losses"]

    print("  Running heuristic backfill...")
    hist_matched = _ncaa_backfill_heuristic(hist_matched)

    print("  Building features...")
    X = ncaa_build_features(hist_matched)

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    print("  Generating ML probabilities...")
    X_s = scaler.transform(X)

    raw_probs = clf.predict_proba(X_s)[:, 1]
    cal_probs = isotonic.predict(raw_probs) if isotonic else raw_probs
    cal_probs = np.clip(cal_probs, 0.05, 0.95)

    margins = reg.predict(X_s) - bias

    # Map back to prediction IDs
    hist_matched = hist_matched.reset_index(drop=True)
    game_id_to_results = {}
    for i, row in hist_matched.iterrows():
        gid = str(row["game_id"])
        game_id_to_results[gid] = {
            "prob": round(float(cal_probs[i]), 4),
            "margin": round(float(margins[i]), 1),
        }

    # Compare old vs new
    changes = []
    for _, prow in pred_df.iterrows():
        gid = str(prow["game_id"])
        if gid not in game_id_to_results:
            continue
        old_p = float(prow["win_pct_home"])
        new_p = game_id_to_results[gid]["prob"]
        new_m = game_id_to_results[gid]["margin"]
        changes.append({
            "id": prow["id"],
            "game_id": gid,
            "old_prob": old_p,
            "new_prob": new_p,
            "new_margin": new_m,
            "shift": new_p - old_p,
        })

    shifts = np.array([c["shift"] for c in changes])
    print(f"\n  Probability changes ({len(changes)} games):")
    print(f"    Mean shift: {shifts.mean():+.4f}")
    print(f"    Abs mean:   {np.abs(shifts).mean():.4f}")
    print(f"    Max shift:  {shifts.max():+.4f}")
    print(f"    Min shift:  {shifts.min():+.4f}")
    print(f"    >5% change: {(np.abs(shifts) > 0.05).sum()} games")
    print(f"    >10% change: {(np.abs(shifts) > 0.10).sum()} games")

    sorted_changes = sorted(changes, key=lambda x: abs(x["shift"]), reverse=True)
    print(f"\n  Largest shifts:")
    for c in sorted_changes[:10]:
        print(f"    {c['game_id'][:20]:20s} old={c['old_prob']:.3f} → new={c['new_prob']:.3f} (shift={c['shift']:+.3f})")

    if dry_run:
        print(f"\n  DRY RUN complete. Would update {len(changes)} predictions.")
        return

    print(f"\n  Pushing {len(changes)} updates to Supabase...")
    success, errors = 0, 0
    for i, c in enumerate(changes):
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/ncaa_predictions?id=eq.{c['id']}",
            headers={
                "apikey": KEY, "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json", "Prefer": "return=minimal",
            },
            json={"win_pct_home": c["new_prob"], "spread_home": c["new_margin"]},
            timeout=10,
        )
        if resp.ok:
            success += 1
        else:
            errors += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(changes)} | success={success} errors={errors}")

    print(f"\n  ✅ Regrade complete: {success} updated, {errors} errors")

if __name__ == "__main__":
    main()
