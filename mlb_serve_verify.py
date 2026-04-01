#!/usr/bin/env python3
"""
mlb_serve_verify.py — Train/serve feature parity verification
═══════════════════════════════════════════════════════════════
Picks N recent games, sends them through both the training feature builder
and the serve-time predict_mlb path, and compares feature values.

Usage:
    python3 mlb_serve_verify.py              # Spot-check 10 games
    python3 mlb_serve_verify.py --n 30       # Check 30 games
"""
import sys, os, random
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

TOLERANCE_ABS = 0.10
TOLERANCE_REL = 0.15

RED, YEL, GRN, RST = "\033[91m", "\033[93m", "\033[92m", "\033[0m"


def main():
    n_samples = 10
    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n_samples = int(sys.argv[i + 1])

    print("=" * 70)
    print("  MLB TRAIN/SERVE FEATURE PARITY CHECK")
    print("=" * 70)

    from db import sb_get, load_model
    from sports.mlb import mlb_build_features, predict_mlb

    bundle = load_model("mlb")
    if not bundle:
        print("  ERROR: No MLB model loaded"); return
    feature_cols = bundle["feature_cols"]
    print(f"  Model: {bundle.get('model_type','?')} with {len(feature_cols)} features\n")

    # Fetch recent completed games with raw stats
    rows = sb_get("mlb_predictions",
                  "result_entered=eq.true&actual_home_runs=not.is.null"
                  "&home_woba=not.is.null&select=*&order=game_date.desc&limit=200")
    if not rows:
        print("  No completed games with raw stats found"); return

    sample = random.sample(rows, min(n_samples, len(rows)))
    print(f"  Checking {len(sample)} games...\n")

    total_fail, total_warn = 0, 0
    feat_issues = {}

    for idx, row in enumerate(sample):
        teams = f"{row.get('away_team','?')}@{row.get('home_team','?')}"
        gdate = row.get("game_date", "?")

        # ── Training path ──
        try:
            df = pd.DataFrame([row])
            X_train = mlb_build_features(df)
            train_vals = {c: float(X_train[c].iloc[0]) for c in feature_cols if c in X_train.columns}
        except Exception as e:
            print(f"  [{idx+1}] {teams} {gdate}  {RED}TRAIN ERROR: {e}{RST}"); continue

        # ── Serve path ──
        try:
            result = predict_mlb(row)
            if "error" in result:
                print(f"  [{idx+1}] {teams} {gdate}  {RED}SERVE ERROR: {result['error']}{RST}"); continue
            serve_vals = {s["feature"]: s["value"] for s in result.get("shap", [])}
        except Exception as e:
            print(f"  [{idx+1}] {teams} {gdate}  {RED}SERVE EXCEPTION: {e}{RST}"); continue

        # ── Compare ──
        mismatches = []
        for feat in feature_cols:
            tv = train_vals.get(feat, 0)
            sv = serve_vals.get(feat, 0)
            diff = abs(tv - sv)
            denom = max(abs(tv), abs(sv), 1e-6)

            if diff < TOLERANCE_ABS:
                status = "OK"
            elif diff / denom < TOLERANCE_REL:
                status = "OK"
            elif diff < 0.5:
                status = "WARN"
            else:
                status = "FAIL"

            if status in ("WARN", "FAIL"):
                mismatches.append((feat, tv, sv, diff, status))
                feat_issues[feat] = feat_issues.get(feat, 0) + 1
                if status == "FAIL": total_fail += 1
                else: total_warn += 1

        tag = f"{GRN}PASS{RST}" if not mismatches else (
            f"{RED}FAIL({sum(1 for m in mismatches if m[4]=='FAIL')}){RST}" if any(m[4]=="FAIL" for m in mismatches)
            else f"{YEL}WARN({len(mismatches)}){RST}")
        cov = result.get("feature_coverage", "?")
        rolling = "✓" if result.get("rolling_stats_loaded") else "✗"
        print(f"  [{idx+1}] {teams:>12s} {gdate}  cov={cov} roll={rolling}  {tag}")
        for feat, tv, sv, diff, status in mismatches:
            c = RED if status == "FAIL" else YEL
            print(f"        {c}{feat:30s}  train={tv:8.4f}  serve={sv:8.4f}  Δ={diff:.4f}{RST}")

    print(f"\n{'='*70}")
    print(f"  TOTAL: {total_fail} FAILs, {total_warn} WARNs across {len(sample)} games")
    if feat_issues:
        print(f"\n  Most frequent mismatches:")
        for f, c in sorted(feat_issues.items(), key=lambda x: -x[1])[:10]:
            print(f"    {f:30s}  {c} games")
    else:
        print(f"\n  {GRN}All features match!{RST}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
