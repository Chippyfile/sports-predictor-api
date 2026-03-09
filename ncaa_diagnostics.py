#!/usr/bin/env python3
"""
NCAA ML Gap Diagnostics — Check historical data quality
Runs 5 diagnostics against Supabase to identify why heuristic outperforms ML.
"""
import requests, json, sys
import numpy as np

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
# We need the anon key — let's try to read it from the project
SUPABASE_KEY = None

# Try common env locations
import os
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    # Try reading from config files
    for path in ["/mnt/project/.env", "/mnt/user-data/uploads/.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if "SUPABASE" in line and "KEY" in line and "=" in line:
                        key, val = line.strip().split("=", 1)
                        if "ANON" in key or "KEY" in key:
                            SUPABASE_KEY = val.strip().strip('"').strip("'")
                            break
        if SUPABASE_KEY:
            break

def sb_get(table, params="", limit=1000):
    """Fetch from Supabase with pagination."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    all_data = []
    offset = 0
    while True:
        headers["Range"] = f"{offset}-{offset + limit - 1}"
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}"
        r = requests.get(url, headers=headers, timeout=30)
        if not r.ok:
            print(f"  ERROR: {r.status_code} — {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data


def diagnostic_1_spread_compression():
    """Check if historical spreads are compressed (the averaging bug)."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: Spread Compression Check")
    print("=" * 70)
    print("  If the old (A+B)/2 averaging bug was baked into training data,")
    print("  historical spread_home values will be ~50% of what they should be.\n")

    rows = sb_get("ncaa_historical",
                  "actual_home_score=not.is.null&select=spread_home,home_adj_em,away_adj_em,actual_home_score,actual_away_score,win_pct_home&limit=5000")
    if not rows:
        print("  ERROR: No ncaa_historical data found")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in ["spread_home", "home_adj_em", "away_adj_em", "actual_home_score",
                "actual_away_score", "win_pct_home"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["spread_home", "home_adj_em", "away_adj_em"])

    # EM gap should roughly predict spread magnitude
    df["em_gap"] = df["home_adj_em"] - df["away_adj_em"]
    df["actual_margin"] = df["actual_home_score"] - df["actual_away_score"]

    # Check: for games with large EM gap, is spread proportional?
    big_gap = df[df["em_gap"].abs() >= 15].copy()
    if len(big_gap) > 0:
        ratio = (big_gap["spread_home"].abs() / big_gap["em_gap"].abs()).median()
        print(f"  Games with |EM gap| >= 15: {len(big_gap)}")
        print(f"  Median |spread| / |EM gap|: {ratio:.3f}")
        print(f"    Expected: ~0.8-1.2 (spread roughly tracks EM gap)")
        print(f"    If < 0.6: SPREAD COMPRESSION — old averaging bug likely baked in")
        if ratio < 0.6:
            print(f"  ❌ COMPRESSED — spreads are {ratio:.0%} of expected magnitude")
            print(f"     This means ML trained on wrong labels!")
        elif ratio < 0.8:
            print(f"  ⚠️  MARGINAL — spreads may be partially compressed")
        else:
            print(f"  ✅ HEALTHY — spreads track EM gap correctly")
    else:
        print(f"  No games with |EM gap| >= 15 found")

    # Also check spread std dev
    print(f"\n  Overall spread distribution:")
    print(f"    N:      {len(df)}")
    print(f"    Mean:   {df['spread_home'].mean():.2f}")
    print(f"    Std:    {df['spread_home'].std():.2f}")
    print(f"    Min:    {df['spread_home'].min():.1f}")
    print(f"    Max:    {df['spread_home'].max():.1f}")
    print(f"    |Spread| > 15: {(df['spread_home'].abs() > 15).sum()} games ({(df['spread_home'].abs() > 15).mean()*100:.1f}%)")
    print(f"    |Spread| > 20: {(df['spread_home'].abs() > 20).sum()} games ({(df['spread_home'].abs() > 20).mean()*100:.1f}%)")

    # Win pct distribution
    if "win_pct_home" in df.columns:
        wp = df["win_pct_home"].dropna()
        print(f"\n  Win probability distribution:")
        print(f"    Std:    {wp.std():.4f}")
        print(f"    Range:  [{wp.min():.3f}, {wp.max():.3f}]")
        print(f"    < 0.20: {(wp < 0.20).sum()} games")
        print(f"    > 0.80: {(wp > 0.80).sum()} games")
        if wp.std() < 0.10:
            print(f"  ❌ WIN PROBS COMPRESSED — std {wp.std():.4f} is too tight")
            print(f"     All games look like coin flips to the model")
        elif wp.std() < 0.15:
            print(f"  ⚠️  WIN PROBS MARGINAL — std {wp.std():.4f}")
        else:
            print(f"  ✅ WIN PROBS HEALTHY — good spread of certainties")

    return df


def diagnostic_2_spread_vs_market_leak():
    """Check if spread_vs_market equals score_diff_pred when no market data."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: spread_vs_market Data Leak Check")
    print("=" * 70)
    print("  When has_market=0, spread_vs_market should be 0.")
    print("  If it equals score_diff_pred, there's a leak.\n")

    rows = sb_get("ncaa_historical",
                  "select=spread_home,market_spread_home,spread_vs_market,pred_home_score,pred_away_score&limit=3000")
    if not rows:
        rows = sb_get("ncaa_predictions",
                      "select=spread_home,market_spread_home,spread_vs_market,pred_home_score,pred_away_score&limit=3000")
    if not rows:
        print("  No data available to check")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Games without market data
    no_mkt = df[(df["market_spread_home"].isna()) | (df["market_spread_home"] == 0)]
    has_mkt = df[(df["market_spread_home"].notna()) & (df["market_spread_home"] != 0)]

    print(f"  Total rows: {len(df)}")
    print(f"  With market data: {len(has_mkt)} ({len(has_mkt)/len(df)*100:.0f}%)")
    print(f"  Without market data: {len(no_mkt)} ({len(no_mkt)/len(df)*100:.0f}%)")

    if "spread_vs_market" in df.columns and len(no_mkt) > 0:
        svm_no_mkt = no_mkt["spread_vs_market"].dropna()
        if len(svm_no_mkt) > 0:
            print(f"\n  spread_vs_market when NO market data:")
            print(f"    Mean:  {svm_no_mkt.mean():.3f}")
            print(f"    Std:   {svm_no_mkt.std():.3f}")
            print(f"    Range: [{svm_no_mkt.min():.1f}, {svm_no_mkt.max():.1f}]")
            if svm_no_mkt.std() > 1.0:
                print(f"  ❌ LEAK DETECTED — spread_vs_market is non-zero when no market!")
                print(f"     This feature is leaking the model's own prediction as 'market signal'")
            else:
                print(f"  ✅ CLEAN — spread_vs_market correctly zeroed when no market data")
    else:
        print(f"  spread_vs_market column not found in data")


def diagnostic_3_heuristic_accuracy_by_bucket():
    """Check heuristic accuracy by spread bucket to see where ML fails."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Heuristic Accuracy by Spread Bucket")
    print("=" * 70)
    print("  Shows where the heuristic is strong vs. weak.\n")

    rows = sb_get("ncaa_historical",
                  "actual_home_score=not.is.null&select=spread_home,actual_home_score,actual_away_score,home_adj_em,away_adj_em&limit=10000")
    if not rows:
        print("  No ncaa_historical data")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["spread_home", "actual_home_score", "actual_away_score"])
    df["actual_margin"] = df["actual_home_score"] - df["actual_away_score"]
    df["heur_correct"] = ((df["spread_home"] > 0) == (df["actual_margin"] > 0))
    df["abs_spread"] = df["spread_home"].abs()

    buckets = [(0, 3, "0-3 (tossup)"), (3, 7, "3-7 (lean)"),
               (7, 12, "7-12 (solid)"), (12, 18, "12-18 (big)"),
               (18, 50, "18+ (blowout)")]

    print(f"  {'Bucket':<20} {'Games':>6} {'Heur Acc':>10} {'Spread MAE':>12}")
    print(f"  {'-'*52}")
    for lo, hi, label in buckets:
        sub = df[(df["abs_spread"] >= lo) & (df["abs_spread"] < hi)]
        if len(sub) > 0:
            acc = sub["heur_correct"].mean()
            mae = (sub["spread_home"] - sub["actual_margin"]).abs().mean()
            print(f"  {label:<20} {len(sub):>6} {acc*100:>9.1f}% {mae:>11.1f}")

    overall_acc = df["heur_correct"].mean()
    overall_mae = (df["spread_home"] - df["actual_margin"]).abs().mean()
    print(f"  {'OVERALL':<20} {len(df):>6} {overall_acc*100:>9.1f}% {overall_mae:>11.1f}")


def diagnostic_4_feature_distribution():
    """Check if key ML features have reasonable distributions."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: ML Feature Distributions")
    print("=" * 70)
    print("  Checks if features used by ML have sane values.\n")

    rows = sb_get("ncaa_historical",
                  "select=home_adj_em,away_adj_em,home_ppg,away_ppg,home_fgpct,away_fgpct,home_tempo,away_tempo,market_spread_home,market_ou_total&limit=5000")
    if not rows:
        print("  No data")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    features = [
        ("home_adj_em", -10, 40, "EM should range -10 to 40"),
        ("away_adj_em", -10, 40, "EM should range -10 to 40"),
        ("home_ppg", 55, 100, "PPG should range 55-100"),
        ("away_ppg", 55, 100, "PPG should range 55-100"),
        ("home_fgpct", 0.35, 0.55, "FG% should range 0.35-0.55"),
        ("away_fgpct", 0.35, 0.55, "FG% should range 0.35-0.55"),
        ("home_tempo", 58, 85, "Tempo should range 58-85"),
        ("away_tempo", 58, 85, "Tempo should range 58-85"),
        ("market_spread_home", -35, 35, "Market spread should range ±35"),
    ]

    for col, exp_lo, exp_hi, desc in features:
        if col not in df.columns:
            print(f"  {col:<25} — NOT PRESENT")
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"  {col:<25} — ALL NULL")
            continue
        pct_in_range = ((vals >= exp_lo) & (vals <= exp_hi)).mean()
        status = "✅" if pct_in_range > 0.90 else "⚠️" if pct_in_range > 0.70 else "❌"
        print(f"  {status} {col:<25} mean={vals.mean():>7.2f}  std={vals.std():>6.2f}  "
              f"range=[{vals.min():.1f}, {vals.max():.1f}]  "
              f"in-range={pct_in_range*100:.0f}%  null={df[col].isna().sum()}")


def diagnostic_5_market_coverage():
    """Check how many historical games have market data."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 5: Market Data Coverage")
    print("=" * 70)
    print("  ML uses market_spread and market_total as features.")
    print("  If most historical games lack market data, these features are noise.\n")

    rows = sb_get("ncaa_historical",
                  "select=season,market_spread_home,market_ou_total&limit=20000")
    if not rows:
        print("  No data")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total = len(df)
    has_spread = ((df["market_spread_home"].notna()) & (df["market_spread_home"] != 0)).sum()
    has_ou = ((df["market_ou_total"].notna()) & (df["market_ou_total"] != 0)).sum()

    print(f"  Total historical games: {total}")
    print(f"  With market spread: {has_spread} ({has_spread/total*100:.1f}%)")
    print(f"  With market O/U: {has_ou} ({has_ou/total*100:.1f}%)")

    if "season" in df.columns:
        print(f"\n  By season:")
        for season in sorted(df["season"].dropna().unique()):
            sub = df[df["season"] == season]
            n = len(sub)
            n_spread = ((sub["market_spread_home"].notna()) & (sub["market_spread_home"] != 0)).sum()
            print(f"    {int(season)}: {n:>5} games, {n_spread:>5} with spread ({n_spread/n*100:.0f}%)")

    if has_spread / total < 0.5:
        print(f"\n  ❌ LOW COVERAGE — only {has_spread/total*100:.0f}% of games have market data")
        print(f"     market_spread and spread_vs_market features are mostly zeros")
        print(f"     → The model can't learn from Vegas on {total-has_spread} games")
    else:
        print(f"\n  ✅ ADEQUATE COVERAGE — {has_spread/total*100:.0f}% of games have market data")


# ═══════════════════════════════════════════════════════════════
# RUN ALL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not SUPABASE_KEY:
        print("ERROR: No Supabase key found.")
        print("Set SUPABASE_ANON_KEY environment variable or provide .env file")
        sys.exit(1)

    print("=" * 70)
    print("  NCAA ML GAP DIAGNOSTICS")
    print("  Why does the heuristic outperform ML? (77.5% vs 73.8%)")
    print("=" * 70)

    df = diagnostic_1_spread_compression()
    diagnostic_2_spread_vs_market_leak()
    diagnostic_3_heuristic_accuracy_by_bucket()
    diagnostic_4_feature_distribution()
    diagnostic_5_market_coverage()

    print("\n" + "=" * 70)
    print("  SUMMARY & NEXT STEPS")
    print("=" * 70)
    print("  If Diag 1 shows compression → re-backfill ncaa_historical")
    print("  If Diag 2 shows leak → fix spread_vs_market zeroing")
    print("  If Diag 3 shows heur weak in tossups → ML should help there")
    print("  If Diag 4 shows bad features → data pipeline issue")
    print("  If Diag 5 shows low coverage → market features are noise")
    print("=" * 70)
