#!/usr/bin/env python3
"""
mlb_lineup_features.py — Build advanced lineup features from backfill data
==========================================================================
Goes beyond raw lineup_woba_diff to extract CHANGE, FORM, and TOTAL signals
that the market may not fully price in.

Features built:
  CHANGE signals (key batter missing proxy):
    - lineup_woba_vs_rolling: tonight vs team's rolling 10-game avg
    - lineup_delta_diff: home delta minus away delta
    - lineup_consistency: std of recent lineup wOBAs (lineup stability)

  FORM signals (hot/cold lineup):
    - lineup_woba_trend: 3-game vs 10-game lineup wOBA (momentum)
    - lineup_trend_diff: home trend minus away trend

  STRUCTURE signals:
    - lineup_top_heavy: top3_woba / lineup_woba ratio
    - lineup_top_heavy_diff: home vs away top-heaviness
    - lineup_bot3_woba_diff: bottom of order quality gap

  O/U signals (total run environment):
    - lineup_total_woba: sum of both lineups' wOBA
    - lineup_total_iso: combined power
    - lineup_total_top3: both top-of-orders quality
    - lineup_woba_vs_rolling_sum: both teams' delta from normal (combined)

Usage:
  python mlb_lineup_features.py          # build and save
  python mlb_lineup_features.py --test   # build + quick ATS/OU test
"""

import numpy as np
import pandas as pd
import os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

BACKFILL_FILE = "mlb_lineup_backfill.parquet"
OUTPUT_FILE = "mlb_lineup_features_advanced.parquet"
ROLLING_WINDOW = 10
TREND_WINDOW = 3


def build_advanced_features():
    """Build advanced lineup features from backfill data."""
    lineup = pd.read_parquet(BACKFILL_FILE)
    print(f"  Loaded {len(lineup)} games from {BACKFILL_FILE}")
    print(f"  Date range: {lineup['game_date'].min()} to {lineup['game_date'].max()}")

    # Sort chronologically
    lineup = lineup.sort_values("game_date").reset_index(drop=True)

    # ── Build rolling tracker per team ──
    # Track each team's recent lineup wOBAs for rolling computations
    team_history = defaultdict(list)  # team -> list of (date, woba, ops, iso, top3, bot3)

    results = []
    for idx, row in lineup.iterrows():
        date = row["game_date"]
        home = row["home_abbr"]
        away = row["away_abbr"]

        if not home or not away:
            continue

        out = {
            "game_pk": row["game_pk"],
            "game_date": date,
            "home_abbr": home,
            "away_abbr": away,
        }

        for side, team, prefix in [("home", home, "home"), ("away", away, "away")]:
            woba = row.get(f"{prefix}_lineup_woba", 0) or 0
            ops = row.get(f"{prefix}_lineup_ops", 0) or 0
            iso = row.get(f"{prefix}_lineup_iso", 0) or 0
            top3 = row.get(f"{prefix}_top3_woba", 0) or 0
            bot3 = row.get(f"{prefix}_bot3_woba", 0) or 0
            matched = row.get(f"{prefix}_lineup_matched", 0) or 0

            if woba < 0.1 or matched < 5:
                # Bad data — skip features for this side
                for feat in ["_woba_vs_rolling", "_lineup_consistency", "_woba_trend",
                             "_top_heavy", "_bot3_woba"]:
                    out[f"{prefix}{feat}"] = 0
                team_history[team].append((date, woba, ops, iso, top3, bot3))
                continue

            history = team_history[team]

            # ── CHANGE: lineup wOBA vs rolling average ──
            if len(history) >= 3:
                recent_wobas = [h[1] for h in history[-ROLLING_WINDOW:] if h[1] > 0.1]
                if recent_wobas:
                    rolling_avg = np.mean(recent_wobas)
                    out[f"{prefix}_woba_vs_rolling"] = round(woba - rolling_avg, 4)
                else:
                    out[f"{prefix}_woba_vs_rolling"] = 0
            else:
                out[f"{prefix}_woba_vs_rolling"] = 0

            # ── CHANGE: lineup consistency (std of recent wOBAs) ──
            if len(history) >= 5:
                recent_wobas = [h[1] for h in history[-ROLLING_WINDOW:] if h[1] > 0.1]
                out[f"{prefix}_lineup_consistency"] = round(np.std(recent_wobas), 4) if len(recent_wobas) >= 3 else 0
            else:
                out[f"{prefix}_lineup_consistency"] = 0

            # ── FORM: 3-game trend vs 10-game ──
            if len(history) >= 5:
                recent_3 = [h[1] for h in history[-TREND_WINDOW:] if h[1] > 0.1]
                recent_10 = [h[1] for h in history[-ROLLING_WINDOW:] if h[1] > 0.1]
                if recent_3 and recent_10:
                    out[f"{prefix}_woba_trend"] = round(np.mean(recent_3) - np.mean(recent_10), 4)
                else:
                    out[f"{prefix}_woba_trend"] = 0
            else:
                out[f"{prefix}_woba_trend"] = 0

            # ── STRUCTURE: top-heavy ratio ──
            if woba > 0.1:
                out[f"{prefix}_top_heavy"] = round(top3 / woba, 4) if top3 > 0 else 1.0
            else:
                out[f"{prefix}_top_heavy"] = 1.0

            # ── STRUCTURE: bottom of order quality ──
            out[f"{prefix}_bot3_woba"] = round(bot3, 4)

            # Update history AFTER computing features (no leakage)
            team_history[team].append((date, woba, ops, iso, top3, bot3))
            # Keep only recent history
            if len(team_history[team]) > 30:
                team_history[team] = team_history[team][-30:]

        # ── DIFF features (ATS signals) ──
        out["lineup_delta_diff"] = round(
            out.get("home_woba_vs_rolling", 0) - out.get("away_woba_vs_rolling", 0), 4)
        out["lineup_consistency_diff"] = round(
            out.get("home_lineup_consistency", 0) - out.get("away_lineup_consistency", 0), 4)
        out["lineup_trend_diff"] = round(
            out.get("home_woba_trend", 0) - out.get("away_woba_trend", 0), 4)
        out["lineup_top_heavy_diff"] = round(
            out.get("home_top_heavy", 1) - out.get("away_top_heavy", 1), 4)
        out["lineup_bot3_diff"] = round(
            out.get("home_bot3_woba", 0) - out.get("away_bot3_woba", 0), 4)

        # ── O/U features (total run environment) ──
        h_woba = row.get("home_lineup_woba", 0) or 0
        a_woba = row.get("away_lineup_woba", 0) or 0
        h_iso = row.get("home_lineup_iso", 0) or 0
        a_iso = row.get("away_lineup_iso", 0) or 0
        h_top3 = row.get("home_top3_woba", 0) or 0
        a_top3 = row.get("away_top3_woba", 0) or 0

        out["lineup_total_woba"] = round(h_woba + a_woba, 4)
        out["lineup_total_iso"] = round(h_iso + a_iso, 4)
        out["lineup_total_top3"] = round(h_top3 + a_top3, 4)
        out["lineup_delta_sum"] = round(
            out.get("home_woba_vs_rolling", 0) + out.get("away_woba_vs_rolling", 0), 4)
        out["lineup_trend_sum"] = round(
            out.get("home_woba_trend", 0) + out.get("away_woba_trend", 0), 4)

        results.append(out)

        if (idx + 1) % 5000 == 0:
            print(f"    [{idx+1}/{len(lineup)}] processed")

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n  Built {len(df)} games with {len(df.columns)} columns")
    print(f"  Saved to {OUTPUT_FILE}")

    # Feature summary
    feat_cols = [c for c in df.columns if c not in ["game_pk", "game_date", "home_abbr", "away_abbr"]]
    print(f"\n  Feature summary ({len(feat_cols)} features):")
    print(f"  {'Feature':>30} {'Mean':>8} {'Std':>8} {'Nonzero':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
    for col in sorted(feat_cols):
        nz = (df[col].abs() > 0.0001).sum()
        if nz > 0:
            print(f"  {col:>30} {df[col].mean():>+7.4f} {df[col].std():>7.4f} {nz:>8}")

    return df


def quick_test(df):
    """Quick ATS + O/U correlation test."""
    from mlb_retrain import load_data, build_features

    train_df = load_data()
    y_margin = (train_df["actual_home_runs"].astype(float) - train_df["actual_away_runs"].astype(float)).values
    y_total = (train_df["actual_home_runs"].astype(float) + train_df["actual_away_runs"].astype(float)).values

    # Merge advanced features
    df["_key"] = df["game_date"] + "|" + df["home_abbr"]
    train_df["_key"] = train_df["game_date"].astype(str) + "|" + train_df["home_team"].astype(str)

    adv_cols = [c for c in df.columns if c not in ["game_pk", "game_date", "home_abbr", "away_abbr", "_key"]]
    df_sub = df[["_key"] + adv_cols].drop_duplicates(subset="_key", keep="first")
    merged = train_df[["_key"]].merge(df_sub, on="_key", how="left")

    matched = merged[adv_cols[0]].notna().sum()
    print(f"\n  Matched {matched}/{len(train_df)} training games")

    # Correlation with margin and total
    print(f"\n  CORRELATION WITH TARGETS:")
    print(f"  {'Feature':>30} {'r(margin)':>10} {'r(total)':>10} {'Signal':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    for col in sorted(adv_cols):
        vals = merged[col].fillna(0).values
        if np.std(vals) < 0.0001:
            continue
        r_margin = np.corrcoef(vals, y_margin)[0, 1]
        r_total = np.corrcoef(vals, y_total)[0, 1]
        best = max(abs(r_margin), abs(r_total))
        signal = "ATS" if abs(r_margin) > abs(r_total) else "O/U"
        if best > 0.005:
            marker = " ⭐" if best > 0.03 else ""
            print(f"  {col:>30} {r_margin:>+9.4f} {r_total:>+9.4f} {signal:>6}{marker}")


if __name__ == "__main__":
    print("=" * 70)
    print("  MLB ADVANCED LINEUP FEATURES")
    print("=" * 70)

    df = build_advanced_features()

    if "--test" in sys.argv:
        quick_test(df)

    print("\n  Done.")
