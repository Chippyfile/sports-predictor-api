#!/usr/bin/env python3
"""
wire_rolling_and_refs.py — Wire rolling player features + referee profiles into production
═══════════════════════════════════════════════════════════════════════════════════════════
Two patches to sports/ncaa.py:
  1. Rolling player features (4 differentials from prior-game averages)
  2. Referee crew features (computed from referee profiles)

Also builds referee_profiles from historical game data.

Usage:
  python3 wire_rolling_and_refs.py --build-profiles    # build ref profiles first
  python3 wire_rolling_and_refs.py --dry-run           # preview ncaa.py changes
  python3 wire_rolling_and_refs.py                     # apply changes
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
NCAA_PY = "sports/ncaa.py"
PROFILES_FILE = "referee_profiles.json"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def sb_get(table, params=""):
    all_data = []
    offset = 0
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}&limit=1000&offset={offset}"
        h = {**HEADERS, "Range": f"{offset}-{offset+999}"}
        r = requests.get(url, headers=h, timeout=60)
        if not r.ok:
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < 1000:
            break
        offset += 1000
        if offset % 10000 == 0:
            print(f"    {len(all_data)} rows...")
    return all_data


def build_referee_profiles():
    """Build referee profiles from historical game data."""
    print("=" * 70)
    print("  BUILDING REFEREE PROFILES")
    print("=" * 70)

    print("  Fetching games with referee data...")
    rows = sb_get("ncaa_historical",
                  "referee_1=not.is.null&actual_home_score=not.is.null"
                  "&select=referee_1,referee_2,referee_3,"
                  "actual_home_score,actual_away_score,"
                  "home_fta_rate,away_fta_rate,home_tempo,away_tempo,"
                  "home_fouls,away_fouls,home_turnovers,away_turnovers")

    if not rows:
        print("  ERROR: No games with referee data")
        return {}

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ["referee_1", "referee_2", "referee_3"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_score"] = df["actual_home_score"].fillna(0) + df["actual_away_score"].fillna(0)
    df["margin"] = (df["actual_home_score"].fillna(0) - df["actual_away_score"].fillna(0)).abs()
    df["home_won"] = (df["actual_home_score"] > df["actual_away_score"]).astype(int)
    df["total_fouls"] = df.get("home_fouls", pd.Series(0)).fillna(0) + df.get("away_fouls", pd.Series(0)).fillna(0)
    df["avg_fta_rate"] = (df["home_fta_rate"].fillna(0.34) + df["away_fta_rate"].fillna(0.34)) / 2
    df["avg_tempo"] = (df["home_tempo"].fillna(68) + df["away_tempo"].fillna(68)) / 2

    print(f"  Games with refs: {len(df)}")

    # Aggregate per referee (appears as ref_1, ref_2, or ref_3)
    ref_stats = {}

    for _, row in df.iterrows():
        game_data = {
            "total_score": row["total_score"],
            "margin": row["margin"],
            "home_won": row["home_won"],
            "total_fouls": row["total_fouls"],
            "avg_fta_rate": row["avg_fta_rate"],
            "avg_tempo": row["avg_tempo"],
        }

        for ref_col in ["referee_1", "referee_2", "referee_3"]:
            name = row.get(ref_col)
            if pd.isna(name) or not name:
                continue
            name = str(name).strip()
            if name not in ref_stats:
                ref_stats[name] = []
            ref_stats[name].append(game_data)

    print(f"  Unique referees: {len(ref_stats)}")

    # Compute profiles (min 20 games for reliability)
    MIN_GAMES = 20
    profiles = {}

    for name, games in ref_stats.items():
        n = len(games)
        if n < MIN_GAMES:
            continue

        total_scores = [g["total_score"] for g in games]
        margins = [g["margin"] for g in games]
        home_wins = [g["home_won"] for g in games]
        fouls = [g["total_fouls"] for g in games if g["total_fouls"] > 0]
        fta_rates = [g["avg_fta_rate"] for g in games]
        tempos = [g["avg_tempo"] for g in games]

        profiles[name] = {
            "games": n,
            "pace_impact": round(float(np.mean(total_scores)), 2),       # avg total points
            "ou_bias": round(float(np.mean(total_scores)) - 145.0, 2),   # vs league avg ~145
            "home_whistle": round(float(np.mean(home_wins)) - 0.5, 4),   # home win rate above 50%
            "foul_rate": round(float(np.mean(fouls)), 2) if fouls else 0, # avg total fouls
            "blowout_rate": round(float(np.mean([1 for m in margins if m > 20]) / n), 4),
            "fta_rate_avg": round(float(np.mean(fta_rates)), 4),
        }

    print(f"  Profiles (≥{MIN_GAMES} games): {len(profiles)}")

    # Summary stats
    if profiles:
        ou_biases = [p["ou_bias"] for p in profiles.values()]
        home_whistles = [p["home_whistle"] for p in profiles.values()]
        print(f"  O/U bias range: [{min(ou_biases):.1f}, {max(ou_biases):.1f}]")
        print(f"  Home whistle range: [{min(home_whistles):.4f}, {max(home_whistles):.4f}]")

    # Save
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"  Saved: {PROFILES_FILE}")

    return profiles


def patch_ncaa_py(dry_run=False):
    """Add rolling player features and referee crew features to ncaa_build_features."""
    print("\n" + "=" * 70)
    print("  PATCHING sports/ncaa.py")
    print("=" * 70)

    with open(NCAA_PY) as f:
        txt = f.read()

    changes = []

    # ── 1. Add rolling player feature raw columns ──
    rolling_defaults = '''
        # ── ROLLING PLAYER FEATURES (legitimate pre-game signal) ──
        "home_roll_star1_share": 0.25, "away_roll_star1_share": 0.25,
        "home_roll_top3_share": 0.65, "away_roll_top3_share": 0.65,
        "home_roll_bench_share": 0.20, "away_roll_bench_share": 0.20,
        "home_roll_bench_pts": 15.0, "away_roll_bench_pts": 15.0,'''

    # Find the last raw_cols entry before the closing brace
    marker = '        "importance_multiplier": 1.0,'
    if "roll_star1_share" not in txt and marker in txt:
        txt = txt.replace(marker, marker + rolling_defaults)
        changes.append("Added rolling player raw columns with defaults")

    # ── 2. Add rolling differentials to feature computation ──
    rolling_features = '''
    # ── Rolling player features (from prior-game averages, no leakage) ──
    df["roll_star1_share_diff"] = df["home_roll_star1_share"] - df["away_roll_star1_share"]
    df["roll_top3_share_diff"] = df["home_roll_top3_share"] - df["away_roll_top3_share"]
    df["roll_bench_share_diff"] = df["home_roll_bench_share"] - df["away_roll_bench_share"]
    df["roll_bench_pts_diff"] = df["home_roll_bench_pts"] - df["away_roll_bench_pts"]
'''

    # Insert before the feature_cols list
    feature_cols_marker = '    feature_cols = ['
    if "roll_star1_share_diff" not in txt and feature_cols_marker in txt:
        txt = txt.replace(feature_cols_marker, rolling_features + feature_cols_marker)
        changes.append("Added rolling differential computations")

    # ── 3. Add rolling features to feature_cols list ──
    rolling_feature_names = '''
        # ── Rolling player features ──
        "roll_star1_share_diff", "roll_top3_share_diff",
        "roll_bench_share_diff", "roll_bench_pts_diff",'''

    # Find the last feature_cols entry
    last_feature_marker = '        "espn_predictor_edge",'
    if "roll_star1_share_diff" not in txt and last_feature_marker in txt:
        txt = txt.replace(last_feature_marker, last_feature_marker + rolling_feature_names)
        changes.append("Added rolling features to feature_cols list")

    # ── 4. Add referee raw columns ──
    ref_defaults = '''
        # ── REFEREE CREW FEATURES ──
        "referee_1": "", "referee_2": "", "referee_3": "",'''

    if '"referee_1"' not in txt and marker in txt:
        txt = txt.replace(marker + rolling_defaults, marker + rolling_defaults + ref_defaults)
        changes.append("Added referee raw columns")

    # ── 5. Add referee profile lookup and crew features ──
    ref_features = '''
    # ── Referee crew features (from prebuilt profiles) ──
    _ref_profiles = getattr(ncaa_build_features, "_ref_profiles", {})
    if _ref_profiles:
        crew_ou_bias = []
        crew_home_whistle = []
        crew_foul_rate = []
        crew_pace = []
        for _, row_data in df.iterrows():
            ou_vals, hw_vals, fr_vals, pace_vals = [], [], [], []
            for ref_col in ["referee_1", "referee_2", "referee_3"]:
                name = str(row_data.get(ref_col, "")).strip()
                if name and name in _ref_profiles:
                    p = _ref_profiles[name]
                    ou_vals.append(p["ou_bias"])
                    hw_vals.append(p["home_whistle"])
                    fr_vals.append(p.get("foul_rate", 0))
                    pace_vals.append(p["pace_impact"])
            crew_ou_bias.append(np.mean(ou_vals) if ou_vals else 0.0)
            crew_home_whistle.append(np.mean(hw_vals) if hw_vals else 0.0)
            crew_foul_rate.append(np.mean(fr_vals) if fr_vals else 0.0)
            crew_pace.append(np.mean(pace_vals) if pace_vals else 145.0)
        df["ref_ou_bias"] = crew_ou_bias
        df["ref_home_whistle"] = crew_home_whistle
        df["ref_foul_rate"] = crew_foul_rate
        df["ref_pace_impact"] = [p - 145.0 for p in crew_pace]
        df["has_ref_data"] = (df["ref_ou_bias"] != 0.0).astype(int)
    else:
        df["ref_ou_bias"] = 0.0
        df["ref_home_whistle"] = 0.0
        df["ref_foul_rate"] = 0.0
        df["ref_pace_impact"] = 0.0
        df["has_ref_data"] = 0
'''

    if "ref_ou_bias" not in txt and feature_cols_marker in txt:
        txt = txt.replace(rolling_features + feature_cols_marker,
                          rolling_features + ref_features + feature_cols_marker)
        changes.append("Added referee profile lookup and crew features")

    # ── 6. Add ref features to feature_cols list ──
    ref_feature_names = '''
        # ── Referee crew features ──
        "ref_ou_bias", "ref_home_whistle", "ref_foul_rate",
        "ref_pace_impact", "has_ref_data",'''

    if "ref_ou_bias" not in txt:
        insert_after = rolling_feature_names if rolling_feature_names.strip() in txt else last_feature_marker
        # Find rolling features in feature_cols
        roll_marker = '        "roll_bench_pts_diff",'
        if roll_marker in txt:
            txt = txt.replace(roll_marker, roll_marker + ref_feature_names)
            changes.append("Added referee features to feature_cols list")

    # ── 7. Add ref profile loading in train_ncaa ──
    ref_load = '''
        # Load referee profiles for feature builder
        try:
            import json as _json
            with open("referee_profiles.json") as _rf:
                ncaa_build_features._ref_profiles = _json.load(_rf)
            print(f"  Loaded {len(ncaa_build_features._ref_profiles)} referee profiles")
        except FileNotFoundError:
            print("  referee_profiles.json not found — ref features will be zero")
            ncaa_build_features._ref_profiles = {}
'''

    train_marker = "        X  = ncaa_build_features(df)"
    if "_ref_profiles" not in txt and train_marker in txt:
        txt = txt.replace(train_marker, ref_load + train_marker)
        changes.append("Added referee profile loading in train_ncaa")

    # Summary
    print(f"\n  Changes ({len(changes)}):")
    for c in changes:
        print(f"    ✓ {c}")

    if dry_run:
        print(f"\n  DRY RUN — no files written.")
        return

    with open(NCAA_PY + ".pre_rolling_ref", "w") as f:
        with open(NCAA_PY) as orig:
            f.write(orig.read())

    with open(NCAA_PY, "w") as f:
        f.write(txt)
    print(f"\n  Backup: {NCAA_PY}.pre_rolling_ref")
    print(f"  Patched: {NCAA_PY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-profiles", action="store_true", help="Build referee profiles")
    parser.add_argument("--dry-run", action="store_true", help="Preview ncaa.py changes")
    parser.add_argument("--patch-only", action="store_true", help="Skip profile building, just patch")
    args = parser.parse_args()

    if args.build_profiles or not args.patch_only:
        profiles = build_referee_profiles()

    patch_ncaa_py(dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"  NEXT STEPS:")
        print(f"  1. Also patch retrain_and_upload.py to load ref profiles")
        print(f"  2. Retrain: python3 retrain_and_upload.py")
        print(f"  3. Deploy: git add . && git push")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
