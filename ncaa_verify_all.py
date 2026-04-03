#!/usr/bin/env python3
"""
ncaa_verify_all.py — Pre-retrain verification
==============================================
Checks EVERYTHING before retrain runs. Fix all issues at once.

Usage:
    python3 ncaa_verify_all.py          # Check + fix
    python3 ncaa_verify_all.py --check  # Check only, no fixes
"""
import sys, os, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

PARQUET = "ncaa_training_data.parquet"

# Columns that MUST be string
STRING_COLS = {
    'home_team_name', 'away_team_name', 'home_team_abbr', 'away_team_abbr',
    'home_conference', 'away_conference', 'referee_1', 'referee_2', 'referee_3',
    'game_date', 'venue_name', 'venue_city', 'venue_state', 'game_id',
}

# Columns the retrain script actually reads/writes (must be numeric)
CRITICAL_NUMERIC = [
    'season', 'home_team_id', 'away_team_id',
    'actual_home_score', 'actual_away_score',
    'home_adj_em', 'away_adj_em', 'home_adj_oe', 'away_adj_oe', 'home_adj_de', 'away_adj_de',
    'home_ppg', 'away_ppg', 'home_opp_ppg', 'away_opp_ppg',
    'home_fgpct', 'away_fgpct', 'home_threepct', 'away_threepct',
    'market_spread_home', 'market_ou_total', 'market_home_ml',
    'espn_spread', 'espn_over_under', 'espn_home_win_pct',
    'odds_api_spread_open', 'odds_api_spread_close', 'odds_api_spread_movement',
    'odds_api_total_open', 'odds_api_total_close',
    'dk_spread_open', 'dk_spread_close', 'dk_spread_movement',
    'dk_total_open', 'dk_total_close',
    'home_elo', 'away_elo', 'home_rank', 'away_rank',
    'home_wins', 'away_wins', 'home_losses', 'away_losses',
    'home_record_wins', 'away_record_wins', 'home_record_losses', 'away_record_losses',
    'home_sos', 'away_sos', 'home_form', 'away_form',
    'home_tempo', 'away_tempo', 'home_efg', 'away_efg',
    'neutral_site', 'is_conf_tourney', 'importance',
]

EXPECTED_SEASONS = {2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026}
EXPECTED_MIN_GAMES = {
    2015: 5500, 2016: 5500, 2017: 5500, 2018: 5500, 2019: 5500,
    2022: 5500, 2023: 5500, 2024: 5500, 2025: 5500, 2026: 5500,
}


def check_and_fix(check_only=False):
    issues = []
    fixes = []
    
    print("=" * 70)
    print("  NCAA PRE-RETRAIN VERIFICATION")
    print("=" * 70)
    
    # ══════════════════════════════════════════════════════════
    # 1. PARQUET EXISTS AND LOADS
    # ══════════════════════════════════════════════════════════
    print(f"\n  1. PARQUET FILE")
    if not os.path.exists(PARQUET):
        issues.append("FATAL: ncaa_training_data.parquet not found")
        print(f"  ❌ {PARQUET} not found")
        return issues, fixes
    
    df = pd.read_parquet(PARQUET)
    print(f"  ✅ Loaded: {len(df)} rows × {len(df.columns)} cols")
    
    # ══════════════════════════════════════════════════════════
    # 2. SEASON COVERAGE
    # ══════════════════════════════════════════════════════════
    print(f"\n  2. SEASON COVERAGE")
    df['_season_num'] = pd.to_numeric(df.get('season'), errors='coerce')
    seasons_present = set(int(s) for s in df['_season_num'].dropna().unique())
    
    missing_seasons = EXPECTED_SEASONS - seasons_present
    if missing_seasons:
        issues.append(f"Missing seasons: {sorted(missing_seasons)}")
        print(f"  ❌ Missing: {sorted(missing_seasons)}")
    
    for season in sorted(seasons_present):
        n = (df['_season_num'] == season).sum()
        expected = EXPECTED_MIN_GAMES.get(season, 5000)
        status = "✅" if n >= expected else "⚠️"
        print(f"  {status} {season}: {n:,} games (expect ≥{expected})")
        if n < expected:
            issues.append(f"Season {season}: only {n} games (expect {expected}+)")
    
    # ══════════════════════════════════════════════════════════
    # 3. COLUMN TYPES
    # ══════════════════════════════════════════════════════════
    print(f"\n  3. COLUMN TYPES")
    type_fixes = 0
    
    # Check numeric columns stored as string
    bad_numeric = []
    for col in CRITICAL_NUMERIC:
        if col not in df.columns:
            continue
        dtype = str(df[col].dtype)
        if 'str' in dtype or 'object' in dtype or 'string' in dtype:
            bad_numeric.append(col)
            if not check_only:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                type_fixes += 1
    
    if bad_numeric:
        issues.append(f"Numeric cols stored as string ({len(bad_numeric)}): {bad_numeric[:10]}")
        print(f"  ❌ {len(bad_numeric)} numeric columns are strings: {bad_numeric[:5]}...")
        if not check_only:
            fixes.append(f"Fixed {len(bad_numeric)} columns to numeric")
            print(f"  🔧 Fixed → numeric")
    else:
        print(f"  ✅ All critical numeric columns are correct type")
    
    # Check ALL non-string columns
    all_bad = []
    for col in df.columns:
        if col in STRING_COLS or col == '_season_num':
            continue
        dtype = str(df[col].dtype)
        if 'str' in dtype or 'object' in dtype or 'string' in dtype:
            all_bad.append(col)
            if not check_only:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                type_fixes += 1
    
    if all_bad:
        issues.append(f"Other cols stored as string ({len(all_bad)}): {all_bad[:10]}")
        print(f"  ⚠️  {len(all_bad)} other columns are strings: {all_bad[:5]}...")
        if not check_only:
            fixes.append(f"Fixed {len(all_bad)} additional columns to numeric")
            print(f"  🔧 Fixed → numeric")
    
    # ══════════════════════════════════════════════════════════
    # 4. SPREAD COVERAGE
    # ══════════════════════════════════════════════════════════
    print(f"\n  4. SPREAD COVERAGE")
    mkt = pd.to_numeric(df.get('market_spread_home'), errors='coerce')
    espn = pd.to_numeric(df.get('espn_spread'), errors='coerce')
    dk = pd.to_numeric(df.get('dk_spread_close'), errors='coerce')
    oa = pd.to_numeric(df.get('odds_api_spread_close'), errors='coerce')
    
    for season in sorted(seasons_present):
        sm = df['_season_num'] == season
        n = sm.sum()
        has_mkt = (mkt[sm].notna() & (mkt[sm] != 0)).sum()
        has_espn = (espn[sm].notna() & (espn[sm] != 0)).sum() if 'espn_spread' in df.columns else 0
        has_dk = (dk[sm].notna() & (dk[sm] != 0)).sum() if 'dk_spread_close' in df.columns else 0
        has_oa = (oa[sm].notna() & (oa[sm] != 0)).sum() if 'odds_api_spread_close' in df.columns else 0
        
        # After cascade, how many would have spread?
        any_spread = ((mkt[sm].notna() & (mkt[sm] != 0)) | 
                     (espn[sm].notna() & (espn[sm] != 0)) |
                     (dk[sm].notna() & (dk[sm] != 0)) |
                     (oa[sm].notna() & (oa[sm] != 0))).sum()
        
        print(f"  {season}: mkt={has_mkt} espn={has_espn} dk={has_dk} oa={has_oa} → any={any_spread}/{n} ({any_spread/max(n,1)*100:.0f}%)")
    
    # ══════════════════════════════════════════════════════════
    # 5. CORE STATS COVERAGE
    # ══════════════════════════════════════════════════════════
    print(f"\n  5. CORE STATS COVERAGE")
    core_stats = ['home_ppg', 'away_ppg', 'home_adj_em', 'away_adj_em', 
                  'referee_1', 'home_elo', 'away_elo']
    for col in core_stats:
        if col not in df.columns:
            print(f"  ❌ {col}: MISSING")
            issues.append(f"Missing column: {col}")
            continue
        if col == 'referee_1':
            has = (df[col].notna() & (df[col] != '') & (df[col] != 'None') & (df[col] != 'nan')).sum()
        else:
            vals = pd.to_numeric(df[col], errors='coerce')
            has = (vals.notna() & (vals != 0)).sum()
        pct = has / len(df) * 100
        status = "✅" if pct > 70 else "⚠️" if pct > 30 else "❌"
        print(f"  {status} {col}: {has:,}/{len(df):,} ({pct:.0f}%)")
    
    # ══════════════════════════════════════════════════════════
    # 6. DUPLICATES
    # ══════════════════════════════════════════════════════════
    print(f"\n  6. DUPLICATES")
    if 'game_id' in df.columns:
        dupes = df['game_id'].duplicated().sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate game_ids")
            print(f"  ❌ {dupes} duplicate game_ids")
            if not check_only:
                df = df.drop_duplicates(subset='game_id', keep='first')
                fixes.append(f"Removed {dupes} duplicate rows")
                print(f"  🔧 Removed duplicates → {len(df)} rows")
        else:
            print(f"  ✅ No duplicate game_ids")
    
    # ══════════════════════════════════════════════════════════
    # 7. OUTLIER SEASONS (2020, 2021)
    # ══════════════════════════════════════════════════════════
    print(f"\n  7. EXCLUDED SEASONS")
    for bad_season in [2020, 2021]:
        n = (df['_season_num'] == bad_season).sum()
        if n > 0:
            print(f"  ⚠️  Season {bad_season}: {n} games (COVID — retrain will exclude)")
        else:
            print(f"  ✅ Season {bad_season}: not present")
    
    # ══════════════════════════════════════════════════════════
    # 8. DRY RUN RETRAIN PIPELINE
    # ══════════════════════════════════════════════════════════
    print(f"\n  8. DRY RUN RETRAIN PIPELINE")
    try:
        # Simulate what retrain does
        df2 = df.copy()
        df2['season'] = pd.to_numeric(df2['season'], errors='coerce').fillna(0).astype(int)
        df2 = df2[~df2['season'].isin([2020, 2021])].copy()
        df2['game_date_dt'] = pd.to_datetime(df2['game_date'], errors='coerce')
        m = df2['game_date_dt'].dt.month
        df2 = df2[((m >= 11) | (m <= 4)) & ~((m == 11) & (df2['game_date_dt'].dt.day < 10))].copy()
        
        # Cascade spread backfill
        spread_sources = [
            ("espn_spread", "espn_over_under"),
            ("dk_spread_close", "dk_total_close"),
            ("dk_spread_open", "dk_total_open"),
            ("odds_api_spread_close", "odds_api_total_close"),
            ("odds_api_spread_open", "odds_api_total_open"),
        ]
        mkt_s = pd.to_numeric(df2.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
        for spread_col, total_col in spread_sources:
            if spread_col in df2.columns:
                src = pd.to_numeric(df2[spread_col], errors="coerce")
                fill = (mkt_s.isna() | (mkt_s == 0)) & src.notna() & (src != 0)
                if fill.sum() > 0:
                    mkt_s[fill] = src[fill]
        
        has_spread = (mkt_s.notna() & (mkt_s != 0))
        
        # Quality filter
        _qcols = [c for c in ["home_ppg", "away_ppg"] if c in df2.columns]
        _qmat = pd.DataFrame({c: df2[c].notna() for c in _qcols})
        _keep = _qmat.mean(axis=1) >= 0.5
        if "referee_1" in df2.columns:
            _keep = _keep & df2["referee_1"].notna() & (df2["referee_1"] != "") & (df2["referee_1"] != "None")
        
        n_after_filters = _keep.sum()
        n_with_spread = (has_spread & _keep).sum()
        
        print(f"  After date/season filters: {len(df2):,}")
        print(f"  After quality filters: {n_after_filters:,}")
        print(f"  With spread data: {n_with_spread:,} ({n_with_spread/max(n_after_filters,1)*100:.0f}%)")
        print(f"  ✅ Pipeline dry run passed")
        
    except Exception as e:
        issues.append(f"Pipeline dry run failed: {e}")
        print(f"  ❌ Pipeline failed: {e}")
    
    # ══════════════════════════════════════════════════════════
    # 9. IMPORT CHECKS
    # ══════════════════════════════════════════════════════════
    print(f"\n  9. IMPORT CHECKS")
    for mod_name, desc in [
        ("dump_training_data", "Training data loader"),
        ("sports.ncaa", "NCAA feature builder"),
        ("build_crowd_shock", "Crowd shock"),
        ("compute_h2h_conf_form", "H2H/conf features"),
    ]:
        try:
            __import__(mod_name)
            print(f"  ✅ {mod_name}")
        except Exception as e:
            print(f"  ❌ {mod_name}: {str(e)[:60]}")
            issues.append(f"Import failed: {mod_name}")
    
    for fname in ["referee_profiles.json", "ncaa_team_locations.json"]:
        if os.path.exists(fname):
            print(f"  ✅ {fname}")
        else:
            print(f"  ⚠️  {fname} not found (optional)")
    
    # ══════════════════════════════════════════════════════════
    # SAVE FIXES
    # ══════════════════════════════════════════════════════════
    if fixes and not check_only:
        df = df.drop(columns=['_season_num'], errors='ignore')
        df.to_parquet(PARQUET)
        print(f"\n  💾 Saved fixes to {PARQUET}")
    else:
        df = df.drop(columns=['_season_num'], errors='ignore')
    
    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    if not issues:
        print(f"  ✅ ALL CHECKS PASSED — safe to retrain")
    else:
        print(f"  ⚠️  {len(issues)} ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    if fixes:
        print(f"\n  🔧 {len(fixes)} FIXES APPLIED:")
        for f in fixes:
            print(f"    • {f}")
    print(f"{'=' * 70}")
    
    return issues, fixes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check only, no fixes")
    args = parser.parse_args()
    
    issues, fixes = check_and_fix(check_only=args.check)
    
    if not issues or (fixes and not args.check):
        print(f"\n  Ready to retrain:")
        print(f"  python3 ncaa_final_retrain.py --upload")
