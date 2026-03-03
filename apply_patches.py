#!/usr/bin/env python3
"""Apply MLB cap increase + NBA historical merge. Run from sports-predictor-api dir."""
import re

# ── Part 1: Bump MLB MAX_TRAIN from 8000 to 15000 ──
with open('sports/mlb.py') as f:
    mlb = f.read()
mlb = mlb.replace('MAX_TRAIN = 8000', 'MAX_TRAIN = 15000', 1)
with open('sports/mlb.py', 'w') as f:
    f.write(mlb)
print('  sports/mlb.py: MAX_TRAIN 8000 -> 15000')

# ── Part 2: Add NBA historical merge to sports/nba.py ──
with open('sports/nba.py') as f:
    nba = f.read()

merge_fn = '''
def _nba_season_weight(season):
    current = 2026
    age = current - season
    if age <= 0: return 1.0
    if age == 1: return 1.0
    if age == 2: return 0.9
    if age == 3: return 0.8
    if age == 4: return 0.7
    if age == 5: return 0.6
    return 0.5


def _nba_merge_historical(current_df):
    hist_rows = sb_get(
        "nba_historical",
        "is_outlier_season=eq.false&actual_home_score=not.is.null&select=*&order=season.desc&limit=50000"
    )
    if not hist_rows:
        print("  WARNING: nba_historical empty -- current season only")
        if current_df is not None and len(current_df) > 0:
            return current_df, None, 0
        return pd.DataFrame(), None, 0
    hist_df = pd.DataFrame(hist_rows)
    for col in hist_df.columns:
        if col not in ['home_team', 'away_team', 'game_date', 'id', 'season', 'is_outlier_season', 'home_win', 'result_entered']:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
    if "season" in hist_df.columns:
        hist_df["season_weight"] = hist_df["season"].apply(
            lambda s: _nba_season_weight(int(s)) if pd.notna(s) else 0.5
        )
    if current_df is not None and len(current_df) > 0:
        combined = pd.concat([hist_df, current_df], ignore_index=True)
    else:
        combined = hist_df
    weights = combined["season_weight"].fillna(1.0).astype(float).values if "season_weight" in combined.columns else None
    n_hist = len(hist_df)
    n_curr = len(current_df) if current_df is not None else 0
    print(f"  NBA corpus: {n_hist} historical + {n_curr} current = {n_hist + n_curr}")
    return combined, weights, n_hist

'''

marker = 'def train_nba():'
if marker in nba:
    nba = nba.replace(marker, merge_fn + marker, 1)
    print('  sports/nba.py: Added _nba_merge_historical')
else:
    print('  ERROR: could not find train_nba')

# ── Part 3: Update train_nba to use historical merge ──
old_fetch = '''rows = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        if not rows or len(rows) < 10:
            return {"error": "Not enough NBA data", "n": len(rows) if rows else 0}

        df = pd.DataFrame(rows)'''

new_fetch = '''rows = sb_get("nba_predictions",
                      "result_entered=eq.true&actual_home_score=not.is.null&select=*")
        current_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Merge with historical corpus (2021-2025)
        df, sample_weights, n_historical = _nba_merge_historical(current_df)
        if len(df) < 10:
            return {"error": "Not enough NBA data", "n": len(df), "n_current": len(current_df)}'''

if old_fetch in nba:
    nba = nba.replace(old_fetch, new_fetch, 1)
    print('  sports/nba.py: train_nba now uses historical merge')
else:
    print('  WARNING: Could not find exact old fetch pattern -- manual edit needed')
    print('  Look for the sb_get nba_predictions block in train_nba and replace with merge')

# ── Part 4: Wire sample_weights into NBA training ──
# Add weights to the _time_series_oof call
old_oof = '_time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds)'
new_oof = '_time_series_oof(reg_models, X_scaled, y_margin, df, n_splits=cv_folds, weights=sample_weights)'
if old_oof in nba:
    nba = nba.replace(old_oof, new_oof, 1)
    print('  sports/nba.py: sample_weights wired into ts-cv')

with open('sports/nba.py', 'w') as f:
    f.write(nba)
print('\nDone. Next steps:')
print('  1. Run nba_historical_migration.sql in Supabase SQL editor')
print('  2. git add . && git commit -m "MLB 15k cap + NBA historical merge" && git push')
print('  3. NBA will train on current data only until nba_historical is populated')
