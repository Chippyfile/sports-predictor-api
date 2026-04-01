#!/usr/bin/env python3
"""
mlb_fix_leakage.py — Replace leaked season-level wOBA/K9/BB9 with rolling values
═══════════════════════════════════════════════════════════════════════════════════
PROBLEM: home_woba, home_k9, home_bb9 are end-of-season values in mlb_historical.
         A game on April 15 uses the team's full-season stats including September.
         This is data leakage — inflates all model metrics.

FIX: Compute rolling 20-game wOBA/K9/BB9 from per-game batting/pitching in teamstats.csv.
     Each game only uses data from PRIOR games (no lookahead).

wOBA formula: (0.69×BB + 0.72×HBP + 0.89×1B + 1.27×2B + 1.62×3B + 2.10×HR) / (AB + BB - IBB + SF + HBP)
K/9 = K * 27 / IP_outs
BB/9 = BB * 27 / IP_outs
"""
import pandas as pd, numpy as np, os

LOOKBACK = 20  # rolling window
MIN_GAMES = 5   # minimum before computing

# wOBA weights (2015-2024 average from FanGraphs Guts!)
W_BB = 0.69
W_HBP = 0.72
W_1B = 0.89
W_2B = 1.27
W_3B = 1.62
W_HR = 2.10

print("=" * 70)
print("  MLB LEAKAGE FIX: Rolling wOBA / K9 / BB9 from per-game data")
print("=" * 70)

# ── Load teamstats ──
print("\nLoading teamstats.csv...")
ts = pd.read_csv("teamstats.csv", low_memory=False)
ts = ts[(ts['date'] >= 20150101) & (ts['stattype'] == 'value')].copy()
ts['game_date'] = pd.to_datetime(ts['date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
ts['season'] = pd.to_datetime(ts['date'].astype(str), format='%Y%m%d').dt.year

# Parse batting columns
for col in ['b_ab', 'b_h', 'b_d', 'b_t', 'b_hr', 'b_w', 'b_iw', 'b_hbp', 'b_sf', 'b_k']:
    ts[col] = pd.to_numeric(ts[col], errors='coerce').fillna(0)

# Parse pitching columns
for col in ['p_k', 'p_w', 'p_ipouts']:
    ts[col] = pd.to_numeric(ts[col], errors='coerce').fillna(0)

# Compute per-game stats
ts['b_1b'] = ts['b_h'] - ts['b_d'] - ts['b_t'] - ts['b_hr']  # singles
ts['woba_num'] = (W_BB * ts['b_w'] + W_HBP * ts['b_hbp'] + W_1B * ts['b_1b'] +
                   W_2B * ts['b_d'] + W_3B * ts['b_t'] + W_HR * ts['b_hr'])
ts['woba_den'] = ts['b_ab'] + ts['b_w'] - ts['b_iw'] + ts['b_sf'] + ts['b_hbp']
ts['game_woba'] = np.where(ts['woba_den'] > 0, ts['woba_num'] / ts['woba_den'], 0.315)

# K/9 and BB/9 (pitching)
ts['game_k9'] = np.where(ts['p_ipouts'] > 0, ts['p_k'] * 27.0 / ts['p_ipouts'], 8.5)
ts['game_bb9'] = np.where(ts['p_ipouts'] > 0, ts['p_w'] * 27.0 / ts['p_ipouts'], 3.2)

ts = ts.sort_values(['team', 'date']).reset_index(drop=True)
print(f"  {len(ts)} team-game rows")
print(f"  Per-game wOBA: mean={ts['game_woba'].mean():.3f}, std={ts['game_woba'].std():.3f}")
print(f"  Per-game K/9: mean={ts['game_k9'].mean():.1f}, std={ts['game_k9'].std():.1f}")

# ── Compute rolling stats per team (PRIOR games only, no lookahead) ──
print(f"\nComputing rolling {LOOKBACK}-game wOBA/K9/BB9...")

# Key: (game_date, team, vishome) → (rolling_woba, rolling_k9, rolling_bb9)
rolling_lookup = {}
for (team, season), grp in ts.groupby(['team', 'season']):
    wobas = grp['game_woba'].values
    k9s = grp['game_k9'].values
    bb9s = grp['game_bb9'].values
    dates = grp['game_date'].values
    vishomes = grp['vishome'].values
    
    for i in range(len(wobas)):
        if i < MIN_GAMES:
            # Not enough prior games — use season default (will be conservative)
            rolling_lookup[(dates[i], team, vishomes[i])] = (0.315, 8.5, 3.2)
        else:
            lo = max(0, i - LOOKBACK)
            rwoba = float(np.mean(wobas[lo:i]))
            rk9 = float(np.mean(k9s[lo:i]))
            rbb9 = float(np.mean(bb9s[lo:i]))
            rolling_lookup[(dates[i], team, vishomes[i])] = (
                round(rwoba, 4), round(rk9, 2), round(rbb9, 2)
            )

print(f"  {len(rolling_lookup)} team-game rolling entries")

# ── Build game-level lookup ──
game_rolling = {}  # match_key → (home_woba, away_woba, home_k9, away_k9, home_bb9, away_bb9)
for gid, group in ts.groupby('gid'):
    home = group[group['vishome'] == 'h']
    away = group[group['vishome'] == 'v']
    if home.empty or away.empty:
        continue
    hr = home.iloc[0]
    ar = away.iloc[0]
    
    h_stats = rolling_lookup.get((hr['game_date'], hr['team'], 'h'), (0.315, 8.5, 3.2))
    a_stats = rolling_lookup.get((ar['game_date'], ar['team'], 'v'), (0.315, 8.5, 3.2))
    
    key = f"{hr['game_date']}|{hr['team']}|{ar['team']}"
    game_rolling[key] = {
        'home_woba': h_stats[0], 'away_woba': a_stats[0],
        'home_k9': h_stats[1], 'away_k9': a_stats[1],
        'home_bb9': h_stats[2], 'away_bb9': a_stats[2],
    }

print(f"  {len(game_rolling)} games with rolling stats")

# ── Apply to training data ──
print(f"\nLoading mlb_training_data.parquet...")
hist = pd.read_parquet("mlb_training_data.parquet")
hist['match_key'] = hist['game_date'] + '|' + hist['home_team'] + '|' + hist['away_team']

# Save originals for comparison
orig_woba = hist['home_woba'].copy()
orig_k9 = hist['home_k9'].copy()

hits = 0
for idx in hist.index:
    r = game_rolling.get(hist.at[idx, 'match_key'])
    if r:
        hits += 1
        hist.at[idx, 'home_woba'] = r['home_woba']
        hist.at[idx, 'away_woba'] = r['away_woba']
        hist.at[idx, 'home_k9'] = r['home_k9']
        hist.at[idx, 'away_k9'] = r['away_k9']
        hist.at[idx, 'home_bb9'] = r['home_bb9']
        hist.at[idx, 'away_bb9'] = r['away_bb9']

print(f"\n  Matched: {hits}/{len(hist)} ({100*hits/len(hist):.1f}%)")

# Verify: values should now vary within a season
print(f"\n  BEFORE (leaked): home_woba unique per team-season = 1 (constant)")
new_unique = hist.groupby([pd.to_datetime(hist['game_date']).dt.year, 'home_team'])['home_woba'].nunique()
print(f"  AFTER (rolling): home_woba unique per team-season = {new_unique.mean():.0f} avg")
print(f"  home_woba: mean={hist['home_woba'].mean():.4f}, std={hist['home_woba'].std():.4f}")
print(f"  home_k9: mean={hist['home_k9'].mean():.2f}, std={hist['home_k9'].std():.2f}")

# Show sample team
sample = hist[hist['home_team'] == 'NYA'].head(5)[['game_date', 'home_woba', 'home_k9', 'home_bb9']]
print(f"\n  NYA sample (should vary game to game):")
for _, r in sample.iterrows():
    print(f"    {r['game_date']}: woba={r['home_woba']:.4f}, k9={r['home_k9']:.2f}, bb9={r['home_bb9']:.2f}")

hist.drop(columns=['match_key'], errors='ignore', inplace=True)
hist.to_parquet("mlb_training_data.parquet", index=False)
print(f"\n✅ Saved ({os.path.getsize('mlb_training_data.parquet')//1024} KB)")
print(f"\nIMPORTANT: Re-run mlb_ensemble_retrain.py to get honest metrics.")
print(f"Expected: MAE ~3.45-3.55, Win% ~56-57% (lower but HONEST)")
