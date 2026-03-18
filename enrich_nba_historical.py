"""
NBA Historical Data Enrichment
Computes missing columns from existing game results — no API calls needed.

Fixes:
  - home_wins / home_losses / away_wins / away_losses: cumulative at game date (leak-free)
  - home_form / away_form: last-5-game weighted form score
  - home_days_rest / away_days_rest: computed from schedule gaps
  - away_travel_dist: haversine distance from previous game city
  - home_tempo / away_tempo: estimated from PPG + FG% when ESPN default (100.0)

Usage:
  python enrich_nba_historical.py                # Enrich parquet only
  python enrich_nba_historical.py --push         # Also push to Supabase
"""

import os, sys, time, math, json
import numpy as np
import pandas as pd

# City coordinates (same as nbaUtils.js)
NBA_CITY_COORDS = {
    "ATL": (33.749, -84.388), "BOS": (42.360, -71.059), "BKN": (40.693, -73.975),
    "CHA": (35.227, -80.843), "CHI": (41.882, -87.628), "CLE": (41.499, -81.694),
    "DAL": (32.777, -96.797), "DEN": (39.739, -104.990), "DET": (42.331, -83.046),
    "GSW": (37.775, -122.419), "HOU": (29.760, -95.370), "IND": (39.768, -86.158),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.150, -90.049),
    "MIA": (25.762, -80.192), "MIL": (43.039, -87.907), "MIN": (44.978, -93.265),
    "NOP": (29.951, -90.072), "NYK": (40.751, -73.993), "OKC": (35.468, -97.516),
    "ORL": (28.538, -81.379), "PHI": (39.953, -75.165), "PHX": (33.448, -112.074),
    "POR": (45.523, -122.677), "SAC": (38.582, -121.494), "SAS": (29.424, -98.494),
    "TOR": (43.653, -79.383), "UTA": (40.761, -111.891), "WAS": (38.907, -77.037),
}

def haversine(coord1, coord2):
    """Distance in miles between two (lat, lng) pairs."""
    R = 3959  # Earth radius in miles
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def enrich(df):
    """
    Enrich NBA data with computed features from game results.
    All computations are leak-free (only use data available BEFORE each game).
    """
    df = df.copy()
    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Ensure numeric
    for col in ["actual_home_score", "actual_away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    n = len(df)
    print(f"  Enriching {n} games...")
    
    # ── Build per-team game logs ──
    # For each team, track chronological game results
    team_games = {}  # {team: [(game_date, won, opponent_abbr, was_home), ...]}
    
    # First pass: collect all games per team
    for idx, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        d = row["game_date"]
        hs = row.get("actual_home_score")
        as_ = row.get("actual_away_score")
        
        if pd.isna(hs) or pd.isna(as_):
            continue
            
        h_won = float(hs) > float(as_)
        
        if h not in team_games:
            team_games[h] = []
        if a not in team_games:
            team_games[a] = []
        
        team_games[h].append({"date": d, "won": h_won, "opp": a, "home": True})
        team_games[a].append({"date": d, "won": not h_won, "opp": h, "home": False})
    
    # Sort each team's games chronologically
    for team in team_games:
        team_games[team].sort(key=lambda g: g["date"])
    
    # ── Second pass: compute features for each game ──
    home_wins_arr = np.zeros(n)
    home_losses_arr = np.zeros(n)
    away_wins_arr = np.zeros(n)
    away_losses_arr = np.zeros(n)
    home_form_arr = np.zeros(n)
    away_form_arr = np.zeros(n)
    home_rest_arr = np.full(n, 2.0)
    away_rest_arr = np.full(n, 2.0)
    away_travel_arr = np.zeros(n)
    
    def _cumulative(team, game_date):
        """Get cumulative W/L/form using only games BEFORE game_date."""
        log = team_games.get(team, [])
        prior = [g for g in log if g["date"] < game_date]
        
        wins = sum(1 for g in prior if g["won"])
        losses = len(prior) - wins
        
        # Form: weighted last-5 results (matches JS: (±1) × (i+1) / 15)
        last5 = prior[-5:]
        if last5:
            form = sum((1 if g["won"] else -1) * (i + 1) for i, g in enumerate(last5)) / 15.0
        else:
            form = 0.0
        
        return wins, losses, round(form, 4)
    
    def _rest_days(team, game_date):
        """Compute days of rest (days since last game - 1)."""
        log = team_games.get(team, [])
        prior = [g for g in log if g["date"] < game_date]
        if not prior:
            return 7  # Season opener
        last_date = prior[-1]["date"]
        try:
            d1 = pd.Timestamp(last_date)
            d2 = pd.Timestamp(game_date)
            diff = (d2 - d1).days - 1  # -1 because game day itself = 0 rest
            return max(0, min(14, diff))
        except:
            return 2
    
    def _prev_game_city(team, game_date):
        """Get the city (team abbr) where this team played their previous game."""
        log = team_games.get(team, [])
        prior = [g for g in log if g["date"] < game_date]
        if not prior:
            return None
        last = prior[-1]
        if last["home"]:
            return team  # Played at home
        else:
            return last["opp"]  # Played at opponent's city
    
    for idx, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        d = row["game_date"]
        
        # Cumulative W/L + form
        hw, hl, hf = _cumulative(h, d)
        aw, al, af = _cumulative(a, d)
        home_wins_arr[idx] = hw
        home_losses_arr[idx] = hl
        away_wins_arr[idx] = aw
        away_losses_arr[idx] = al
        home_form_arr[idx] = hf
        away_form_arr[idx] = af
        
        # Rest days
        home_rest_arr[idx] = _rest_days(h, d)
        away_rest_arr[idx] = _rest_days(a, d)
        
        # Travel distance (away team's previous city → this game's home city)
        prev_city = _prev_game_city(a, d)
        if prev_city and prev_city in NBA_CITY_COORDS and h in NBA_CITY_COORDS:
            dist = haversine(NBA_CITY_COORDS[prev_city], NBA_CITY_COORDS[h])
            away_travel_arr[idx] = round(dist)
        
        if idx % 2000 == 0 and idx > 0:
            print(f"    Processed {idx}/{n} games...")
    
    # ── Apply to DataFrame ──
    df["home_wins"] = home_wins_arr.astype(int)
    df["home_losses"] = home_losses_arr.astype(int)
    df["away_wins"] = away_wins_arr.astype(int)
    df["away_losses"] = away_losses_arr.astype(int)
    df["home_form"] = home_form_arr
    df["away_form"] = away_form_arr
    df["home_days_rest"] = home_rest_arr.astype(int)
    df["away_days_rest"] = away_rest_arr.astype(int)
    df["away_travel_dist"] = away_travel_arr.astype(int)
    
    # ── Fix tempo: estimate from PPG + FG% when default 100.0 ──
    # Dean Oliver: Poss ≈ FGA - ORB + TO + 0.475×FTA
    # Approximate: FGA ≈ PPG / FG%, FTA ≈ PPG × 0.22, ORB ≈ 10, TO ≈ 14
    for side in ["home", "away"]:
        tempo_col = f"{side}_tempo"
        ppg_col = f"{side}_ppg"
        fgpct_col = f"{side}_fgpct"
        
        if tempo_col in df.columns and ppg_col in df.columns and fgpct_col in df.columns:
            ppg = pd.to_numeric(df[ppg_col], errors="coerce").fillna(112)
            fgpct = pd.to_numeric(df[fgpct_col], errors="coerce").fillna(0.471)
            tempo = pd.to_numeric(df[tempo_col], errors="coerce")
            
            # Estimate pace when tempo is default (100.0) or missing
            is_default = (tempo == 100.0) | tempo.isna()
            if is_default.sum() > 0:
                fga_est = ppg / fgpct.clip(0.35)
                fta_est = ppg * 0.22
                orb_est = 10.5
                to_est = 14.0
                pace_est = fga_est - orb_est + to_est + 0.475 * fta_est
                pace_est = pace_est.clip(90, 108)
                df.loc[is_default, tempo_col] = pace_est[is_default].round(1)
                print(f"    Fixed {is_default.sum()} {side} tempo values (was 100.0 default)")
    
    # ── Diagnostics ──
    print(f"\n  Enrichment complete:")
    print(f"    home_wins range: [{df['home_wins'].min()}, {df['home_wins'].max()}]")
    print(f"    home_form range: [{df['home_form'].min():.3f}, {df['home_form'].max():.3f}]")
    print(f"    home_days_rest: mean={df['home_days_rest'].mean():.1f}, B2B count={( df['home_days_rest'] == 0).sum()}")
    print(f"    away_travel_dist: mean={df['away_travel_dist'].mean():.0f} mi, max={df['away_travel_dist'].max():.0f} mi")
    print(f"    home_tempo range: [{df['home_tempo'].min():.1f}, {df['home_tempo'].max():.1f}]")
    
    return df


def push_to_supabase(df):
    """Push enriched columns back to nba_historical in Supabase."""
    import requests
    
    SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
    KEY = os.environ.get("SUPABASE_ANON_KEY", "")
    if not KEY:
        print("  ERROR: SUPABASE_ANON_KEY not set")
        return
    
    headers = {
        "apikey": KEY, "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    
    # Only update historical rows (those with season column)
    hist = df[df.get("season", pd.Series(dtype=float)).notna()].copy()
    if len(hist) == 0:
        print("  No historical rows to push")
        return
    
    # Build update payloads — match on game_date + home_team + away_team
    enriched_cols = ["home_wins", "home_losses", "away_wins", "away_losses",
                     "home_form", "away_form", "home_days_rest", "away_days_rest",
                     "away_travel_dist", "home_tempo", "away_tempo"]
    
    batch = []
    pushed = 0
    
    for idx, row in hist.iterrows():
        update = {
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
        }
        for col in enriched_cols:
            val = row.get(col)
            if pd.notna(val):
                update[col] = float(val) if isinstance(val, (np.floating, np.integer)) else val
        
        batch.append(update)
        
        if len(batch) >= 100:
            try:
                resp = requests.post(
                    f"{SUPABASE_URL}/rest/v1/nba_historical",
                    headers=headers, json=batch, timeout=30,
                )
                if resp.ok:
                    pushed += len(batch)
                else:
                    print(f"    Batch error: {resp.status_code} {resp.text[:200]}")
            except Exception as e:
                print(f"    Batch exception: {e}")
            batch = []
            if pushed % 1000 == 0:
                print(f"    Pushed {pushed}/{len(hist)}...")
    
    # Flush remaining
    if batch:
        try:
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/nba_historical",
                headers=headers, json=batch, timeout=30,
            )
            if resp.ok:
                pushed += len(batch)
        except:
            pass
    
    print(f"  Pushed {pushed}/{len(hist)} rows to Supabase")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Historical Data Enrichment")
    print("=" * 60)
    
    # Load parquet
    parquet_path = "nba_training_data.parquet"
    if not os.path.exists(parquet_path):
        print(f"  ERROR: {parquet_path} not found — run dump_nba_training_data.py first")
        sys.exit(1)
    
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows from {parquet_path}")
    
    # Before stats
    print(f"\n  BEFORE enrichment:")
    print(f"    home_wins coverage: {df['home_wins'].notna().sum()}/{len(df)} ({df['home_wins'].notna().mean()*100:.1f}%)")
    print(f"    home_form coverage: {df['home_form'].notna().sum()}/{len(df)} ({df['home_form'].notna().mean()*100:.1f}%)")
    print(f"    home_days_rest non-default: {(df['home_days_rest'] != 2).sum()}/{len(df)}")
    print(f"    away_travel_dist > 0: {(df['away_travel_dist'] > 0).sum()}/{len(df)}")
    
    # Enrich
    t0 = time.time()
    df = enrich(df)
    print(f"\n  Enrichment took {time.time()-t0:.1f}s")
    
    # After stats
    print(f"\n  AFTER enrichment:")
    print(f"    home_wins coverage: {df['home_wins'].notna().sum()}/{len(df)} ({df['home_wins'].notna().mean()*100:.1f}%)")
    print(f"    home_form coverage: {df['home_form'].notna().sum()}/{len(df)} ({df['home_form'].notna().mean()*100:.1f}%)")
    print(f"    home_days_rest non-default: {(df['home_days_rest'] != 2).sum()}/{len(df)}")
    print(f"    away_travel_dist > 0: {(df['away_travel_dist'] > 0).sum()}/{len(df)}")
    
    # Save enriched parquet
    df.to_parquet(parquet_path, index=False)
    print(f"\n  ✅ Saved enriched data to {parquet_path} ({os.path.getsize(parquet_path)/1024:.0f} KB)")
    
    # Optionally push to Supabase
    if "--push" in sys.argv:
        print("\n  Pushing to Supabase...")
        push_to_supabase(df)
    else:
        print("\n  To push to Supabase: python3 enrich_nba_historical.py --push")
