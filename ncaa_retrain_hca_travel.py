#!/usr/bin/env python3
"""
ncaa_retrain_hca_travel.py — Retrain with rolling HCA + travel distance
========================================================================
Replaces static conference HCA with per-team rolling HCA.
Adds travel_advantage for neutral site games.

1. Computes rolling_hca for all training data
2. Fetches venue locations for neutral games → computes travel_advantage
3. Patches hca_pts → rolling_hca in feature pipeline
4. Retrains Lasso+LGBM and compares to baseline

Usage:
    python3 ncaa_retrain_hca_travel.py              # Full retrain + comparison
    python3 ncaa_retrain_hca_travel.py --upload      # Upload best model to Supabase
"""
import sys, os, time, json, math, warnings, argparse, pickle
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from datetime import datetime

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42; N_FOLDS = 30
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
TEAM_LOCATIONS_FILE = "ncaa_team_locations.json"
VENUE_CACHE_FILE = "ncaa_venue_cache.json"

# ── Original 43 features (baseline) ──
FEATURES_43 = [
    "mkt_spread","player_rating_diff","ref_home_whistle","weakest_starter_diff",
    "crowd_shock_diff","lineup_stability_diff","lineup_changes_diff","adj_oe_diff",
    "hca_pts","blowout_asym_diff","threepct_diff","pit_sos_diff","orb_pct_diff",
    "blocks_diff","drb_pct_diff","opp_to_rate_diff","elo_diff","is_early",
    "spread_regime","assist_rate_diff","opp_ppg_diff","opp_suppression_diff",
    "roll_ats_margin_gated","has_ats_data","tempo_avg","form_x_familiarity",
    "to_conversion_diff","conf_strength_diff","roll_rotation_diff","roll_dominance_diff",
    "importance","twopt_diff","roll_ats_diff_gated","overreaction_diff",
    "three_rate_diff","ppp_diff","to_margin_diff","momentum_halflife_diff",
    "starter_experience_diff","style_familiarity","fatigue_x_quality","ato_diff",
    "consistency_x_spread"
]

# ── New features: replace hca_pts with rolling_hca, add travel_advantage ──
FEATURES_NEW = [f if f != "hca_pts" else "rolling_hca" for f in FEATURES_43] + ["travel_advantage"]

# ══════════════════════════════════════════════════════════
# ROLLING HCA
# ══════════════════════════════════════════════════════════

CONF_HCA = {
    "Big 12": 7.3, "SEC": 7.3, "Big Ten": 6.2, "ACC": 5.8,
    "Big East": 4.6, "Pac-12": 4.2, "Mountain West": 5.0,
    "AAC": 5.2, "WCC": 10.6, "Atlantic 10": 6.4,
    "Missouri Valley": 6.1, "Ivy League": 5.4,
}
ESPN_CONF = {
    "8": "Big 12", "23": "SEC", "7": "Big Ten", "2": "ACC", "4": "Big East",
    "21": "Pac-12", "44": "Mountain West", "62": "AAC", "26": "WCC",
    "3": "Atlantic 10", "18": "Missouri Valley", "22": "Ivy League",
}
DEFAULT_HCA = 6.6
WINDOW = 20; MIN_GAMES = 5

def compute_rolling_hca_col(df):
    """Add rolling_hca column. Must be sorted by date."""
    df = df.sort_values("game_date_dt").reset_index(drop=True)
    
    for col in ["home_conference", "away_conference"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: ESPN_CONF.get(x, x))
    
    margin = (pd.to_numeric(df["actual_home_score"], errors="coerce") - 
              pd.to_numeric(df["actual_away_score"], errors="coerce"))
    neutral = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)
    
    team_home = defaultdict(lambda: deque(maxlen=WINDOW))
    team_away = defaultdict(lambda: deque(maxlen=WINDOW))
    
    hca = np.full(len(df), DEFAULT_HCA)
    
    for i in range(len(df)):
        home_id = str(df.at[i, "home_team_id"] if "home_team_id" in df.columns else df.at[i, "home_team"])
        away_id = str(df.at[i, "away_team_id"] if "away_team_id" in df.columns else df.at[i, "away_team"])
        is_n = neutral.iloc[i]
        m = margin.iloc[i]
        conf = str(df.at[i, "home_conference"]) if "home_conference" in df.columns else ""
        
        # Compute BEFORE updating
        hh = list(team_home[home_id])
        ha = list(team_away[home_id])
        if len(hh) >= MIN_GAMES and len(ha) >= MIN_GAMES:
            hca[i] = (np.mean(hh) - np.mean(ha)) / 2
        else:
            hca[i] = CONF_HCA.get(conf, DEFAULT_HCA)
        
        # For neutral sites, HCA = 0 (the rolling value is still computed but zeroed)
        if is_n:
            hca[i] = 0
        
        # Update AFTER computing
        if pd.notna(m) and not is_n:
            team_home[home_id].append(m)
            team_away[away_id].append(-m)
    
    df["rolling_hca"] = hca
    return df

# ══════════════════════════════════════════════════════════
# TRAVEL ADVANTAGE
# ══════════════════════════════════════════════════════════

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 6371 * 0.621371
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def load_team_locations():
    if os.path.exists(TEAM_LOCATIONS_FILE):
        with open(TEAM_LOCATIONS_FILE) as f:
            return json.load(f)
    return None

def load_venue_cache():
    if os.path.exists(VENUE_CACHE_FILE):
        with open(VENUE_CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_venue_cache(cache):
    with open(VENUE_CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Common tournament venue cities → coords
VENUE_CITIES = {
    "Indianapolis": (39.768, -86.158), "Houston": (29.760, -95.370),
    "San Antonio": (29.424, -98.494), "New Orleans": (29.951, -90.072),
    "Minneapolis": (44.978, -93.265), "Glendale": (33.531, -112.189),
    "Atlanta": (33.757, -84.397), "Phoenix": (33.448, -112.074),
    "Omaha": (41.257, -95.935), "Kansas City": (39.100, -94.579),
    "Dallas": (32.777, -96.797), "Denver": (39.739, -104.990),
    "Portland": (45.505, -122.675), "San Jose": (37.338, -121.886),
    "Sacramento": (38.582, -121.494), "Salt Lake City": (40.761, -111.891),
    "Las Vegas": (36.170, -115.140), "Greenville": (34.852, -82.394),
    "Buffalo": (42.886, -78.878), "Oklahoma City": (35.468, -97.516),
    "Memphis": (35.150, -90.049), "Nashville": (36.163, -86.782),
    "Charlotte": (35.227, -80.843), "Raleigh": (35.780, -78.638),
    "Pittsburgh": (40.441, -79.996), "Columbus": (39.961, -82.999),
    "Cleveland": (41.499, -81.694), "Detroit": (42.331, -83.046),
    "Milwaukee": (43.044, -87.917), "Chicago": (41.878, -87.630),
    "St. Louis": (38.627, -90.199), "Louisville": (38.253, -85.759),
    "Lexington": (38.041, -84.504), "Tampa": (27.951, -82.458),
    "Orlando": (28.538, -81.379), "Jacksonville": (30.332, -81.656),
    "Boise": (43.615, -116.202), "Spokane": (47.659, -117.426),
    "Tucson": (32.222, -110.975), "Birmingham": (33.521, -86.802),
    "Brooklyn": (40.683, -73.976), "Albany": (42.651, -73.755),
    "Des Moines": (41.586, -93.625), "Tulsa": (36.154, -95.993),
    "San Diego": (32.716, -117.161), "Dayton": (39.759, -84.192),
    "Hartford": (41.764, -72.685), "Wichita": (37.687, -97.330),
    "Fort Worth": (32.756, -97.331), "Columbia": (34.001, -81.035),
    "Greensboro": (36.073, -79.792), "Providence": (41.824, -71.413),
}

def compute_travel_col(df, locations):
    """Add travel_advantage column for neutral site games."""
    neutral = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)
    travel = np.zeros(len(df))
    
    if locations is None:
        print("  ⚠ No team locations — travel_advantage will be 0")
        df["travel_advantage"] = travel
        return df
    
    # Build abbr → coords lookup
    abbr_coords = {}
    id_coords = {}
    for tid, info in locations.items():
        if info.get("lat"):
            abbr_coords[info.get("abbr", "")] = (info["lat"], info["lng"])
            id_coords[tid] = (info["lat"], info["lng"])
    
    # Load venue cache
    venue_cache = load_venue_cache()
    
    n_neutral = neutral.sum()
    n_computed = 0; n_cached = 0; n_fetched = 0; n_missing = 0
    
    print(f"  {n_neutral:,} neutral site games to process")
    
    neutral_idx = np.where(neutral.values)[0]
    
    for count, i in enumerate(neutral_idx):
        game_id = str(df.at[i, "game_id"]) if "game_id" in df.columns else ""
        home_id = str(df.at[i, "home_team_id"] if "home_team_id" in df.columns else df.at[i, "home_team"])
        away_id = str(df.at[i, "away_team_id"] if "away_team_id" in df.columns else df.at[i, "away_team"])
        home_abbr = str(df.at[i, "home_team"]) if "home_team" in df.columns else ""
        away_abbr = str(df.at[i, "away_team"]) if "away_team" in df.columns else ""
        
        home_loc = id_coords.get(home_id) or abbr_coords.get(home_abbr)
        away_loc = id_coords.get(away_id) or abbr_coords.get(away_abbr)
        
        if not home_loc or not away_loc:
            n_missing += 1
            continue
        
        # Get venue location
        venue_lat, venue_lng = None, None
        
        # Check cache first
        if game_id and game_id in venue_cache:
            venue_lat, venue_lng = venue_cache[game_id]
            n_cached += 1
        elif game_id:
            # Fetch from ESPN
            try:
                resp = requests.get(f"{ESPN_BASE}/summary?event={game_id}", timeout=8)
                if resp.ok:
                    gd = resp.json()
                    venue = gd.get("gameInfo", {}).get("venue", {})
                    addr = venue.get("address", {})
                    city = addr.get("city", "")
                    
                    if addr.get("latitude"):
                        venue_lat = float(addr["latitude"])
                        venue_lng = float(addr["longitude"])
                    elif city in VENUE_CITIES:
                        venue_lat, venue_lng = VENUE_CITIES[city]
                    
                    if venue_lat:
                        venue_cache[game_id] = (venue_lat, venue_lng)
                        n_fetched += 1
                time.sleep(0.05)  # rate limit
            except:
                pass
        
        if venue_lat is None:
            n_missing += 1
            continue
        
        hd = haversine_miles(home_loc[0], home_loc[1], venue_lat, venue_lng)
        ad = haversine_miles(away_loc[0], away_loc[1], venue_lat, venue_lng)
        hd = max(hd, 1); ad = max(ad, 1)
        travel[i] = math.log(ad / hd)
        n_computed += 1
        
        if (count + 1) % 500 == 0:
            save_venue_cache(venue_cache)
            print(f"    {count+1}/{n_neutral}: {n_computed} computed, {n_cached} cached, {n_fetched} fetched, {n_missing} missing")
    
    save_venue_cache(venue_cache)
    df["travel_advantage"] = travel
    
    print(f"  Travel: {n_computed:,} computed, {n_cached:,} cached, {n_fetched:,} fetched, {n_missing:,} missing")
    nonzero = np.count_nonzero(travel)
    if nonzero:
        ta = travel[travel != 0]
        print(f"  Stats: mean={ta.mean():+.3f}, std={ta.std():.3f}, min={ta.min():.3f}, max={ta.max():.3f}")
    
    return df

# ══════════════════════════════════════════════════════════
# WALK-FORWARD + TRAINING
# ══════════════════════════════════════════════════════════

MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED),
}

def walk_forward(X_s, y, n_folds):
    n = len(X_s); fold_size = n // (n_folds + 1); min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder(); m.fit(X_s[:ts], y[:ts]); preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof

def eval_model(oof, y, spreads, label):
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.5)
    pred = oof[valid]; act = y[valid]; sp = spreads[valid]
    mae = np.mean(np.abs(pred - act))
    ats = pred + sp; covers = (ats > 0) == (act + sp > 0)
    disagree = np.abs(pred + sp)
    for thresh, name in [(4, "4+"), (7, "7+"), (10, "10+")]:
        mask = disagree >= thresh
        if mask.sum() > 0:
            acc = covers[mask].mean()
            print(f"    ATS@{name}: {acc:.1%} on {mask.sum()} picks")
    
    # Brier score
    prob_home = 1.0 / (1.0 + np.exp(-pred / 10.0))
    actual_home_win = (act > 0).astype(float)
    brier = np.mean((prob_home - actual_home_win) ** 2)
    
    print(f"    MAE: {mae:.3f}, Brier: {brier:.4f}")
    return mae, brier

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-travel", action="store_true", help="Skip travel feature (faster)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NCAA RETRAIN: Rolling HCA + Travel Distance")
    print("=" * 70)
    
    # Load data
    df = load_cached()
    if df is None:
        df = dump()
    df = df[df["actual_home_score"].notna()].copy()
    df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
    df = df[~df["season"].isin([2020, 2021])].copy()
    df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
    
    # Date filters
    season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
    early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
    df = df[season_mask & early_mask].copy()
    
    # ESPN spread backfill
    if "espn_spread" in df.columns:
        espn_s = pd.to_numeric(df["espn_spread"], errors="coerce")
        mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
        fill = (mkt_s.isna() | (mkt_s == 0)) & espn_s.notna()
        df.loc[fill, "market_spread_home"] = espn_s[fill]
    
    # Quality filter
    _qcols = [c for c in ["home_adj_em","away_adj_em","home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
    _qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["home_adj_em","away_adj_em","market_spread_home","market_ou_total"] else True) for c in _qcols})
    _keep = _qmat.mean(axis=1) >= 0.8
    if "referee_1" in df.columns:
        _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
    df = df.loc[_keep].reset_index(drop=True)
    
    for col in ["actual_home_score","actual_away_score","season","home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
        if s in df.columns and d not in df.columns: df[d] = df[s]
    
    print(f"\n  {len(df):,} games after filters")
    
    # ── Compute features ──
    print("\n  Computing standard features...")
    df = _ncaa_backfill_heuristic(df)
    df = compute_crowd_shock(df, n_games=5)
    df = compute_missing_features(df)
    try:
        from training_data_fixes import apply_training_fixes
        df = apply_training_fixes(df)
    except ImportError: pass
    try:
        with open("referee_profiles.json") as f:
            ncaa_build_features._ref_profiles = json.load(f)
    except: pass
    
    df = df.dropna(subset=["actual_home_score","actual_away_score"])
    
    # ── Compute rolling HCA ──
    print("\n  Computing rolling HCA...")
    df = compute_rolling_hca_col(df)
    rhca = df["rolling_hca"]
    nonzero = (rhca != 0) & rhca.notna()
    print(f"  Rolling HCA: mean={rhca[nonzero].mean():+.2f}, coverage={nonzero.mean():.1%}")
    
    # ── Compute travel advantage ──
    if not args.skip_travel:
        print("\n  Computing travel advantage...")
        locations = load_team_locations()
        df = compute_travel_col(df, locations)
    else:
        print("\n  Skipping travel (--skip-travel)")
        df["travel_advantage"] = 0
    
    # ── Build feature matrix ──
    print("\n  Building feature matrix...")
    X_full = ncaa_build_features(df)
    
    # Inject rolling_hca and travel_advantage into X_full
    X_full["rolling_hca"] = df["rolling_hca"].values
    X_full["travel_advantage"] = df["travel_advantage"].values
    
    y = df["actual_home_score"].values - df["actual_away_score"].values
    spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
    
    # ══════════════════════════════════════════════════════════
    # TEST 1: Baseline (original 43 features)
    # ══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  BASELINE: Original 43 features (static HCA)")
    print(f"{'='*70}")
    
    available_43 = [f for f in FEATURES_43 if f in X_full.columns]
    X1 = X_full[available_43].values
    scaler1 = StandardScaler()
    X1s = scaler1.fit_transform(X1)
    
    print(f"  Features: {len(available_43)}")
    oof1 = walk_forward(X1s, y, N_FOLDS)
    eval_model(oof1, y, spreads, "Baseline")
    
    # ══════════════════════════════════════════════════════════
    # TEST 2: Rolling HCA (replace hca_pts)
    # ══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  NEW: Rolling HCA (replaces static hca_pts)")
    print(f"{'='*70}")
    
    features_rolling = [f if f != "hca_pts" else "rolling_hca" for f in FEATURES_43]
    available_rolling = [f for f in features_rolling if f in X_full.columns]
    X2 = X_full[available_rolling].values
    scaler2 = StandardScaler()
    X2s = scaler2.fit_transform(X2)
    
    print(f"  Features: {len(available_rolling)}")
    oof2 = walk_forward(X2s, y, N_FOLDS)
    eval_model(oof2, y, spreads, "Rolling HCA")
    
    # ══════════════════════════════════════════════════════════
    # TEST 3: Rolling HCA + Travel Advantage
    # ══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  NEW: Rolling HCA + Travel Advantage ({len(FEATURES_NEW)} features)")
    print(f"{'='*70}")
    
    available_new = [f for f in FEATURES_NEW if f in X_full.columns]
    X3 = X_full[available_new].values
    scaler3 = StandardScaler()
    X3s = scaler3.fit_transform(X3)
    
    print(f"  Features: {len(available_new)}")
    oof3 = walk_forward(X3s, y, N_FOLDS)
    eval_model(oof3, y, spreads, "Rolling HCA + Travel")
    
    # ══════════════════════════════════════════════════════════
    # LASSO FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  LASSO COEFFICIENTS (new features)")
    print(f"{'='*70}")
    
    lasso = Lasso(alpha=0.1, max_iter=5000)
    lasso.fit(X3s, y)
    coefs = dict(zip(available_new, lasso.coef_))
    
    # Show new features
    for f in ["rolling_hca", "travel_advantage", "hca_pts"]:
        if f in coefs:
            print(f"  {f:>25s}: {coefs[f]:+.4f} {'✅ KEPT' if abs(coefs[f]) > 0.001 else '❌ DROPPED'}")
    
    print(f"\n  All non-zero features:")
    kept = {k: v for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.001}
    for f, c in kept.items():
        marker = " ← NEW" if f in ["rolling_hca", "travel_advantage"] else ""
        print(f"    {f:>30s}: {c:+.4f}{marker}")
    print(f"\n  Lasso kept {len(kept)}/{len(available_new)} features")
    
    # ══════════════════════════════════════════════════════════
    # UPLOAD
    # ══════════════════════════════════════════════════════════
    
    if args.upload:
        print(f"\n{'='*70}")
        print(f"  UPLOADING BEST MODEL")
        print(f"{'='*70}")
        # TODO: Train final model on full data + upload to Supabase
        print("  Upload not yet implemented — review results first")
    
    print(f"\n  Done.")
