#!/usr/bin/env python3
"""
ncaa_final_retrain.py — Comprehensive NCAA retrain with ALL fixes
=================================================================
1. Rolling HCA (replaces static conference HCA)
2. Travel advantage from venue_city (100% neutral coverage)
3. Corrected neutral_em_diff (strip ~6.5 pts not ~1.5)
4. σ=6.0 for win probability
5. Parlay confidence gate re-sweep

Usage:
    python3 ncaa_final_retrain.py              # Run comparison + sweep
    python3 ncaa_final_retrain.py --upload     # Upload best model
"""
import sys, os, time, json, math, warnings, argparse, pickle, io
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from datetime import datetime

from sports.ncaa import ncaa_build_features, _ncaa_backfill_heuristic
from dump_training_data import dump, load_cached
from build_crowd_shock import compute_crowd_shock
from compute_h2h_conf_form import compute_missing_features

SEED = 42; N_FOLDS = 30; SIGMA = 6.0  # Brier-optimal (Apr 2026 sweep on backfilled data)
TEAM_LOCATIONS_FILE = "ncaa_team_locations.json"

# ── Current 43 features ──
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

# New feature set: swap hca_pts → rolling_hca, add travel_advantage
FEATURES_NEW = [f if f != "hca_pts" else "rolling_hca" for f in FEATURES_43] + ["travel_advantage"]

MODELS = {
    "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, verbose=-1, random_state=SEED),
}

# ══════════════════════════════════════════════════════════
# ROLLING HCA
# ══════════════════════════════════════════════════════════
CONF_HCA = {
    "Big 12": 7.3, "SEC": 7.3, "Big Ten": 6.2, "ACC": 5.8,
    "Big East": 4.6, "Pac-12": 4.2, "Mountain West": 5.0,
    "AAC": 5.2, "WCC": 10.6, "Atlantic 10": 6.4,
    "Missouri Valley": 6.1,
}
ESPN_CONF = {
    "8": "Big 12", "23": "SEC", "7": "Big Ten", "2": "ACC", "4": "Big East",
    "21": "Pac-12", "44": "Mountain West", "62": "AAC", "26": "WCC",
    "3": "Atlantic 10", "18": "Missouri Valley", "22": "Ivy League",
}
DEFAULT_HCA = 6.6; WINDOW = 20; MIN_GAMES = 5

def compute_rolling_hca_col(df):
    df = df.sort_values("game_date_dt").reset_index(drop=True)
    for col in ["home_conference", "away_conference"]:
        if col in df.columns:
            df[f"_{col}_norm"] = df[col].astype(str).map(lambda x: ESPN_CONF.get(x.replace("conf_",""), x))
    
    margin = pd.to_numeric(df["actual_home_score"], errors="coerce") - pd.to_numeric(df["actual_away_score"], errors="coerce")
    neutral = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)
    
    team_home = defaultdict(lambda: deque(maxlen=WINDOW))
    team_away = defaultdict(lambda: deque(maxlen=WINDOW))
    hca = np.full(len(df), DEFAULT_HCA)
    
    for i in range(len(df)):
        home_id = str(df.at[i, "home_team_id"] if "home_team_id" in df.columns else "")
        away_id = str(df.at[i, "away_team_id"] if "away_team_id" in df.columns else "")
        is_n = neutral.iloc[i]
        m = margin.iloc[i]
        conf = df.at[i, "_home_conference_norm"] if "_home_conference_norm" in df.columns else ""
        
        hh = list(team_home[home_id]); ha = list(team_away[home_id])
        if len(hh) >= MIN_GAMES and len(ha) >= MIN_GAMES:
            hca[i] = (np.mean(hh) - np.mean(ha)) / 2
        else:
            hca[i] = CONF_HCA.get(conf, DEFAULT_HCA)
        
        if is_n: hca[i] = 0
        if pd.notna(m) and not is_n:
            team_home[home_id].append(m)
            team_away[away_id].append(-m)
    
    df["rolling_hca"] = hca
    return df

# ══════════════════════════════════════════════════════════
# TRAVEL ADVANTAGE (from venue_city in training data)
# ══════════════════════════════════════════════════════════

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 6371 * 0.621371
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Common venue cities → coords (tournament sites, conference tourney venues)
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
    "Sioux Falls": (43.550, -96.700), "Honolulu": (21.307, -157.858),
    "Aguadilla": (18.427, -67.154), "San Juan": (18.466, -66.106),
    "New York": (40.751, -73.994), "Boston": (42.366, -71.062),
    "Allentown": (40.602, -75.470), "Fargo": (46.877, -96.790),
    "Charleston": (32.784, -79.938), "Ypsilanti": (42.241, -83.613),
    "Kent": (41.154, -81.358), "Ruston": (32.523, -92.638),
    "West Point": (41.391, -73.957), "Hamden": (41.356, -72.897),
    "Los Angeles": (34.052, -118.244), "Annapolis": (38.978, -76.492),
    "Mount Pleasant": (43.598, -84.776), "West Long Branch": (40.291, -74.015),
    "Saint Thomas": (18.338, -64.931), "Washington": (38.898, -77.021),
    "Philadelphia": (39.952, -75.164), "Uncasville": (41.435, -72.111),
    "Bridgeport": (41.187, -73.195), "Frisco": (33.150, -96.824),
    "Sunrise": (26.159, -80.326), "Palm Desert": (33.722, -116.377),
    "Myrtle Beach": (33.689, -78.887), "Lake Buena Vista": (28.388, -81.506),
    "Estero": (26.438, -81.807), "North Augusta": (33.502, -81.965),
    # ── Batch 2: ALL missing venue cities (2,409 games) ──
    "Nassau": (25.048, -77.355), "Asheville": (35.595, -82.551),
    "Daytona Beach": (29.211, -81.023), "Maui": (20.798, -156.332),
    "Norfolk": (36.851, -76.286), "Conway": (33.836, -79.048),
    "Cancun": (21.161, -86.851), "Fort Myers": (26.640, -81.872),
    "Katy": (29.786, -95.817), "Pensacola": (30.421, -87.217),
    "Atlantic City": (39.364, -74.423), "Anaheim": (33.836, -117.914),
    "George Town": (19.295, -81.381), "Evansville": (37.975, -87.556),
    "Niceville": (30.517, -86.482), "Henderson": (36.040, -114.982),
    "Fullerton": (33.870, -117.925), "Kissimmee": (28.292, -81.408),
    "Palm Springs": (33.830, -116.545), "Reno": (39.530, -119.814),
    "Baltimore": (39.290, -76.612), "Seattle": (47.606, -122.332),
    "Anchorage": (61.218, -149.900), "North Charleston": (32.855, -79.975),
    "Savannah": (32.081, -81.091), "El Paso": (31.762, -106.485),
    "Freeport": (26.533, -78.697), "Newark": (40.736, -74.172),
    "Montego Bay": (18.471, -77.922), "Huntsville": (34.730, -86.586),
    "Johnson City": (36.313, -82.354), "Buies Creek": (35.412, -78.733),
    "San Juan Capistrano": (33.502, -117.663), "Laval": (45.570, -73.691),
    "White Sulphur Springs": (37.797, -80.298), "Missoula": (46.872, -113.994),
    "Lake Charles": (30.213, -93.217), "St. Petersburg": (27.771, -82.679),
    "Miami": (25.762, -80.192), "College Park": (38.981, -76.937),
    "Destin": (30.394, -86.496), "Corpus Christi": (27.801, -97.397),
    "Rock Hill": (34.925, -81.025), "Richmond": (37.541, -77.436),
    "Lynchburg": (37.414, -79.142), "Naples": (26.142, -81.795),
    "Bimini": (25.727, -79.265), "Belfast": (54.597, -5.930),
    "Mobile": (30.695, -88.040), "Harrisonburg": (38.450, -78.869),
    "Springfield": (37.215, -93.298), "Santa Clara": (37.354, -121.955),
    "High Point": (35.956, -80.005), "Denton": (33.215, -97.133),
    "Akron": (41.081, -81.519), "Oxford": (34.366, -89.519),
    "Spartanburg": (34.949, -81.932), "Bronx": (40.837, -73.866),
    "Jackson": (32.299, -90.185), "Rochester": (43.157, -77.616),
    "Tempe": (33.425, -111.940), "San Francisco": (37.775, -122.418),
    "Chattanooga": (35.046, -85.309), "Uniondale": (40.724, -73.590),
    "Homewood": (33.472, -86.801), "Santa Cruz": (36.974, -122.031),
    "Boca Raton": (26.359, -80.084), "Cincinnati": (39.103, -84.512),
    "Stockton": (37.958, -121.291), "Logan": (41.735, -111.834),
    "London": (51.507, -0.128), "Bloomington": (39.165, -86.526),
    "West Lafayette": (40.426, -86.908), "Farmville": (37.302, -78.392),
    "Lethbridge": (49.694, -112.833), "Cedar Park": (30.505, -97.820),
    "Cheney": (47.487, -117.576), "Dekalb": (41.930, -88.750),
    "Pearl Harbor": (21.365, -157.974), "Bayamon": (18.399, -66.155),
    "Wilmington": (34.226, -77.945), "Easton": (40.691, -75.221),
    "North Little Rock": (34.769, -92.267), "Vancouver": (49.283, -123.121),
    "Moncton": (46.088, -64.768), "Laie": (21.642, -157.926),
    "Little Rock": (34.747, -92.290), "Colorado Springs": (38.834, -104.821),
    "Highland Heights": (39.036, -84.459), "Melbourne": (28.084, -80.608),
    "Durham": (35.994, -78.899), "Shanghai": (31.230, 121.474),
    "Elmont": (40.724, -73.713), "Inglewood": (33.962, -118.353),
    "Moon Township": (40.519, -80.222), "St. Augustine": (29.895, -81.315),
    "Valparaiso": (41.473, -87.061), "Niagara Falls": (43.094, -79.057),
    "Dublin": (53.350, -6.260), "Trenton": (40.217, -74.742),
    "Youngstown": (41.100, -80.650), "Bowling Green": (36.990, -86.444),
    "South Padre Island": (26.108, -97.168), "Cambridge": (42.373, -71.110),
    "Southaven": (34.992, -90.013), "Boone": (36.217, -81.675),
    "St. George": (37.096, -113.568), "Tupelo": (34.258, -88.703),
    "Syracuse": (43.049, -76.147), "Idaho Falls": (43.467, -112.034),
    "Bossier City": (32.516, -93.732), "Kennesaw": (34.023, -84.615),
    "Nacogdoches": (31.603, -94.655), "Greeley": (40.423, -104.709),
    "Albuquerque": (35.084, -106.651), "Troy": (31.809, -85.970),
    "Hot Springs": (34.502, -93.055), "Biloxi": (30.396, -88.885),
    "Fort Wayne": (41.079, -85.139), "Northridge": (34.229, -118.537),
    "Hickory": (35.733, -81.341), "Flagstaff": (35.198, -111.651),
    "Princeton": (40.349, -74.659), "Jonesboro": (35.842, -90.704),
    "Xenia": (39.685, -83.930), "Toledo": (41.664, -83.556),
    "Shreveport": (32.525, -93.750), "Rio Rancho": (35.233, -106.664),
    "West Hartford": (41.762, -72.742), "New Britain": (41.661, -72.780),
    "Towson": (39.402, -76.602), "Jersey City": (40.728, -74.078),
    "Smithfield": (41.922, -71.549), "Elon": (36.103, -79.507),
    "Kennewick": (46.211, -119.137), "Malibu": (34.026, -118.780),
    "Dothan": (31.223, -85.390), "Champaign": (40.116, -88.243),
    "Arlington": (32.736, -97.108), "Manhattan": (39.184, -96.572),
    "New Haven": (41.308, -72.928), "Lincoln": (40.814, -96.703),
    "Ogden": (41.223, -111.974), "Hershey": (40.286, -76.651),
    "Kalamazoo": (42.292, -85.587), "Abilene": (32.449, -99.733),
    "Natchitoches": (31.761, -93.087), "Bozeman": (45.680, -111.038),
    "Fresno": (36.738, -119.787), "Terre Haute": (39.467, -87.414),
    "Cookeville": (36.163, -85.502), "Hackensack": (40.886, -74.043),
    "Long Beach": (33.770, -118.194), "Wilkes Barre Township": (41.246, -75.881),
    "East Rutherford": (40.834, -74.097), "Macomb": (40.461, -90.672),
    "Saint Charles": (38.784, -90.481), "Moline": (41.507, -90.515),
    "Florence": (34.180, -79.764), "Hoffman Estates": (42.063, -88.123),
    "Fairfield": (41.141, -73.264), "Corvallis": (44.565, -123.261),
    "Stephenville": (32.221, -98.203), "Billings": (45.784, -108.510),
    "Vermillion": (42.780, -96.929), "Marietta": (33.953, -84.549),
    "Queens": (40.728, -73.795), "San Luis Obispo": (35.283, -120.660),
    "Fort Bliss": (31.812, -106.415), "Edinburg": (26.302, -98.163),
    "Alexandria": (31.311, -92.445), "Auburn Hills": (42.688, -83.235),
    "Grand Forks": (47.925, -97.033), "Ramstein": (49.437, 7.600),
    "Reading": (40.336, -75.927), "Tuba City": (36.135, -111.240),
    "Union": (40.698, -74.263), "Worcester": (42.263, -71.802),
    "Santa Barbara": (34.421, -119.697), "Williamsburg": (37.271, -76.707),
    "Tuscaloosa": (33.210, -87.569), "Lakeland": (28.040, -81.950),
    "Riverside": (33.953, -117.396), "Glens Falls": (43.310, -73.644),
    "Garland": (32.912, -96.639), "Iowa City": (41.661, -91.530),
    "Fort Hood": (31.138, -97.776), "Grambling": (32.528, -92.714),
}

def compute_travel_col(df, locations):
    """Compute travel_advantage from venue_city in training data."""
    neutral = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool)
    travel = np.zeros(len(df))
    
    if locations is None:
        print("  ⚠ No team locations — travel_advantage = 0")
        df["travel_advantage"] = travel
        return df
    
    # Build team lookup: id → coords, abbr → coords
    id_coords = {}
    for tid, info in locations.items():
        if info.get("lat"):
            id_coords[tid] = (info["lat"], info["lng"])
    
    n_neutral = neutral.sum()
    n_computed = 0; n_missing = 0; n_no_venue = 0
    
    venue_city_col = "venue_city" if "venue_city" in df.columns else None
    if not venue_city_col:
        print("  ⚠ No venue_city column — travel_advantage = 0")
        df["travel_advantage"] = travel
        return df
    
    for i in range(len(df)):
        if not neutral.iloc[i]:
            continue
        
        home_id = str(df.at[i, "home_team_id"]) if "home_team_id" in df.columns else ""
        away_id = str(df.at[i, "away_team_id"]) if "away_team_id" in df.columns else ""
        
        home_loc = id_coords.get(home_id)
        away_loc = id_coords.get(away_id)
        
        if not home_loc or not away_loc:
            n_missing += 1
            continue
        
        # Get venue coords from venue_city
        city = df.at[i, venue_city_col]
        if not city or pd.isna(city):
            n_no_venue += 1
            continue
        
        venue_coords = VENUE_CITIES.get(str(city))
        if not venue_coords:
            n_no_venue += 1
            continue
        
        hd = haversine_miles(home_loc[0], home_loc[1], venue_coords[0], venue_coords[1])
        ad = haversine_miles(away_loc[0], away_loc[1], venue_coords[0], venue_coords[1])
        hd = max(hd, 1); ad = max(ad, 1)
        travel[i] = math.log(ad / hd)
        n_computed += 1
    
    df["travel_advantage"] = travel
    
    print(f"  Neutral: {n_neutral:,} | Computed: {n_computed:,} | No team loc: {n_missing:,} | No venue city: {n_no_venue:,}")
    if n_computed > 0:
        ta = travel[travel != 0]
        print(f"  Stats: mean={ta.mean():+.3f}, std={ta.std():.3f}, range=[{ta.min():.2f}, {ta.max():.2f}]")
    
    # Print missing venue cities for future addition
    if n_no_venue > 0:
        missing_cities = set()
        for i in range(len(df)):
            if neutral.iloc[i]:
                city = df.at[i, venue_city_col]
                if city and not pd.isna(city) and str(city) not in VENUE_CITIES:
                    missing_cities.add(str(city))
        if missing_cities:
            print(f"  Missing venue cities ({len(missing_cities)}): {sorted(missing_cities)[:20]}")
    
    return df

# ══════════════════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════════════════

def walk_forward(X_s, y, n_folds, weights=None):
    n = len(X_s); fold_size = n // (n_folds + 1); min_train = fold_size * 2
    oof = np.full(n, np.nan)
    for fold in range(n_folds):
        ts = min_train + fold * fold_size; te = min(ts + fold_size, n)
        if ts >= n: break
        preds = []
        for name, builder in MODELS.items():
            m = builder()
            if weights is not None:
                try:
                    m.fit(X_s[:ts], y[:ts], sample_weight=weights[:ts])
                except TypeError:
                    m.fit(X_s[:ts], y[:ts])
            else:
                m.fit(X_s[:ts], y[:ts])
            preds.append(m.predict(X_s[ts:te]))
        oof[ts:te] = np.mean(preds, axis=0)
    return oof

def eval_model(oof, y, spreads, label):
    valid = ~np.isnan(oof) & (np.abs(spreads) > 0.5)
    pred = oof[valid]; act = y[valid]; sp = spreads[valid]
    mae = np.mean(np.abs(pred - act))
    prob_home = 1.0 / (1.0 + np.exp(-pred / SIGMA))
    hw = (act > 0).astype(float)
    brier = np.mean((prob_home - hw) ** 2)
    
    ats = pred + sp; covers = (ats > 0) == (act + sp > 0)
    disagree = np.abs(pred + sp)
    for thresh, name in [(4, "4+"), (7, "7+"), (10, "10+")]:
        mask = disagree >= thresh
        if mask.sum() > 0:
            print(f"    ATS@{name}: {covers[mask].mean():.1%} on {mask.sum()} picks")
    print(f"    MAE: {mae:.3f}, Brier: {brier:.4f} (σ={SIGMA})")
    return mae, brier

def spread_to_ml(spread):
    s = abs(spread)
    if s < 0.5: return -110
    for lim, f in [(1,-120),(2,-140),(3,-160),(4,-185),(5,-210),(6,-245),(7,-280),(8,-320),(9,-370),(10,-420),(12,-550),(14,-700),(16,-900),(18,-1200),(20,-1500)]:
        if s <= lim: return f if spread < 0 else -f
    return -2000 if spread < 0 else 2000

def ml_to_decimal(ml):
    return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NCAA FINAL RETRAIN — All Fixes Combined")
    print(f"  σ={SIGMA} | Rolling HCA | Travel Advantage | Fixed neutral_em_diff")
    print("=" * 70)
    
    # ── Load data ──
    df = load_cached()
    if df is None: df = dump()
    df = df[df["actual_home_score"].notna()].copy()
    
    # ── Fix Arrow string types from parquet/CSV merge ──
    # Force all non-text columns to numeric
    _text_cols = {"home_team_name","away_team_name","home_team_abbr","away_team_abbr",
                  "home_conference","away_conference","referee_1","referee_2","referee_3",
                  "game_date","venue_name","venue_city","venue_state","game_id"}
    for col in df.columns:
        if col in _text_cols:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
    df = df[~df["season"].isin([2020, 2021])].copy()
    df["game_date_dt"] = pd.to_datetime(df.get("game_date", ""), errors="coerce")
    season_mask = (df["game_date_dt"].dt.month >= 11) | (df["game_date_dt"].dt.month <= 4)
    early_mask = ~((df["game_date_dt"].dt.month == 11) & (df["game_date_dt"].dt.day < 10))
    df = df[season_mask & early_mask].copy()
    
    # ── Force numeric types for spread/total columns (parquet may have mixed Arrow types) ──
    for col in ["market_spread_home", "market_ou_total", "espn_spread", "espn_over_under",
                "dk_spread_close", "dk_spread_open", "dk_total_close", "dk_total_open",
                "odds_api_spread_close", "odds_api_spread_open", "odds_api_total_close", "odds_api_total_open"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # ── Cascade spread backfill: ESPN → DraftKings → Odds API ──
    # Different seasons have different spread sources available
    spread_sources = [
        ("espn_spread", "espn_over_under"),
        ("dk_spread_close", "dk_total_close"),
        ("dk_spread_open", "dk_total_open"),
        ("odds_api_spread_close", "odds_api_total_close"),
        ("odds_api_spread_open", "odds_api_total_open"),
    ]
    
    for spread_col, total_col in spread_sources:
        if spread_col in df.columns:
            src_s = pd.to_numeric(df[spread_col], errors="coerce")
            mkt_s = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
            fill = (mkt_s.isna() | (mkt_s == 0)) & src_s.notna() & (src_s != 0)
            if fill.sum() > 0:
                df.loc[fill, "market_spread_home"] = src_s[fill]
                print(f"  Spread backfill from {spread_col}: {fill.sum():,} rows")
        
        if total_col in df.columns:
            src_ou = pd.to_numeric(df[total_col], errors="coerce")
            mkt_ou = pd.to_numeric(df.get("market_ou_total", pd.Series(dtype=float)), errors="coerce")
            fill_ou = (mkt_ou.isna() | (mkt_ou == 0)) & src_ou.notna() & (src_ou != 0)
            if fill_ou.sum() > 0:
                df.loc[fill_ou, "market_ou_total"] = src_ou[fill_ou]
    
    # Report final spread coverage per season
    mkt_final = pd.to_numeric(df.get("market_spread_home", pd.Series(dtype=float)), errors="coerce")
    has_spread = mkt_final.notna() & (mkt_final != 0)
    for season in sorted(df["season"].unique()):
        sm = df["season"] == season
        print(f"  {season}: {has_spread[sm].sum()}/{sm.sum()} spread coverage ({has_spread[sm].mean():.0%})")
    
    # Quality filter — require core stats but adj_em is optional (not available for backfilled games)
    _qcols = [c for c in ["home_ppg","away_ppg","market_spread_home","market_ou_total"] if c in df.columns]
    _qmat = pd.DataFrame({c: df[c].notna() & (df[c] != 0 if c in ["market_spread_home","market_ou_total"] else True) for c in _qcols})
    _keep = _qmat.mean(axis=1) >= 0.75  # 3 of 4 core stats required
    if "referee_1" in df.columns:
        _keep = _keep & df["referee_1"].notna() & (df["referee_1"] != "")
    df = df.loc[_keep].reset_index(drop=True)
    
    for col in ["actual_home_score","actual_away_score","season","home_record_wins","away_record_wins","home_record_losses","away_record_losses"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for s, d in [("home_record_wins","home_wins"),("away_record_wins","away_wins"),("home_record_losses","home_losses"),("away_record_losses","away_losses")]:
        if s in df.columns and d not in df.columns: df[d] = df[s]
    
    # ── Remove games involving teams with <10 total appearances (noise) ──
    MIN_TEAM_GAMES = 10
    home_counts = df["home_team_id"].astype(str).value_counts()
    away_counts = df["away_team_id"].astype(str).value_counts()
    team_total = home_counts.add(away_counts, fill_value=0)
    small_teams = set(team_total[team_total < MIN_TEAM_GAMES].index)
    small_mask = df["home_team_id"].astype(str).isin(small_teams) | df["away_team_id"].astype(str).isin(small_teams)
    n_before = len(df)
    df = df[~small_mask].reset_index(drop=True)
    print(f"\n  Removed {small_mask.sum()} games with <{MIN_TEAM_GAMES}-game teams ({small_mask.sum()/n_before:.1%})")
    
    print(f"  {len(df):,} games after filters")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    print(f"  Latest game: {df['game_date_dt'].max().strftime('%Y-%m-%d')}")
    
    # ── Standard features ──
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
    
    # ── Rolling HCA ──
    print("\n  Computing rolling HCA...")
    df = compute_rolling_hca_col(df)
    rhca = df["rolling_hca"]
    print(f"  Rolling HCA: mean={rhca[rhca > 0].mean():+.2f}, coverage={((rhca != 0) | df.get('neutral_site', False).fillna(False).astype(bool)).mean():.1%}")
    
    # ── Travel advantage ──
    print("\n  Computing travel advantage from venue_city...")
    locations = json.load(open(TEAM_LOCATIONS_FILE)) if os.path.exists(TEAM_LOCATIONS_FILE) else None
    df = compute_travel_col(df, locations)
    
    # ── Build features ──
    print("\n  Building feature matrix...")
    X_full = ncaa_build_features(df)
    X_full["rolling_hca"] = df["rolling_hca"].values
    X_full["travel_advantage"] = df["travel_advantage"].values
    
    y = df["actual_home_score"].values - df["actual_away_score"].values
    spreads = pd.to_numeric(df.get("market_spread_home", 0), errors="coerce").fillna(0).values
    
    # Season weights: 1.0/0.9/0.75/0.6/0.5 by age (validated best ATS@7+ = 91.1%)
    current_year = 2026
    def _season_weight(s):
        age = current_year - s
        if age <= 0: return 1.0
        if age == 1: return 0.9
        if age == 2: return 0.75
        if age == 3: return 0.6
        return 0.5
    
    seasons = df["season"].values
    weights = np.array([_season_weight(s) for s in seasons])
    print(f"  Season weights: {dict(zip(sorted(set(seasons)), [_season_weight(s) for s in sorted(set(seasons))]))}")
    
    # ══════════════════════════════════════════════════════════
    # TEST 1: Baseline (old features, old neutral_em_diff)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TEST 1: BASELINE (static HCA, old neutral_em_diff, σ={SIGMA})")
    print(f"{'='*70}")
    
    old_em = X_full["neutral_em_diff"].copy() if "neutral_em_diff" in X_full.columns else None
    avail_43 = [f for f in FEATURES_43 if f in X_full.columns]
    X_base = X_full[avail_43].copy()
    if old_em is not None: X_base["neutral_em_diff"] = old_em
    oof1 = walk_forward(StandardScaler().fit_transform(X_base.values), y, N_FOLDS, weights)
    mae1, brier1 = eval_model(oof1, y, spreads, "Baseline")
    
    # ── Fix neutral_em_diff ──
    if "neutral_em_diff" in X_full.columns:
        raw_em = df["home_adj_em"].fillna(0).values - df["away_adj_em"].fillna(0).values
        neutral_mask = df.get("neutral_site", pd.Series(False)).fillna(False).astype(bool).values
        rolling = df["rolling_hca"].values
        X_full["neutral_em_diff"] = raw_em - np.where(neutral_mask, 0, rolling)
        shift = np.abs(X_full["neutral_em_diff"].values - old_em.values).mean() if old_em is not None else 0
        print(f"\n  Fixed neutral_em_diff: avg shift {shift:.2f} pts")
    
    # ══════════════════════════════════════════════════════════
    # TEST 2: All fixes (rolling HCA + fixed em + travel)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TEST 2: ALL FIXES (rolling HCA + fixed em + travel, σ={SIGMA})")
    print(f"{'='*70}")
    
    avail_new = [f for f in FEATURES_NEW if f in X_full.columns]
    X_new = X_full[avail_new]
    print(f"  Features: {len(avail_new)}")
    oof2 = walk_forward(StandardScaler().fit_transform(X_new.values), y, N_FOLDS, weights)
    mae2, brier2 = eval_model(oof2, y, spreads, "All Fixes")
    
    # ══════════════════════════════════════════════════════════
    # LASSO COEFFICIENTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  LASSO COEFFICIENTS")
    print(f"{'='*70}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new.values)
    lasso = Lasso(alpha=0.1, max_iter=5000)
    lasso.fit(X_scaled, y, sample_weight=weights)
    coefs = dict(zip(avail_new, lasso.coef_))
    
    for f in ["rolling_hca", "travel_advantage"]:
        if f in coefs:
            status = "✅ KEPT" if abs(coefs[f]) > 0.001 else "❌ DROPPED"
            print(f"  {f:>25s}: {coefs[f]:+.4f} {status}")
    
    kept = {k: v for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.001}
    print(f"\n  Lasso kept {len(kept)}/{len(avail_new)} features")
    for f, c in kept.items():
        marker = " ← NEW" if f in ["rolling_hca", "travel_advantage"] else ""
        print(f"    {f:>30s}: {c:+.4f}{marker}")
    
    # ══════════════════════════════════════════════════════════
    # PARLAY CONFIDENCE GATE SWEEP (σ=6.0)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PARLAY GATE SWEEP at σ={SIGMA}")
    print(f"  3+5 stack, ML cap -500, $50/$25")
    print(f"{'='*70}")
    
    valid = ~np.isnan(oof2) & (np.abs(spreads) > 0.1)
    pred = oof2[valid]; act = y[valid]; sp = spreads[valid]
    seasons = df["season"].values[valid]
    dates = df["game_date_dt"].values[valid]
    
    prob_home = 1.0 / (1.0 + np.exp(-pred / SIGMA))
    conf = np.maximum(prob_home, 1 - prob_home)
    pick_home = pred > 0
    actual_home_win = act > 0
    not_tie = act != 0
    pick_won = (pick_home == actual_home_win) & not_tie
    
    # ML odds
    pick_ml = np.array([spread_to_ml(s if ph else -s) for s, ph in zip(sp, pick_home)])
    pick_dec = np.array([ml_to_decimal(ml) for ml in pick_ml])
    ml_ok = (pick_ml > -500) | (pick_ml > 0)
    
    date_strs = pd.to_datetime(pd.Series(dates)).dt.strftime("%Y-%m-%d").values
    
    print(f"\n  {'Gate':>5s} {'Days':>5s} {'3L W/L':>8s} {'3L Hit':>7s} {'5L W/L':>8s} {'5L Hit':>7s} {'Wagered':>9s} {'Profit':>9s} {'ROI':>7s}")
    print(f"  {'-'*72}")
    
    test_seasons = [2019, 2022, 2023, 2024, 2025, 2026]
    
    for gate in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        tw = 0; tr = 0; w3 = 0; l3 = 0; w5 = 0; l5 = 0; days = 0
        
        for season in test_seasons:
            smask = seasons == season
            s_dates = date_strs[smask]; s_conf = conf[smask]; s_won = pick_won[smask]
            s_dec = pick_dec[smask]; s_ml_ok = ml_ok[smask]; s_tie = not_tie[smask]
            
            for date in sorted(set(s_dates)):
                dm = (s_dates == date) & s_tie & (s_conf >= gate) & s_ml_ok
                di = np.where(dm)[0]
                if len(di) < 3: continue
                si = di[np.argsort(-s_conf[di])]
                days += 1
                
                # 3-leg
                top3 = si[:3]; pd3 = 1.0; aw3 = True
                for idx in top3:
                    pd3 *= s_dec[idx]
                    if not s_won[idx]: aw3 = False
                tw += 50
                if aw3: tr += 50 * pd3; w3 += 1
                else: l3 += 1
                
                # 5-leg (check month for seasonal override)
                month = pd.to_datetime(date).month
                if month in [1, 2]: continue  # Jan/Feb: 3-leg only
                if month == 1 and pd.to_datetime(date).day <= 7: continue  # Jan W1: skip
                
                if len(si) >= 5:
                    top5 = si[:5]; pd5 = 1.0; aw5 = True
                    for idx in top5:
                        pd5 *= s_dec[idx]
                        if not s_won[idx]: aw5 = False
                    tw += 25
                    if aw5: tr += 25 * pd5; w5 += 1
                    else: l5 += 1
        
        profit = tr - tw
        roi = profit / tw * 100 if tw > 0 else 0
        h3 = w3 / max(w3 + l3, 1) * 100
        h5 = w5 / max(w5 + l5, 1) * 100
        tag = "✅" if profit > 0 else "❌"
        n5 = w5 + l5
        h5_str = f"{h5:>6.1f}%" if n5 > 0 else "   n/a"
        print(f"  {gate:>5.0%} {days:>5d} {w3:>3d}/{l3:<3d} {h3:>6.1f}% {w5:>3d}/{l5:<3d} {h5_str} ${tw:>8,d} ${profit:>+8,.0f} {roi:>+6.1f}% {tag}")
    
    # ══════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20s} {'Baseline':>10s} {'All Fixes':>10s} {'Change':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'MAE':<20s} {mae1:>10.3f} {mae2:>10.3f} {mae2-mae1:>+10.3f}")
    print(f"  {'Brier (σ=6)':<20s} {brier1:>10.4f} {brier2:>10.4f} {brier2-brier1:>+10.4f}")
    
    if args.upload:
        print(f"\n  Upload requested — training final model on full data...")
        # Train on ALL data
        lasso_final = Lasso(alpha=0.1, max_iter=5000)
        lgbm_final = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, verbose=-1, random_state=SEED)
        try:
            lasso_final.fit(X_scaled, y, sample_weight=weights)
        except TypeError:
            lasso_final.fit(X_scaled, y)
        lgbm_final.fit(X_scaled, y, sample_weight=weights)
        
        # SHAP explainer (from LGBM model)
        import shap
        explainer = shap.TreeExplainer(lgbm_final)
        
        bundle = {
            "scaler": scaler,
            "models": [lasso_final, lgbm_final],
            "feature_names": avail_new,
            "feature_cols": avail_new,  # alias for compatibility
            "model_type": "Lasso_LGBM_v31_HCA",
            "mae_cv": mae2,
            "sigma": SIGMA,
            "trained_at": datetime.utcnow().isoformat(),
            "n_train": len(y),
            "n_features": len(avail_new),
            "explainer": explainer,
        }
        
        print(f"  {len(avail_new)} features, {len(y):,} training games")
        
        from db import save_model
        save_model("ncaa", bundle)
        print(f"  ✅ Uploaded as 'ncaa' to model_store")
        print(f"  Reload: curl -X POST https://sports-predictor-api-production.up.railway.app/debug/reload-model/ncaa")
    
    print(f"\n  Done.")
