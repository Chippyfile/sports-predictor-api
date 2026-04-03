#!/usr/bin/env python3
"""
build_travel_distance.py — NCAA neutral site proximity feature
==============================================================
1. Scrapes team locations (lat/lng) from ESPN team API
2. Gets venue coordinates from ESPN game data  
3. Computes haversine distance from each team to game venue
4. Creates travel_advantage feature for neutral site games

Usage:
    python3 build_travel_distance.py --build-teams    # Step 1: Build team location DB
    python3 build_travel_distance.py --backfill       # Step 2: Backfill historical games
    python3 build_travel_distance.py --test           # Test on today's games
"""
import sys, os, json, time, math, argparse
sys.path.insert(0, '.')

import requests
import numpy as np
import pandas as pd

TEAM_LOCATIONS_FILE = "ncaa_team_locations.json"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# ── Haversine distance (km) ──
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_miles(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2) * 0.621371


# ══════════════════════════════════════════════════════════
# STEP 1: Build team location database from ESPN
# ══════════════════════════════════════════════════════════

def build_team_locations():
    """Build team lat/lng from ESPN location field + city geocoder."""
    print("=" * 60)
    print("  BUILDING TEAM LOCATION DATABASE")
    print("=" * 60)

    # Get all teams from bulk endpoint (has location field)
    print("\n  Fetching team list...")
    try:
        resp = requests.get(f"{ESPN_BASE}/teams?limit=500", timeout=30)
        data = resp.json()
        teams_raw = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        print(f"  Found {len(teams_raw)} teams")
    except Exception as e:
        print(f"  Failed: {e}")
        return {}

    locations = {}
    found = 0
    city_match = 0
    state_match = 0
    missing = []

    for entry in teams_raw:
        t = entry.get("team", {})
        tid = str(t.get("id", ""))
        abbr = t.get("abbreviation", "")
        name = t.get("displayName", "")
        loc = t.get("location", "")  # e.g. "Auburn", "Duke", "North Carolina"

        if not tid:
            continue

        lat, lng = None, None

        # Try direct city match
        if loc in US_CITY_COORDS:
            lat, lng = US_CITY_COORDS[loc]
            city_match += 1
        else:
            # Try matching location to known cities (partial match)
            for city_name, coords in US_CITY_COORDS.items():
                if city_name.lower() in loc.lower() or loc.lower() in city_name.lower():
                    lat, lng = coords
                    city_match += 1
                    break

        if lat is not None:
            found += 1

        locations[tid] = {
            "abbr": abbr, "name": name, "location": loc,
            "lat": lat, "lng": lng,
        }

        if lat is None:
            missing.append(f"{abbr} ({loc})")

    # Save
    with open(TEAM_LOCATIONS_FILE, "w") as f:
        json.dump(locations, f, indent=2)

    print(f"\n  Saved {len(locations)} teams")
    print(f"  {found} with coordinates ({city_match} city matches)")
    print(f"  {len(locations) - found} missing")

    if missing:
        print(f"\n  Missing locations ({len(missing)}):")
        for m in sorted(missing)[:30]:
            print(f"    {m}")
        if len(missing) > 30:
            print(f"    ... and {len(missing)-30} more")

    print(f"\n  To improve coverage, add missing cities to US_CITY_COORDS in the script.")
    print(f"  Or run: python3 build_travel_distance.py --fill-missing")

    return locations


# ── Simple geocoder for US college cities ──
# Major cities pre-loaded, others estimated from state centroid
# NCAA school location → coords (school city name from ESPN "location" field)
# Covers all Power 5, major mid-majors, and most D1 programs
US_CITY_COORDS = {
    # ── ACC ──
    "Boston College": (42.335, -71.169), "Clemson": (34.683, -82.837),
    "Duke": (36.001, -78.939), "Florida State": (30.438, -84.281),
    "Georgia Tech": (33.772, -84.393), "Louisville": (38.253, -85.759),
    "Miami": (25.762, -80.192), "North Carolina": (35.905, -79.047),
    "NC State": (35.787, -78.644), "Notre Dame": (41.705, -86.235),
    "Pittsburgh": (40.441, -79.996), "Syracuse": (43.039, -76.136),
    "Virginia": (38.033, -78.508), "Virginia Tech": (37.228, -80.424),
    "Wake Forest": (36.135, -80.279), "California": (37.872, -122.259),
    "SMU": (32.843, -96.783), "Stanford": (37.429, -122.170),
    # ── Big 12 ──
    "Arizona": (32.228, -110.949), "Arizona State": (33.424, -111.928),
    "Baylor": (31.549, -97.114), "BYU": (40.252, -111.649),
    "UCF": (28.602, -81.200), "Cincinnati": (39.131, -84.515),
    "Colorado": (40.007, -105.265), "Houston": (29.722, -95.343),
    "Iowa State": (42.027, -93.635), "Kansas": (38.954, -95.252),
    "Kansas State": (39.187, -96.572), "Oklahoma State": (36.126, -97.070),
    "TCU": (32.710, -97.363), "Texas Tech": (33.584, -101.845),
    "Utah": (40.765, -111.843), "West Virginia": (39.636, -79.955),
    # ── Big East ──
    "Butler": (39.839, -86.169), "UConn": (41.808, -72.249),
    "Connecticut": (41.808, -72.249),
    "Creighton": (41.257, -95.935), "DePaul": (41.878, -87.640),
    "Georgetown": (38.908, -77.073), "Marquette": (43.039, -87.930),
    "Providence": (41.824, -71.413), "Seton Hall": (40.742, -74.246),
    "St. John's": (40.724, -73.794), "Villanova": (40.038, -75.346),
    "Xavier": (39.150, -84.473),
    # ── Big Ten ──
    "Illinois": (40.102, -88.227), "Indiana": (39.168, -86.526),
    "Iowa": (41.661, -91.530), "Maryland": (38.986, -76.944),
    "Michigan": (42.281, -83.743), "Michigan State": (42.724, -84.481),
    "Minnesota": (44.974, -93.228), "Nebraska": (40.820, -96.700),
    "Northwestern": (42.060, -87.675), "Ohio State": (40.007, -83.031),
    "Oregon": (44.046, -123.074), "Penn State": (40.799, -77.860),
    "Purdue": (40.424, -86.921), "Rutgers": (40.501, -74.449),
    "UCLA": (34.071, -118.445), "USC": (34.022, -118.286),
    "Washington": (47.655, -122.303), "Wisconsin": (43.076, -89.412),
    # ── SEC ──
    "Alabama": (33.209, -87.569), "Arkansas": (36.068, -94.175),
    "Auburn": (32.607, -85.488), "Florida": (29.650, -82.343),
    "Georgia": (33.948, -83.373), "Kentucky": (38.030, -84.504),
    "LSU": (30.413, -91.180), "Mississippi State": (33.456, -88.790),
    "Missouri": (38.940, -92.328), "Oklahoma": (35.208, -97.443),
    "Ole Miss": (34.360, -89.534), "South Carolina": (33.993, -81.025),
    "Tennessee": (35.955, -83.930), "Texas": (30.284, -97.733),
    "Texas A&M": (30.611, -96.340), "Vanderbilt": (36.144, -86.803),
    # ── AAC ──
    "East Carolina": (35.607, -77.367), "FAU": (26.369, -80.102),
    "Memphis": (35.149, -90.049), "Navy": (38.984, -76.489),
    "North Texas": (33.210, -97.148), "Rice": (29.717, -95.402),
    "South Florida": (28.061, -82.413), "Temple": (39.981, -75.155),
    "Tulane": (29.940, -90.121), "Tulsa": (36.154, -95.993),
    "UAB": (33.502, -86.806), "UTSA": (29.582, -98.619),
    "Charlotte": (35.308, -80.733), "Wichita State": (37.721, -97.296),
    # ── Mountain West ──
    "Air Force": (38.998, -104.862), "Boise State": (43.603, -116.203),
    "Colorado State": (40.574, -105.085), "Fresno State": (36.813, -119.748),
    "Nevada": (39.544, -119.816), "New Mexico": (35.084, -106.619),
    "San Diego State": (32.775, -117.071), "San José State": (37.335, -121.881),
    "UNLV": (36.108, -115.143), "Utah State": (41.745, -111.810),
    "Wyoming": (41.313, -105.588),
    # ── WCC ──
    "Gonzaga": (47.667, -117.403), "Loyola Marymount": (33.969, -118.418),
    "Pacific": (37.979, -121.311), "Pepperdine": (34.045, -118.709),
    "Portland": (45.505, -122.675), "Saint Mary's": (37.840, -122.108),
    "San Diego": (32.771, -117.188), "San Francisco": (37.776, -122.451),
    "Santa Clara": (37.349, -121.939),
    # ── A10 ──
    "Davidson": (35.500, -80.844), "Dayton": (39.740, -84.179),
    "Duquesne": (40.436, -79.991), "Fordham": (40.861, -73.886),
    "George Mason": (38.832, -77.309), "George Washington": (38.900, -77.049),
    "La Salle": (40.038, -75.155), "Loyola Chicago": (41.999, -87.658),
    "Massachusetts": (42.387, -72.530), "Rhode Island": (41.485, -71.527),
    "Richmond": (37.575, -77.540), "Saint Joseph's": (40.001, -75.237),
    "St. Bonaventure": (42.078, -78.480), "VCU": (37.549, -77.453),
    # ── MVC ──
    "Belmont": (36.133, -86.795), "Bradley": (40.698, -89.615),
    "Drake": (41.601, -93.654), "Evansville": (37.978, -87.535),
    "Illinois State": (40.510, -88.998), "Indiana State": (39.470, -87.407),
    "Missouri State": (37.198, -93.282), "Murray State": (36.612, -88.318),
    "Northern Iowa": (42.514, -92.462), "Southern Illinois": (37.714, -89.220),
    "UIC": (41.869, -87.648), "Valparaiso": (41.462, -87.040),
    # ── CAA / Patriot / others ──
    "College of Charleston": (32.783, -79.937), "Delaware": (39.678, -75.752),
    "Drexel": (39.957, -75.190), "Hofstra": (40.714, -73.601),
    "James Madison": (38.434, -78.870), "Northeastern": (42.340, -71.089),
    "Stony Brook": (40.912, -73.123), "Towson": (39.394, -76.607),
    "William & Mary": (37.272, -76.719), "UNCW": (34.226, -77.873),
    "Army": (41.391, -73.957), "Bucknell": (40.955, -76.884),
    "Colgate": (42.818, -75.539), "Holy Cross": (42.237, -71.810),
    "Lafayette": (40.697, -75.212), "Lehigh": (40.607, -75.377),
    # ── Ivy ──
    "Brown": (41.827, -71.403), "Columbia": (40.808, -73.962),
    "Cornell": (42.447, -76.483), "Dartmouth": (43.703, -72.289),
    "Harvard": (42.377, -71.117), "Penn": (39.951, -75.193),
    "Princeton": (40.349, -74.652), "Yale": (41.311, -72.924),
    # ── Other notable ──
    "Gonzaga": (47.667, -117.403), "Saint Mary's (CA)": (37.840, -122.108),
    "Oral Roberts": (36.060, -95.945), "Liberty": (37.353, -79.173),
    "Furman": (34.925, -82.440), "Chattanooga": (35.046, -85.309),
    "UAB": (33.502, -86.806), "Middle Tennessee": (35.850, -86.369),
    "Western Kentucky": (36.987, -86.453), "Marshall": (38.423, -82.425),
    "Old Dominion": (36.886, -76.305), "UTEP": (31.772, -106.505),
    "New Mexico State": (32.282, -106.748), "Louisiana Tech": (32.525, -92.648),
    "Southern Miss": (31.327, -89.335), "North Carolina A&T": (36.069, -79.776),
    "Hampton": (37.022, -76.336), "Norfolk State": (36.885, -76.260),
    "Iona": (40.932, -73.886), "Marist": (41.727, -73.932),
    "Siena": (42.719, -73.752), "Fairfield": (41.173, -73.256),
    "Manhattan": (40.890, -73.901), "Niagara": (43.137, -79.016),
    "Canisius": (42.937, -78.838), "Rider": (40.278, -74.753),
    "Monmouth": (40.277, -74.006), "Quinnipiac": (41.419, -72.893),
    "Sacred Heart": (41.188, -73.188), "Wagner": (40.617, -74.095),
    "Central Connecticut": (41.680, -72.779), "Fairleigh Dickinson": (40.909, -74.121),
    "LIU": (40.689, -73.978), "Merrimack": (42.712, -71.121),
    "Robert Morris": (40.519, -80.184), "Saint Francis (PA)": (40.503, -78.637),
    "Mount St. Mary's": (39.697, -77.371), "St. Francis Brooklyn": (40.697, -73.976),
    # ── Sun Belt / CUSA / others ──
    "Appalachian State": (36.215, -81.688), "Coastal Carolina": (33.793, -79.012),
    "Georgia State": (33.753, -84.385), "Georgia Southern": (32.422, -81.783),
    "Louisiana": (30.210, -92.022), "Louisiana-Monroe": (32.529, -92.076),
    "Arkansas State": (35.842, -90.676), "South Alabama": (30.696, -88.183),
    "Texas State": (29.888, -97.944), "Troy": (31.798, -85.968),
    "ULM": (32.529, -92.076),
    "FIU": (25.756, -80.373), "Sam Houston": (30.714, -95.544),
    "Kennesaw State": (34.038, -84.581), "Jacksonville State": (33.822, -85.764),
    "Lipscomb": (36.176, -86.808), "FGCU": (26.463, -81.770),
    "North Alabama": (34.783, -87.685), "Stetson": (29.045, -81.303),
    "Austin Peay": (36.534, -87.359), "Eastern Kentucky": (37.751, -84.295),
    "Morehead State": (38.185, -83.435), "Tennessee Tech": (36.178, -85.508),
    "Tennessee State": (36.168, -86.823), "Southeast Missouri": (37.311, -89.571),
    "SIU Edwardsville": (38.794, -89.998), "UT Martin": (36.343, -88.864),
    "Eastern Illinois": (39.479, -88.175), "Lindenwood": (38.788, -90.683),
    "Little Rock": (34.747, -92.276), "Southern Indiana": (38.011, -87.572),
    "Western Illinois": (40.477, -90.682),
    # ── MEAC / SWAC / others ──
    "Alcorn State": (31.877, -90.898), "Alabama A&M": (34.783, -86.568),
    "Alabama State": (32.364, -86.296), "Bethune-Cookman": (29.185, -81.063),
    "Coppin State": (39.311, -76.648), "Delaware State": (39.186, -75.540),
    "Florida A&M": (30.425, -84.281), "Howard": (38.922, -77.020),
    "Jackson State": (32.297, -90.209), "Maryland Eastern Shore": (38.481, -75.626),
    "Mississippi Valley State": (33.497, -90.723), "Morgan State": (39.344, -76.582),
    "Norfolk State": (36.885, -76.260), "Prairie View A&M": (30.088, -95.985),
    "Savannah State": (32.053, -81.103), "South Carolina State": (33.496, -80.862),
    "Southern": (30.526, -91.188), "Texas Southern": (29.725, -95.360),
    "Grambling State": (32.524, -92.714),
    # ── Missing schools (batch 2) ──
    "Abilene Christian": (32.449, -99.733), "Akron": (41.076, -81.512),
    "American University": (38.937, -77.089), "App State": (36.215, -81.688),
    "Ball State": (40.206, -85.409), "Bellarmine": (38.218, -85.690),
    "Bowling Green": (41.378, -83.651), "Binghamton": (42.090, -75.968),
    "Bryant": (41.846, -71.457), "Boston University": (42.351, -71.106),
    "Buffalo": (42.886, -78.878), "Campbell": (35.418, -78.860),
    "Chicago State": (41.720, -87.613), "The Citadel": (32.798, -79.958),
    "Cleveland State": (41.502, -81.675), "Cal Poly": (35.301, -120.660),
    "Cal State Bakersfield": (35.350, -119.104), "Cal State Fullerton": (33.883, -117.886),
    "Cal State Northridge": (34.240, -118.529), "Denver": (39.678, -104.962),
    "Detroit Mercy": (42.419, -83.140), "Elon": (36.102, -79.503),
    "Green Bay": (44.531, -87.924), "Grand Canyon": (33.508, -112.124),
    "Gardner-Webb": (35.233, -81.684), "Hawai'i": (21.297, -157.817),
    "High Point": (35.971, -79.995), "Idaho": (46.727, -117.012),
    "Idaho State": (42.862, -112.431), "Kent State": (41.149, -81.342),
    "Le Moyne": (43.041, -76.069), "Lamar": (30.043, -94.077),
    "Long Beach State": (33.783, -118.115), "Longwood": (37.297, -78.396),
    "Loyola Maryland": (39.348, -76.624), "Maine": (44.901, -68.672),
    "McNeese": (30.207, -93.208), "Mercyhurst": (42.099, -80.094),
    "Milwaukee": (43.078, -87.881), "Montana": (46.862, -113.984),
    "Montana State": (45.667, -111.050),
    "New Hampshire": (43.134, -70.934), "New Orleans": (30.028, -90.067),
    "NJIT": (40.742, -74.178), "Nicholls": (29.791, -90.821),
    "Northern Colorado": (40.397, -104.696), "Northern Illinois": (41.934, -88.770),
    "Northern Kentucky": (39.030, -84.465), "Ohio": (39.324, -82.101),
    "Oakland": (42.674, -83.218), "Portland State": (45.512, -122.685),
    "Presbyterian": (34.185, -81.881), "Queens University": (35.191, -80.795),
    "Radford": (37.135, -80.552), "Sacramento State": (38.561, -121.423),
    "Samford": (33.463, -86.794), "South Dakota": (42.890, -96.928),
    "South Dakota State": (44.312, -96.786), "SE Louisiana": (30.516, -90.462),
    "Southern Utah": (37.677, -113.062),
    "St. Thomas": (44.946, -93.189), "Stephen F. Austin": (31.614, -94.649),
    "Stonehill": (42.107, -71.111), "Tarleton State": (32.218, -98.217),
    "Texas A&M-CC": (27.711, -97.326), "Tex A&M Commerce": (33.246, -95.911),
    "UC Davis": (38.538, -121.762), "UC Irvine": (33.643, -117.842),
    "UC Riverside": (33.975, -117.328), "UC San Diego": (32.880, -117.234),
    "UC Santa Barbara": (34.414, -119.849), "UT Rio Grande Valley": (26.305, -98.175),
    "UT Arlington": (32.731, -97.111), "VMI": (37.790, -79.436),
    "Vermont": (44.478, -73.195), "Weber State": (41.192, -111.934),
    "Western Carolina": (35.310, -83.186), "Western Michigan": (42.283, -85.614),
    "Winthrop": (34.943, -81.030), "Wofford": (34.950, -81.930),
    "Wright State": (39.782, -84.061), "Youngstown State": (41.106, -80.646),
}

STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.0, -111.1),
    "AR": (34.8, -92.2), "CA": (36.8, -119.4), "CO": (39.1, -105.4),
    "CT": (41.6, -72.7), "DE": (39.3, -75.5), "FL": (27.8, -81.7),
    "GA": (33.0, -83.6), "HI": (19.9, -155.6), "ID": (44.2, -114.4),
    "IL": (40.3, -89.0), "IN": (39.8, -86.1), "IA": (42.0, -93.2),
    "KS": (38.5, -98.0), "KY": (37.7, -84.7), "LA": (31.2, -91.9),
    "ME": (44.7, -69.4), "MD": (39.0, -76.6), "MA": (42.2, -71.5),
    "MI": (43.3, -84.5), "MN": (46.7, -94.7), "MS": (32.7, -89.5),
    "MO": (38.5, -92.3), "MT": (46.8, -110.4), "NE": (41.1, -98.3),
    "NV": (38.8, -116.4), "NH": (43.5, -71.6), "NJ": (40.3, -74.5),
    "NM": (34.8, -106.2), "NY": (43.3, -74.5), "NC": (35.6, -79.8),
    "ND": (47.5, -99.8), "OH": (40.4, -82.9), "OK": (35.0, -97.1),
    "OR": (43.8, -120.6), "PA": (41.2, -77.2), "RI": (41.6, -71.5),
    "SC": (33.8, -81.2), "SD": (43.9, -99.4), "TN": (35.5, -86.6),
    "TX": (31.0, -97.6), "UT": (39.3, -111.1), "VT": (44.6, -72.6),
    "VA": (37.8, -78.2), "WA": (47.4, -120.7), "WV": (38.5, -80.5),
    "WI": (43.8, -88.8), "WY": (43.1, -107.6), "DC": (38.9, -77.0),
}


def geocode_missing(locations):
    """Fill in lat/lng for teams that only have city/state."""
    filled = 0
    for tid, info in locations.items():
        if info["lat"] is not None:
            continue
        city = info.get("city", "")
        state = info.get("state", "")

        # Try city lookup
        if city in US_CITY_COORDS:
            info["lat"], info["lng"] = US_CITY_COORDS[city]
            filled += 1
        elif state in STATE_CENTROIDS:
            info["lat"], info["lng"] = STATE_CENTROIDS[state]
            filled += 1

    if filled:
        with open(TEAM_LOCATIONS_FILE, "w") as f:
            json.dump(locations, f, indent=2)
        print(f"  Geocoded {filled} additional teams from city/state")


def load_team_locations():
    """Load cached team locations."""
    if os.path.exists(TEAM_LOCATIONS_FILE):
        with open(TEAM_LOCATIONS_FILE) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════
# STEP 2: Compute travel distance for games
# ══════════════════════════════════════════════════════════

def compute_travel_for_game(home_team_id, away_team_id, venue_lat, venue_lng, locations):
    """Compute travel distances from each team to the game venue."""
    home_loc = locations.get(str(home_team_id))
    away_loc = locations.get(str(away_team_id))

    home_dist = None
    away_dist = None

    if home_loc and home_loc.get("lat") and venue_lat:
        home_dist = haversine_miles(home_loc["lat"], home_loc["lng"], venue_lat, venue_lng)

    if away_loc and away_loc.get("lat") and venue_lat:
        away_dist = haversine_miles(away_loc["lat"], away_loc["lng"], venue_lat, venue_lng)

    return home_dist, away_dist


def compute_travel_advantage(home_dist, away_dist):
    """
    Travel advantage feature:
    - Positive = home team is closer (advantage)
    - Negative = away team is closer
    - Uses log ratio to compress extreme values
    - 0 = equal distance
    """
    if home_dist is None or away_dist is None:
        return 0.0
    if home_dist < 1:
        home_dist = 1  # Avoid log(0)
    if away_dist < 1:
        away_dist = 1
    # Log ratio: positive when away travels more
    return math.log(away_dist / home_dist)


# ══════════════════════════════════════════════════════════
# STEP 3: Backfill historical games
# ══════════════════════════════════════════════════════════

def backfill_historical():
    """Add travel features to training data."""
    print("=" * 60)
    print("  BACKFILLING TRAVEL DISTANCE")
    print("=" * 60)

    locations = load_team_locations()
    if not locations:
        print("  ❌ No team locations found. Run --build-teams first.")
        return

    print(f"  Loaded {len(locations)} team locations")
    with_coords = sum(1 for v in locations.values() if v.get("lat"))
    print(f"  {with_coords} with coordinates")

    # Load training data
    from dump_training_data import load_cached
    df = load_cached()
    if df is None:
        print("  ❌ No cached training data")
        return

    print(f"  {len(df)} games in training data")

    # For neutral sites, we need venue location from ESPN
    # For non-neutral, travel_advantage = 0 (home court = home team venue)
    neutral = df.get("neutral_site", pd.Series(False, index=df.index)).fillna(False)
    print(f"  Neutral site games: {neutral.sum()}")

    # For neutral sites, get venue from game data
    # ESPN summary endpoint has venue lat/lng
    n_computed = 0
    n_missing = 0
    travel_adv = np.zeros(len(df))

    for i, row in df.iterrows():
        if not neutral.iloc[i] if isinstance(neutral, pd.Series) else not neutral:
            continue  # Non-neutral: travel_advantage = 0

        home_id = str(row.get("home_team_id", ""))
        away_id = str(row.get("away_team_id", ""))
        game_id = str(row.get("game_id", ""))

        # Get venue location from ESPN
        venue_lat, venue_lng = None, None
        if game_id:
            try:
                resp = requests.get(
                    f"{ESPN_BASE}/summary?event={game_id}",
                    timeout=10
                )
                if resp.ok:
                    gd = resp.json()
                    venue = gd.get("gameInfo", {}).get("venue", {})
                    addr = venue.get("address", {})
                    if addr.get("latitude"):
                        venue_lat = float(addr["latitude"])
                        venue_lng = float(addr["longitude"])
            except:
                pass

        if venue_lat is None:
            # Fallback: if we know both team locations, estimate advantage from team-to-team distance
            home_loc = locations.get(home_id)
            away_loc = locations.get(away_id)
            if home_loc and away_loc and home_loc.get("lat") and away_loc.get("lat"):
                # Midpoint approximation for neutral site
                mid_lat = (home_loc["lat"] + away_loc["lat"]) / 2
                mid_lng = (home_loc["lng"] + away_loc["lng"]) / 2
                hd = haversine_miles(home_loc["lat"], home_loc["lng"], mid_lat, mid_lng)
                ad = haversine_miles(away_loc["lat"], away_loc["lng"], mid_lat, mid_lng)
                travel_adv[i] = compute_travel_advantage(hd, ad)
                n_computed += 1
            else:
                n_missing += 1
            continue

        hd, ad = compute_travel_for_game(home_id, away_id, venue_lat, venue_lng, locations)
        travel_adv[i] = compute_travel_advantage(hd, ad)
        n_computed += 1

        if n_computed % 100 == 0:
            print(f"    {n_computed} neutral games processed...")
            time.sleep(0.5)  # rate limit

    df["travel_advantage"] = travel_adv
    print(f"\n  Computed: {n_computed}, Missing: {n_missing}")
    print(f"  travel_advantage stats:")
    ta = travel_adv[travel_adv != 0]
    if len(ta):
        print(f"    mean: {ta.mean():.3f}, std: {ta.std():.3f}")
        print(f"    min: {ta.min():.3f}, max: {ta.max():.3f}")
        print(f"    >0 (home closer): {(ta > 0).sum()}, <0 (away closer): {(ta < 0).sum()}")

    return df


# ══════════════════════════════════════════════════════════
# STEP 4: Test on today's games
# ══════════════════════════════════════════════════════════

def test_today():
    """Show travel distances for neutral site games."""
    print("=" * 60)
    print("  NEUTRAL SITE TRAVEL ANALYSIS")
    print("=" * 60)

    locations = load_team_locations()
    if not locations:
        print("  ❌ Run --build-teams first")
        return

    # Build reverse lookup: abbr → location data
    abbr_to_loc = {}
    for tid, info in locations.items():
        if info.get("abbr"):
            abbr_to_loc[info["abbr"]] = info

    # Try today first, then fall back to recent dates with games
    today = pd.Timestamp.now().strftime("%Y%m%d")
    test_dates = [today, "20260319", "20260320", "20260321", "20260328"]

    for date in test_dates:
        resp = requests.get(f"{ESPN_BASE}/scoreboard?dates={date}&limit=50", timeout=15)
        data = resp.json()
        events = data.get("events", [])
        neutrals = [e for e in events if e["competitions"][0].get("neutralSite", False)]

        if not neutrals:
            continue

        print(f"\n  Date: {date} — {len(neutrals)} neutral site games\n")

        for ev in neutrals:
            comp = ev["competitions"][0]
            venue = comp.get("venue", {})
            addr = venue.get("address", {})
            venue_name = venue.get("fullName", "Unknown")
            venue_city = addr.get("city", "")
            venue_state = addr.get("state", "")

            # Geocode venue from city name
            venue_lat, venue_lng = None, None
            if addr.get("latitude"):
                venue_lat = float(addr["latitude"])
                venue_lng = float(addr["longitude"])
            elif venue_city in US_CITY_COORDS:
                venue_lat, venue_lng = US_CITY_COORDS[venue_city]
            elif venue_city:
                # Try partial match
                for city_name, coords in US_CITY_COORDS.items():
                    if city_name.lower() in venue_city.lower() or venue_city.lower() in city_name.lower():
                        venue_lat, venue_lng = coords
                        break
            # Fall back to state centroid
            if venue_lat is None and venue_state in STATE_CENTROIDS:
                venue_lat, venue_lng = STATE_CENTROIDS[venue_state]

            teams = {}
            for t in comp.get("competitors", []):
                team = t.get("team", {})
                ha = t["homeAway"]
                teams[ha] = {
                    "abbr": team.get("abbreviation", ""),
                    "name": team.get("displayName", ""),
                }

            home = teams.get("home", {})
            away = teams.get("away", {})

            home_loc = abbr_to_loc.get(home["abbr"])
            away_loc = abbr_to_loc.get(away["abbr"])

            print(f"  {away['abbr']} @ {home['abbr']} — {venue_name}, {venue_city}, {venue_state}")

            if venue_lat is None:
                print(f"    ⚠ Could not geocode venue: {venue_city}, {venue_state}")
                continue

            hd, ad = None, None
            if home_loc and home_loc.get("lat"):
                hd = haversine_miles(home_loc["lat"], home_loc["lng"], venue_lat, venue_lng)
                print(f"    {home['abbr']:>6s}: {hd:>6.0f} miles from {home_loc.get('location','?')}")

            if away_loc and away_loc.get("lat"):
                ad = haversine_miles(away_loc["lat"], away_loc["lng"], venue_lat, venue_lng)
                print(f"    {away['abbr']:>6s}: {ad:>6.0f} miles from {away_loc.get('location','?')}")

            if hd is not None and ad is not None:
                ta = compute_travel_advantage(hd, ad)
                closer = home["abbr"] if ta > 0 else away["abbr"]
                farther = away["abbr"] if ta > 0 else home["abbr"]
                ratio = max(hd, ad) / max(min(hd, ad), 1)
                print(f"    Travel advantage: {ta:+.3f} → {closer} is {ratio:.1f}x closer")
                if abs(ta) > 1.5:
                    print(f"    🔴 HUGE proximity edge — {closer} essentially has home court")
                elif abs(ta) > 0.7:
                    print(f"    🟡 Significant proximity advantage for {closer}")
                elif abs(ta) > 0.3:
                    print(f"    🟢 Moderate proximity advantage for {closer}")
                else:
                    print(f"    ⚪ Roughly equal travel")
            print()

        return  # Only process the first date with games

    print("  No neutral site games found on any test date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-teams", action="store_true", help="Scrape team locations from ESPN")
    parser.add_argument("--backfill", action="store_true", help="Add travel features to training data")
    parser.add_argument("--test", action="store_true", help="Test on today's games")
    args = parser.parse_args()

    if args.build_teams:
        build_team_locations()
    elif args.backfill:
        backfill_historical()
    elif args.test:
        test_today()
    else:
        print("Usage:")
        print("  python3 build_travel_distance.py --build-teams   # Step 1")
        print("  python3 build_travel_distance.py --test          # Test today")
        print("  python3 build_travel_distance.py --backfill      # Step 2")
