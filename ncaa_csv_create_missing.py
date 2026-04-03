#!/usr/bin/env python3
"""
ncaa_csv_create_missing.py — Create missing ncaa_historical rows from CSV files
================================================================================
For games in the CSV that don't exist in ncaa_historical, create skeleton rows
with: date, teams, scores, spreads. No box scores, but enough for spread-based training.

Usage:
    python3 ncaa_csv_create_missing.py --dry-run
    python3 ncaa_csv_create_missing.py
"""
import sys, os, time, argparse, re, json
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, '.')

try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
    SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_KEY", ""))

SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

CSV_FILES = {
    "ncaabb22.csv": (2023, "2022-23"),
    "ncaabb23.csv": (2024, "2023-24"),
    "ncaabb24.csv": (2025, "2024-25"),
    "ncaabb25.csv": (2026, "2025-26"),
}


# ══════════════════════════════════════════════════════════
# TEAM NAME → ID MAPPING
# ══════════════════════════════════════════════════════════

def build_name_to_id():
    """Build CSV team name → ESPN team_id mapping from multiple sources."""
    mapping = {}
    
    # Source 1: ncaa_team_locations.json (id → name, reverse it)
    try:
        with open("ncaa_team_locations.json") as f:
            locs = json.load(f)
        for tid, info in locs.items():
            name = info.get("name", "").strip()
            if name:
                mapping[name.lower()] = tid
                # Also add without common suffixes
                for suffix in [" state", " st"]:
                    if name.lower().endswith(suffix):
                        mapping[name.lower().replace(suffix, "")] = tid
    except:
        print("  ⚠ ncaa_team_locations.json not found")
    
    # Source 2: Supabase — get all unique team_id + team_name pairs
    try:
        url = (f"{SUPABASE_URL}/rest/v1/ncaa_historical"
               f"?home_team_name=not.is.null"
               f"&select=home_team_id,home_team_name"
               f"&limit=5000")
        r = requests.get(url, headers=SB_HEADERS, timeout=30)
        if r.ok:
            for row in r.json():
                name = str(row.get("home_team_name", "")).strip()
                tid = str(row.get("home_team_id", ""))
                if name and tid:
                    mapping[name.lower()] = tid
                    # Strip mascot for alternate lookup
                    parts = name.split()
                    if len(parts) > 1:
                        # Try without last word (mascot)
                        mapping[" ".join(parts[:-1]).lower()] = tid
    except:
        pass
    
    # Source 3: Manual overrides for common CSV names that differ from ESPN
    manual = {
        "abilene christian": "2000", "air force": "2005", "akron": "2006",
        "alabama": "333", "alabama a&m": "2010", "alabama st.": "2011",
        "albany-ny": "399", "alcorn st.": "2016", "american": "44",
        "appalachian st.": "2026", "arizona": "12", "arizona st.": "9",
        "arkansas": "8", "arkansas pine bluff": "2029", "arkansas st.": "2032",
        "arkansas-little rock": "2031", "army": "349", "auburn": "2",
        "austin peay": "2046", "ball st.": "2050", "baylor": "239",
        "belmont": "2057", "bethune-cookman": "2065", "binghamton": "2066",
        "boise st.": "68", "boston college": "103", "boston university": "104",
        "bowling green": "189", "bradley": "71", "brown": "225",
        "bryant": "2803", "bucknell": "2083", "buffalo": "2084",
        "butler": "2086", "cal poly": "13", "cal st. bakersfield": "2934",
        "cal st. fullerton": "2239", "cal st. northridge": "2463",
        "california": "25", "california baptist": "2856", "campbell": "2097",
        "canisius": "2099", "central arkansas": "2110", "central connecticut st.": "2115",
        "central florida": "2116", "central michigan": "2117", "charleston": "232",
        "charleston southern": "2127", "charlotte": "2429", "chattanooga": "236",
        "chicago st.": "2130", "cincinnati": "2132", "citadel": "2643",
        "clemson": "228", "cleveland st.": "325", "coastal carolina": "324",
        "colgate": "2142", "colorado": "38", "colorado st.": "36",
        "columbia": "171", "connecticut": "41", "coppin st.": "2154",
        "cornell": "172", "creighton": "156", "dartmouth": "159",
        "davidson": "2166", "dayton": "2168", "delaware": "48",
        "delaware st.": "2169", "denver": "2172", "depaul": "305",
        "detroit mercy": "2174", "drake": "2181", "drexel": "2182",
        "duke": "150", "duquesne": "2184", "east carolina": "151",
        "east tennessee st.": "2193", "eastern illinois": "2197",
        "eastern kentucky": "2198", "eastern michigan": "2199",
        "eastern washington": "331", "elon": "2210", "evansville": "339",
        "fairfield": "2217", "fairleigh dickinson": "2218",
        "fiu": "2229", "florida": "57", "florida a&m": "50",
        "florida atlantic": "2226", "florida gulf coast": "526",
        "florida st.": "52", "fordham": "2230", "fresno st.": "278",
        "furman": "231", "gardner-webb": "2241", "george mason": "2244",
        "george washington": "45", "georgetown": "46", "georgia": "61",
        "georgia southern": "290", "georgia st.": "2247", "georgia tech": "59",
        "gonzaga": "2250", "grambling": "2755", "grand canyon": "2253",
        "green bay": "2739", "hampton": "2261", "hartford": "42",
        "harvard": "108", "hawaii": "62", "high point": "2272",
        "hofstra": "2275", "holy cross": "107", "houston": "248",
        "houston christian": "2277", "howard": "47", "idaho": "70",
        "idaho st.": "304", "illinois": "356", "illinois st.": "2287",
        "incarnate word": "2916", "indiana": "84", "indiana st.": "282",
        "iona": "314", "iowa": "2294", "iowa st.": "66",
        "iupui": "85", "jackson st.": "2296", "jacksonville": "294",
        "jacksonville st.": "55", "james madison": "256",
        "kansas": "2305", "kansas st.": "2306", "kennesaw st.": "2309",
        "kent st.": "2309", "kentucky": "96", "la salle": "2325",
        "lafayette": "322", "lamar": "2320", "lehigh": "2329",
        "liberty": "2335", "lindenwood": "2815", "lipscomb": "288",
        "long beach st.": "299", "long island": "2344",
        "louisiana": "309", "louisiana tech": "2348",
        "louisville": "97", "loyola chicago": "2350",
        "loyola marymount": "2351", "loyola-md": "2352", "lsu": "99",
        "maine": "311", "manhattan": "2363", "marist": "2368",
        "marquette": "269", "marshall": "276", "maryland": "120",
        "maryland-eastern shore": "2379", "massachusetts": "113",
        "mcneese st.": "2377", "memphis": "235", "mercer": "2382",
        "merrimack": "2931", "miami-fl": "2390", "miami-oh": "193",
        "michigan": "130", "michigan st.": "127",
        "middle tennessee": "2393", "milwaukee": "270",
        "minnesota": "135", "mississippi": "145",
        "mississippi st.": "344", "mississippi valley st.": "2400",
        "missouri": "142", "missouri st.": "2623",
        "monmouth": "2405", "montana": "149", "montana st.": "147",
        "morehead st.": "2413", "morgan st.": "2415",
        "mount st. mary's": "116", "mt. st. mary's": "116",
        "murray st.": "93", "navy": "2426", "nc a&t": "2428",
        "nc central": "2428", "nc state": "152", "nebraska": "158",
        "nevada": "2440", "new hampshire": "160", "new mexico": "167",
        "new mexico st.": "166", "new orleans": "2443", "niagara": "315",
        "nicholls st.": "2447", "njit": "2885", "norfolk st.": "2450",
        "north alabama": "2453", "north carolina": "153",
        "north carolina a&t": "2448", "north dakota": "155",
        "north dakota st.": "2449", "north florida": "2454",
        "north texas": "249", "northeastern": "111",
        "northern arizona": "2464", "northern colorado": "2458",
        "northern illinois": "2459", "northern iowa": "2460",
        "northern kentucky": "94", "northwestern": "77",
        "northwestern st.": "2466", "notre dame": "87",
        "oakland": "2473", "ohio": "195", "ohio st.": "194",
        "oklahoma": "201", "oklahoma st.": "197",
        "old dominion": "295", "ole miss": "145", "omaha": "2437",
        "oral roberts": "198", "oregon": "2483", "oregon st.": "204",
        "pacific": "279", "penn": "219", "penn st.": "213",
        "pepperdine": "2492", "pittsburgh": "221", "portland": "2501",
        "portland st.": "2502", "prairie view a&m": "2504",
        "presbyterian": "2506", "princeton": "163", "providence": "2507",
        "purdue": "2509", "purdue fort wayne": "2870",
        "queens university": "2511", "quinnipiac": "2514",
        "radford": "2515", "rhode island": "227",
        "rice": "242", "richmond": "257", "rider": "2520",
        "robert morris": "2523", "rutgers": "164",
        "sacramento st.": "16", "sacred heart": "2529",
        "saint francis-pa": "2598", "saint joseph's": "2603",
        "saint louis": "139", "saint mary's": "2608",
        "saint peter's": "2612", "sam houston": "2534",
        "samford": "2535", "san diego": "301", "san diego st.": "21",
        "san francisco": "2539", "san jose st.": "23",
        "santa clara": "2541", "seattle": "2547",
        "se louisiana": "2545", "se missouri st.": "2546",
        "seton hall": "2550", "siena": "2561",
        "siu edwardsville": "2565", "smu": "2567",
        "south alabama": "6", "south carolina": "2579",
        "south carolina st.": "2569", "south dakota": "233",
        "south dakota st.": "2571", "south florida": "58",
        "southeast missouri": "2546",
        "southern": "2582", "southern illinois": "79",
        "southern indiana": "2584", "southern miss": "2572",
        "southern utah": "253", "st. bonaventure": "2599",
        "st. francis-pa": "2598", "st. john's": "2599",
        "st. thomas-mn": "2900", "stanford": "24",
        "stephen f. austin": "2617", "stetson": "56",
        "stonehill": "2919", "stony brook": "2619",
        "syracuse": "183", "tarleton st.": "2867",
        "tcu": "2628", "temple": "218", "tennessee": "2633",
        "tennessee st.": "2634", "tennessee tech": "2635",
        "tenn-martin": "2630", "texas": "251",
        "texas a&m": "245", "texas a&m-cc": "357",
        "texas a&m-commerce": "2837", "texas southern": "2640",
        "texas st.": "326", "texas tech": "2641",
        "the citadel": "2643", "toledo": "2649",
        "towson": "119", "troy": "2653", "tulane": "2655",
        "tulsa": "2656", "uab": "5", "uc davis": "302",
        "uc irvine": "300", "uc riverside": "27", "uc san diego": "28",
        "uc santa barbara": "2540", "ucf": "2116",
        "ucla": "26", "umass lowell": "2349",
        "umbc": "2378", "unc asheville": "2427",
        "unc greensboro": "2430", "unc wilmington": "350",
        "unlv": "2439", "usc": "30", "usc upstate": "2908",
        "ut arlington": "250", "ut rio grande valley": "2638",
        "utah": "254", "utah st.": "328", "utah tech": "2862",
        "utah valley": "3084", "utep": "2638",
        "utsa": "2636", "valparaiso": "2674",
        "vanderbilt": "238", "vermont": "261",
        "villanova": "222", "virginia": "258", "virginia tech": "259",
        "vmi": "2678", "wagner": "2681",
        "wake forest": "154", "washington": "264",
        "washington st.": "265", "weber st.": "2692",
        "west virginia": "277", "western carolina": "2717",
        "western illinois": "2710", "western kentucky": "98",
        "western michigan": "2711", "wichita st.": "2724",
        "william & mary": "2729", "winthrop": "2737",
        "wisconsin": "275", "wofford": "2747",
        "wright st.": "2750", "wyoming": "2751",
        "xavier": "2752", "yale": "43", "youngstown st.": "2754",
        "cs bakersfield": "2934", "cs fullerton": "2239",
        "cs northridge": "2463",
        # Additional CSV name variants
        "a&m-commerce": "2837", "byu": "252", "bellarmine": "91",
        "bethune cookman": "2065", "boston": "104",
        "cs sacramento": "16", "cal poly slo": "13", "cal riverside": "27",
        "central conn. st.": "2115", "detroit": "2174",
        "east texas a&m": "2837", "florida international": "2229",
        "gardner webb": "2241", "grambling st.": "2755",
        "ipfw": "2870", "illinois-chicago": "82",
        "iu indianapolis": "85", "iupui": "85",
        "le moyne": "2329", "lindenwood": "2815",
        "little rock": "2031", "liu": "2344",
        "loyola (il)": "2350", "loyola (md)": "2352",
        "loyola-chicago": "2350",
        "mass.-lowell": "2349", "umass lowell": "2349",
        "miami (fl)": "2390", "miami (oh)": "193",
        "milwaukee": "270", "mississippi val. st.": "2400",
        "montana state": "147", "morehead state": "2413",
        "n.c. a&t": "2448", "nc a&t": "2448",
        "nicholls": "2447", "north ala.": "2453",
        "northern ky.": "94", "northwest st.": "2466",
        "northwestern st.": "2466",
        "pv a&m": "2504", "prairie view": "2504",
        "purdue-fort wayne": "2870", "queens": "2511",
        "sam houston st.": "2534", "se louisiana": "2545",
        "south carolina upstate": "2908",
        "southern ind.": "2584", "southern indiana": "2584",
        "southern u.": "2582",
        "st. bonaventure": "179", "st. francis (pa)": "2598",
        "st. francis-bkn": "2597", "st. john's": "2599",
        "st. thomas (mn)": "2900", "st. thomas-mn": "2900",
        "tenn. st.": "2634", "tenn. tech": "2635",
        "texas a&m-corpus christi": "357",
        "texas a&m commerce": "2837",
        "unc-asheville": "2427", "unc-greensboro": "2430",
        "unc-wilmington": "350",
        "ut martin": "2630", "ut-martin": "2630",
        "utah valley st.": "3084",
        "w. carolina": "2717", "w. illinois": "2710",
        "w. kentucky": "98", "w. michigan": "2711",
        "wm. & mary": "2729",
        # Round 2 fixes
        "longwood": "2374", "louisiana-lafayette": "309", "louisiana-monroe": "2433",
        "loyola-maryland": "2352", "md baltimore co": "2378", "md. eastern shore": "2379",
        "miami-florida": "2390", "miami-ohio": "193", "middle tenn st.": "2393",
        "miss valley st.": "2400", "mo kansas city": "140", "mount st. marys": "116",
        "nc asheville": "2427", "nc charlotte": "2429", "nc greensboro": "2430",
        "nc wilmington": "350", "n.c. state": "152", "nw state": "2466",
        "penn state": "213", "saint francis": "2598",
        "s. carolina st.": "2569", "s. dakota st.": "2571",
        "s. illinois": "79", "s. indiana": "2584",
        "so. mississippi": "2572", "southern miss.": "2572",
        "tx a&m-cc": "357", "tx southern": "2640",
        "w. virginia": "277", "wright state": "2750",
        "mercyhurst": "2922", "le moyne": "2329",
        "st. francis-bkn.": "2597", "st. francis-pa.": "2598",
        "ut san antonio": "2636", "ut arlington": "250",
        "n. dakota st.": "2449", "n. dakota": "155",
        "s. dakota": "233", "n. carolina": "153",
        "n. iowa": "2460", "n. arizona": "2464",
        "n. colorado": "2458", "n. kentucky": "94",
        "n. illinois": "2459", "n. alabama": "2453",
        "e. kentucky": "2198", "e. michigan": "2199",
        "e. illinois": "2197", "e. washington": "331",
        "e. carolina": "151", "e. tennessee st.": "2193",
        "w. carolina": "2717", "w. illinois": "2710",
        "se missouri": "2546", "sw missouri st.": "2623",
    }
    for name, tid in manual.items():
        mapping[name] = tid
    
    print(f"  Team mapping: {len(mapping)} entries")
    return mapping


# ══════════════════════════════════════════════════════════
# DB LOOKUP
# ══════════════════════════════════════════════════════════

def get_existing_scores(season):
    """Get set of (date, home_score, away_score) tuples for dedup."""
    existing = set()
    year_start = season - 1
    months = [
        (f"{year_start}-11-01", f"{year_start}-11-30"),
        (f"{year_start}-12-01", f"{year_start}-12-31"),
        (f"{season}-01-01", f"{season}-01-31"),
        (f"{season}-02-01", f"{season}-02-28"),
        (f"{season}-03-01", f"{season}-03-31"),
        (f"{season}-04-01", f"{season}-04-15"),
    ]
    
    for start, end in months:
        offset = 0
        while True:
            url = (f"{SUPABASE_URL}/rest/v1/ncaa_historical"
                   f"?season=eq.{season}"
                   f"&game_date=gte.{start}&game_date=lte.{end}"
                   f"&select=game_date,actual_home_score,actual_away_score"
                   f"&order=game_date.asc&offset={offset}&limit=1000")
            try:
                r = requests.get(url, headers=SB_HEADERS, timeout=60)
                if not r.ok: break
                rows = r.json()
                if not rows: break
                for row in rows:
                    d = row.get("game_date", "")
                    h = row.get("actual_home_score")
                    a = row.get("actual_away_score")
                    if h and a:
                        existing.add((d, int(float(h)), int(float(a))))
                        existing.add((d, int(float(a)), int(float(h))))  # both directions
                offset += len(rows)
                if len(rows) < 1000: break
                time.sleep(1)
            except:
                break
    
    return existing


# ══════════════════════════════════════════════════════════
# PROCESS
# ══════════════════════════════════════════════════════════

def process_csv(csv_path, season, name_map, dry_run=False):
    csv_name = os.path.basename(csv_path)
    print(f"\n  {csv_name} → season {season}")
    print(f"  {'─' * 50}")
    
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")
    print(f"  CSV: {len(df)} games")
    
    # Get existing games for dedup
    existing = get_existing_scores(season)
    print(f"  DB existing score combos: {len(existing)//2}")
    
    # Find games NOT in DB
    new_rows = []
    no_id = set()
    already = 0
    
    for _, row in df.iterrows():
        hs = int(float(row["hscore"])) if pd.notna(row.get("hscore")) else None
        rs = int(float(row["rscore"])) if pd.notna(row.get("rscore")) else None
        d = row["game_date"]
        
        if hs is None or rs is None:
            continue
        
        # Skip if already in DB (either direction)
        if (d, hs, rs) in existing:
            already += 1
            continue
        
        # Look up team IDs
        home_name = str(row.get("home", "")).strip()
        away_name = str(row.get("road", "")).strip()
        home_id = name_map.get(home_name.lower())
        away_id = name_map.get(away_name.lower())
        
        if not home_id:
            no_id.add(home_name)
            continue
        if not away_id:
            no_id.add(away_name)
            continue
        
        # Build row
        line = float(row["line"]) if pd.notna(row.get("line")) else None
        lineopen = float(row["lineopen"]) if pd.notna(row.get("lineopen")) else None
        lineespn = float(row["lineespn"]) if pd.notna(row.get("lineespn")) else None
        neutral = int(float(row["neutral"])) if pd.notna(row.get("neutral")) else 0
        
        # Generate synthetic game_id (900xxxxxx range, won't collide with ESPN 401xxxxxx)
        # Deterministic: season * 100000 + row index ensures uniqueness
        synthetic_id = 900000000 + (season * 10000) + len(new_rows)
        
        game_row = {
            "game_id": str(synthetic_id),
            "game_date": d,
            "season": season,
            "home_team_id": int(home_id),
            "away_team_id": int(away_id),
            "home_team_name": home_name,
            "away_team_name": away_name,
            "actual_home_score": hs,
            "actual_away_score": rs,
            "neutral_site": bool(neutral),
        }
        
        if line is not None:
            game_row["market_spread_home"] = round(-line, 1)
        if lineopen is not None:
            game_row["odds_api_spread_open"] = round(-lineopen, 1)
            if line is not None:
                game_row["odds_api_spread_movement"] = round(lineopen - line, 1)
        if lineespn is not None:
            game_row["espn_spread"] = round(-lineespn, 1)
        
        new_rows.append(game_row)
    
    print(f"  Already in DB: {already}")
    print(f"  New rows to create: {len(new_rows)}")
    if no_id:
        print(f"  Missing team IDs ({len(no_id)}): {sorted(no_id)[:15]}...")
    
    if dry_run or not new_rows:
        print(f"  {'DRY RUN — ' if dry_run else ''}would write {len(new_rows)}")
        return 0
    
    # Write in batches
    written = 0
    batch_size = 50
    errors_shown = 0
    for i in range(0, len(new_rows), batch_size):
        batch = new_rows[i:i+batch_size]
        try:
            url = f"{SUPABASE_URL}/rest/v1/ncaa_historical"
            r = requests.post(url, headers=SB_HEADERS, json=batch, timeout=30)
            if r.ok:
                written += len(batch)
            else:
                if errors_shown < 3:
                    print(f"    Batch error: {r.status_code} — {r.text[:200]}")
                    errors_shown += 1
                # Try one by one
                for row in batch:
                    try:
                        r2 = requests.post(url, headers=SB_HEADERS, json=row, timeout=10)
                        if r2.ok:
                            written += 1
                        elif errors_shown < 5:
                            print(f"    Row error: {r2.status_code} — {r2.text[:150]}")
                            errors_shown += 1
                    except Exception as e:
                        if errors_shown < 5:
                            print(f"    Exception: {e}")
                            errors_shown += 1
        except Exception as e:
            if errors_shown < 3:
                print(f"    Batch exception: {e}")
                errors_shown += 1
        
        if (i + batch_size) % 500 < batch_size:
            print(f"    {min(i+batch_size, len(new_rows))}/{len(new_rows)} — {written} ok")
    
    print(f"  ✅ Created: {written}/{len(new_rows)}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NCAA CSV — CREATE MISSING GAMES")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)
    
    name_map = build_name_to_id()
    
    total = 0
    for csv_name, (season, desc) in CSV_FILES.items():
        for path in [csv_name, f"/mnt/project/{csv_name}", f"data/{csv_name}"]:
            if os.path.exists(path):
                total += process_csv(path, season, name_map, dry_run=args.dry_run)
                break
        else:
            print(f"\n  ⚠ {csv_name} not found")
    
    print(f"\n  Total created: {total}")
    print(f"  Done.")
