#!/usr/bin/env python3
"""
ncaa_csv_spread_ingest.py — Ingest historical spreads from CSV files
====================================================================
Matches CSV games to ncaa_historical by date + team name, writes:
  - market_spread_home (closing line, negated to match DB convention)
  - espn_spread (same as market_spread_home for compatibility)
  - Opening spread + movement

CSV sign convention: positive = home favored
DB sign convention:  negative = home favored
Transform: market_spread_home = -line

Usage:
    python3 ncaa_csv_spread_ingest.py
    python3 ncaa_csv_spread_ingest.py --dry-run
"""
import sys, os, re, time, argparse
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
    "Prefer": "return=representation",
}

def norm(name):
    if not name: return ""
    name = str(name).lower().strip()
    name = name.replace("st.", "state").replace("'", "").replace("'", "").replace("-", " ")
    name = name.replace("(", "").replace(")", "").replace("  ", " ").strip()
    mascots = [
        "wildcats","bulldogs","tigers","bears","eagles","hawks","lions","panthers",
        "knights","warriors","cougars","wolves","huskies","cardinals","bruins","trojans",
        "gators","seminoles","cavaliers","wolverines","spartans","buckeyes","badgers",
        "jayhawks","cyclones","longhorns","sooners","razorbacks","volunteers","gamecocks",
        "blue devils","tar heels","fighting irish","mountaineers","wolfpack","beavers",
        "ducks","utes","buffaloes","golden bears","sun devils","pirates","owls","rams",
        "miners","aztecs","aggies","rebels","mustangs","bobcats","bearcats","flyers",
        "gaels","peacocks","explorers","hoyas","red storm","billikens","musketeers",
        "toreros","friars","shockers","golden eagles","demon deacons","yellow jackets",
        "crimson tide","scarlet knights","nittany lions","hawkeyes","cornhuskers",
        "gophers","fighting illini","boilermakers","hoosiers","terrapins","hokies",
        "commodores","red raiders","horned frogs","orange","antelopes","phoenix",
        "flames","salukis","redhawks","seahawks","matadors","retrievers","privateers",
    ]
    for m in mascots:
        name = name.replace(f" {m}", "")
    return name.strip()

NAME_MAP = {
    "miami fl": "miami", "miami ohio": "miami oh", "uconn": "connecticut",
    "umass": "massachusetts", "vmi": "virginia military", "smu": "southern methodist",
    "tcu": "texas christian", "ucf": "central florida", "unlv": "nevada las vegas",
    "utep": "texas el paso", "utsa": "texas san antonio", "lsu": "louisiana state",
    "ole miss": "mississippi", "pitt": "pittsburgh", "usc": "southern california",
    "byu": "brigham young", "nc state": "north carolina state",
    "nc a&t": "north carolina a&t", "nc central": "north carolina central",
    "unc asheville": "north carolina asheville", "unc wilmington": "north carolina wilmington",
    "unc greensboro": "north carolina greensboro", "penn state": "penn state",
    "penn": "pennsylvania", "app state": "appalachian state",
    "appalachian st.": "appalachian state", "fla atlantic": "florida atlantic",
    "fla gulf coast": "florida gulf coast", "se missouri": "southeast missouri state",
    "s illinois": "southern illinois", "e illinois": "eastern illinois",
    "e michigan": "eastern michigan", "e kentucky": "eastern kentucky",
    "e washington": "eastern washington", "w illinois": "western illinois",
    "w michigan": "western michigan", "w kentucky": "western kentucky",
    "n illinois": "northern illinois", "n kentucky": "northern kentucky",
    "n iowa": "northern iowa", "n arizona": "northern arizona",
    "cent michigan": "central michigan", "c michigan": "central michigan",
    "col of charleston": "college of charleston", "tenn martin": "tennessee martin",
    "tenn state": "tennessee state", "tenn tech": "tennessee tech",
    "alab a&m": "alabama a&m", "miss state": "mississippi state",
    "miss valley state": "mississippi valley state", "mt. st. mary's": "mount state marys",
    "st. john's": "state johns", "st. peter's": "state peters",
    "st. francis pa": "state francis pa", "st. bonaventure": "state bonaventure",
    "nebraska omaha": "nebraska omaha", "loyola maryland": "loyola maryland",
    "loyola chicago": "loyola chicago", "ga southern": "georgia southern",
    "ga tech": "georgia tech", "bethune cookman": "bethune cookman",
    "e tennessee state": "east tennessee state",
}

CSV_FILES = {
    "ncaabb22.csv": (2023, "2022-23"),
    "ncaabb23.csv": (2024, "2023-24"),
    "ncaabb24.csv": (2025, "2024-25"),
    "ncaabb25.csv": (2026, "2025-26"),
}

def load_db_games(season):
    """Load games with minimal columns for matching (avoids Supabase timeout)."""
    # Try parquet first (instant)
    if os.path.exists("ncaa_training_data.parquet"):
        try:
            df = pd.read_parquet("ncaa_training_data.parquet")
            s = df[df["season"] == season].copy()
            if len(s) > 100:
                print(f"  Loaded {len(s)} games from parquet cache")
                games = []
                for _, row in s.iterrows():
                    games.append({
                        "game_id": str(row.get("game_id", "")),
                        "game_date": str(row.get("game_date", ""))[:10],
                        "home_team_id": str(row.get("home_team_id", "")),
                        "away_team_id": str(row.get("away_team_id", "")),
                        "home_team_name": str(row["home_team_name"]) if pd.notna(row.get("home_team_name")) else "",
                        "away_team_name": str(row["away_team_name"]) if pd.notna(row.get("away_team_name")) else "",
                        "actual_home_score": float(row["actual_home_score"]) if pd.notna(row.get("actual_home_score")) else None,
                        "actual_away_score": float(row["actual_away_score"]) if pd.notna(row.get("actual_away_score")) else None,
                        "market_spread_home": float(row["market_spread_home"]) if pd.notna(row.get("market_spread_home")) and float(row.get("market_spread_home", 0)) != 0 else None,
                    })
                _enrich_names(games)
                return games
        except: pass
    
    # Fallback: query Supabase with MINIMAL columns to avoid timeout
    print(f"  Querying Supabase (minimal columns)...")
    all_games = []
    # Only select what we need for matching — 9 columns instead of 395
    select = "game_id,game_date,home_team_id,away_team_id,home_team_name,away_team_name,actual_home_score,actual_away_score,market_spread_home"
    
    # Chunk by month
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
                   f"&select={select}"
                   f"&order=game_date.asc&offset={offset}&limit=1000")
            try:
                r = requests.get(url, headers=SB_HEADERS, timeout=60)
                if not r.ok:
                    time.sleep(5)
                    break
                rows = r.json()
                if not rows: break
                all_games.extend(rows)
                offset += len(rows)
                if len(rows) < 1000: break
                time.sleep(1)
            except:
                time.sleep(5)
                break
    
    print(f"  Loaded {len(all_games)} games from Supabase")
    _enrich_names(all_games)
    return all_games


def _enrich_names(games):
    """Fill missing team names from locations file."""
    try:
        import json
        with open("ncaa_team_locations.json") as f:
            locs = json.load(f)
        names = {tid: info.get("name", "") for tid, info in locs.items()}
        for g in games:
            hid = str(g.get("home_team_id", ""))
            aid = str(g.get("away_team_id", ""))
            if not g.get("home_team_name"):
                g["home_team_name"] = names.get(hid, "")
            if not g.get("away_team_name"):
                g["away_team_name"] = names.get(aid, "")
    except: pass

def process_csv(csv_path, season, dry_run=False):
    csv_name = os.path.basename(csv_path)
    print(f"\n  {csv_name} → season {season}")
    print(f"  {'─' * 50}")
    
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")
    print(f"  CSV: {len(df)} games, dates: {df['game_date'].min()} → {df['game_date'].max()}")
    
    db_games = load_db_games(season)
    print(f"  DB:  {len(db_games)} games")
    already_has = sum(1 for g in db_games if g.get("market_spread_home") is not None)
    print(f"  Already has spread: {already_has}, Needs: {len(db_games) - already_has}")
    
    # Build lookup: date → {norm_home → game}
    lookup = {}
    for g in db_games:
        d = g.get("game_date", "")
        hn = norm(g.get("home_team_name", ""))
        if d and hn:
            lookup.setdefault(d, {})[hn] = g
    
    matched = 0; already = 0; no_match = 0; no_line = 0
    patches = {}
    unmatched = set()
    
    for _, row in df.iterrows():
        date = row["game_date"]
        home = str(row.get("home", ""))
        line = row.get("line")
        lineopen = row.get("lineopen")
        
        if pd.isna(line): no_line += 1; continue
        
        date_dict = lookup.get(date, {})
        if not date_dict: no_match += 1; continue
        
        cn = norm(home)
        mapped = NAME_MAP.get(cn, cn)
        
        found = None
        for db_key, game in date_dict.items():
            if mapped == db_key or cn == db_key:
                found = game; break
            elif len(mapped) >= 5 and len(db_key) >= 5:
                if mapped in db_key or db_key in mapped:
                    found = game; break
                elif mapped[:5] == db_key[:5]:
                    found = game; break
            if len(cn) >= 5 and len(db_key) >= 5:
                if cn in db_key or db_key in cn:
                    found = game; break
        
        if not found:
            no_match += 1
            unmatched.add(home)
            continue
        
        if found.get("market_spread_home") is not None:
            already += 1; continue
        
        matched += 1
        gid = found["game_id"]
        
        patch = {
            "market_spread_home": round(-float(line), 1),
            "espn_spread": round(-float(line), 1),
        }
        
        if not pd.isna(lineopen):
            open_val = round(-float(lineopen), 1)
            movement = round(patch["market_spread_home"] - open_val, 1)
            patch["odds_api_spread_open"] = open_val
            patch["odds_api_spread_close"] = patch["market_spread_home"]
            if movement != 0:
                patch["odds_api_spread_movement"] = movement
        
        patches[str(gid)] = patch
    
    print(f"\n  Matched: {matched} | Already: {already} | No match: {no_match} | No line: {no_line}")
    if unmatched:
        print(f"  Unmatched ({len(unmatched)}): {sorted(unmatched)[:15]}...")
    
    if patches and not dry_run:
        print(f"\n  Writing {len(patches)} to Supabase...")
        ok = 0; err = 0
        for i, (gid, data) in enumerate(patches.items()):
            for attempt in range(3):
                try:
                    url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{gid}"
                    r = requests.patch(url, headers=SB_HEADERS, json=data, timeout=15)
                    if r.ok: ok += 1; break
                    time.sleep(1)
                except:
                    time.sleep(2)
            else:
                err += 1
            if (i+1) % 500 == 0:
                print(f"    {i+1}/{len(patches)} — {ok} ok, {err} err")
        print(f"  Done: {ok}/{len(patches)} ({err} errors)")
    elif dry_run:
        print(f"  DRY RUN — would write {len(patches)}")
    
    return matched

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NCAA SPREAD INGESTION FROM CSV")
    print("=" * 70)
    
    total = 0
    for csv_name, (season, desc) in CSV_FILES.items():
        path = csv_name
        if not os.path.exists(path):
            print(f"\n  ⚠ {csv_name} not found — skipping")
            continue
        total += process_csv(path, season, dry_run=args.dry_run)
    
    print(f"\n  Total matched: {total}")
    print(f"\n  Done. Next: refresh cache + retrain")
