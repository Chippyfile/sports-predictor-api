#!/usr/bin/env python3
"""
extract_clutch_ft.py — Extract clutch free throw data from ESPN PBP
=====================================================================
Clutch = last 5 min of 2H/OT, margin <= 8 points.

Extracts: home_clutch_fta, home_clutch_ftm, away_clutch_fta, away_clutch_ftm
Then computes: home_clutch_ft_pct, away_clutch_ft_pct (clutch FT%)
Then computes rolling averages: home_roll_clutch_ft_pct, away_roll_clutch_ft_pct

Usage:
  python3 -u extract_clutch_ft.py --check    # test on 20 games
  python3 -u extract_clutch_ft.py            # extract + push all
  python3 -u extract_clutch_ft.py --rolling  # just compute + push rolling averages
"""
import sys, os, json, time, argparse
import requests
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}

CLUTCH_MINUTES = 5       # Last 5 minutes
CLUTCH_MARGIN = 8        # Margin <= 8 points
CLUTCH_SECS = CLUTCH_MINUTES * 60

CACHE_FILE = "clutch_ft_cache.json"


def extract_clutch_ft(game_id):
    """Extract clutch FT stats from ESPN PBP for a single game."""
    url = f"{ESPN_BASE}?event={game_id}"
    try:
        r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        if not r.ok:
            return None
        data = r.json()
    except:
        return None

    plays = data.get("plays", [])
    if not plays:
        return None

    # Get home/away team IDs from header
    comps = data.get("header", {}).get("competitions", [{}])
    if not comps:
        return None

    competitors = comps[0].get("competitors", [])
    home_tid = None
    away_tid = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home_tid = str(c.get("team", {}).get("id", ""))
        elif c.get("homeAway") == "away":
            away_tid = str(c.get("team", {}).get("id", ""))

    result = {
        "home_clutch_fta": 0, "home_clutch_ftm": 0,
        "away_clutch_fta": 0, "away_clutch_ftm": 0,
    }

    for p in plays:
        period = p.get("period", {}).get("number", 0)
        clock = p.get("clock", {}).get("displayValue", "")
        text = p.get("text", "")
        home_score = int(p.get("homeScore", 0) or 0)
        away_score = int(p.get("awayScore", 0) or 0)
        margin = abs(home_score - away_score)
        play_team_id = str(p.get("team", {}).get("id", ""))

        # Parse clock to seconds remaining
        try:
            parts = clock.split(":")
            secs = int(parts[0]) * 60 + int(parts[1])
        except:
            continue

        # Clutch: 2H (period 2) or OT (period >= 3), <=5 min left, margin <= 8
        is_clutch = (period >= 2 and secs <= CLUTCH_SECS and margin <= CLUTCH_MARGIN)

        if is_clutch and "free throw" in text.lower():
            # Determine home/away
            if play_team_id == home_tid:
                side = "home"
            elif play_team_id == away_tid:
                side = "away"
            else:
                # Try homeAway field
                ha = p.get("homeAway", "")
                side = "home" if ha == "home" else "away"

            result[f"{side}_clutch_fta"] += 1
            if "made" in text.lower():
                result[f"{side}_clutch_ftm"] += 1

    return result


def compute_rolling_clutch(df, window=10):
    """Compute rolling clutch FT% from per-game clutch data."""
    df = df.sort_values(["game_date", "game_id"]).copy()

    # Build per-team game list
    teams = {}  # tid -> list of (game_date, game_id, clutch_fta, clutch_ftm)
    for _, row in df.iterrows():
        gid = row["game_id"]
        gdate = row["game_date"]
        for side, tid_col in [("home", "home_team_id"), ("away", "away_team_id")]:
            tid = str(row.get(tid_col, ""))
            if not tid:
                continue
            fta = row.get(f"{side}_clutch_fta", 0) or 0
            ftm = row.get(f"{side}_clutch_ftm", 0) or 0
            if tid not in teams:
                teams[tid] = []
            teams[tid].append({
                "game_id": gid, "game_date": gdate,
                "fta": float(fta), "ftm": float(ftm), "side": side,
            })

    # Sort each team's games chronologically
    for tid in teams:
        teams[tid].sort(key=lambda x: (x["game_date"], x["game_id"]))

    # Compute rolling clutch FT% for each game
    rolling_data = {}  # game_id -> {home_roll_clutch_ft_pct, away_roll_clutch_ft_pct}
    for tid, games in teams.items():
        for i, g in enumerate(games):
            # Look back at prior `window` games
            prior = games[max(0, i - window):i]
            if len(prior) >= 3:  # need at least 3 prior games
                total_fta = sum(p["fta"] for p in prior)
                total_ftm = sum(p["ftm"] for p in prior)
                roll_pct = total_ftm / max(total_fta, 1)
            else:
                roll_pct = 0.75  # league average default

            gid = g["game_id"]
            side = g["side"]
            if gid not in rolling_data:
                rolling_data[gid] = {}
            rolling_data[gid][f"{side}_roll_clutch_ft_pct"] = round(roll_pct, 4)

    return rolling_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Test on 20 games only")
    parser.add_argument("--rolling", action="store_true", help="Just compute + push rolling averages")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXTRACT CLUTCH FREE THROW DATA")
    print(f"  Clutch = last {CLUTCH_MINUTES}min, margin <= {CLUTCH_MARGIN}pts")
    print("=" * 70)

    if not args.rolling:
        # ── Phase 1: Extract per-game clutch FT from ESPN PBP ───────────
        print("\n  Loading game IDs...")
        rows = sb_get("ncaa_historical",
                       "select=game_id,game_date,home_team_id,away_team_id,home_clutch_fta&order=game_date.asc")
        df = pd.DataFrame(rows)
        print(f"  Total games: {len(df)}")

        # Skip games that already have clutch data
        already_done = df[pd.to_numeric(df["home_clutch_fta"], errors="coerce").fillna(0) > 0]
        print(f"  Already have clutch data: {len(already_done)}")

        # Load cache
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print(f"  Loaded cache: {len(cache)} entries")
        except:
            cache = {}

        # Filter to games needing extraction (not in cache, no data)
        to_extract = df[~df["game_id"].astype(str).isin(cache.keys())]
        to_extract = to_extract[pd.to_numeric(to_extract["home_clutch_fta"], errors="coerce").fillna(0) == 0]
        print(f"  Games to extract: {len(to_extract)}")

        if args.check:
            to_extract = to_extract.head(20)
            print(f"  CHECK MODE: testing {len(to_extract)} games")

        # Extract
        patches = []
        total = len(to_extract)
        games_with_clutch = 0
        total_clutch_fta = 0

        for i, (_, row) in enumerate(to_extract.iterrows()):
            game_id = str(row["game_id"])

            if game_id in cache:
                result = cache[game_id]
            else:
                result = extract_clutch_ft(game_id)
                cache[game_id] = result
                time.sleep(0.15)  # Rate limit

            if result:
                fta_total = result["home_clutch_fta"] + result["away_clutch_fta"]
                if fta_total > 0:
                    games_with_clutch += 1
                    total_clutch_fta += fta_total

                patches.append({"game_id": int(game_id), **result})

            if (i + 1) % 500 == 0 or i == total - 1:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f)
                pct = (i + 1) / total * 100
                rate = (i + 1) / max(time.time() - t0, 1) if 't0' in dir() else 0
                remaining = (total - i - 1) * 0.15 / 60
                print(f"    {i+1}/{total} ({pct:.1f}%) | with_clutch: {games_with_clutch} | "
                      f"avg_fta: {total_clutch_fta/max(games_with_clutch,1):.1f} | "
                      f"~{remaining:.0f}min left")

            if i == 0:
                t0 = time.time()

        # Save cache
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

        print(f"\n  Extraction complete: {games_with_clutch}/{len(patches)} games had clutch FTs")
        print(f"  Average clutch FTA per game (when present): {total_clutch_fta/max(games_with_clutch,1):.1f}")

        if args.check:
            print("\n  CHECK MODE — sample results:")
            for p in patches[:10]:
                gid = p["game_id"]
                h_pct = p["home_clutch_ftm"] / max(p["home_clutch_fta"], 1)
                a_pct = p["away_clutch_ftm"] / max(p["away_clutch_fta"], 1)
                print(f"    {gid}: home {p['home_clutch_ftm']}/{p['home_clutch_fta']} ({h_pct:.0%}) "
                      f"away {p['away_clutch_ftm']}/{p['away_clutch_fta']} ({a_pct:.0%})")
            print("  No data pushed.")
            return

        # Push per-game clutch data
        if patches:
            print(f"\n  Pushing {len(patches)} per-game clutch patches...")
            success, errors = 0, 0
            for i, patch in enumerate(patches):
                game_id = patch.pop("game_id")
                try:
                    url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
                    r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
                    if r.ok:
                        success += 1
                    else:
                        errors += 1
                except:
                    errors += 1

                if (i + 1) % 500 == 0 or i == len(patches) - 1:
                    print(f"    {i+1}/{len(patches)} | success:{success} errors:{errors}")

            print(f"  Per-game push complete: {success} success, {errors} errors")

    # ── Phase 2: Compute + push rolling clutch FT% ──────────────────
    print(f"\n{'=' * 70}")
    print("  COMPUTING ROLLING CLUTCH FT%")
    print(f"{'=' * 70}")

    print("  Loading clutch data...")
    rows = sb_get("ncaa_historical",
                   "select=game_id,game_date,home_team_id,away_team_id,home_clutch_fta,home_clutch_ftm,away_clutch_fta,away_clutch_ftm&order=game_date.asc")
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} games")

    # Check coverage
    has_data = pd.to_numeric(df["home_clutch_fta"], errors="coerce").fillna(0) > 0
    has_data |= pd.to_numeric(df["away_clutch_fta"], errors="coerce").fillna(0) > 0
    print(f"  Games with clutch FT data: {has_data.sum()}/{len(df)} ({has_data.mean()*100:.1f}%)")

    rolling_data = compute_rolling_clutch(df, window=10)
    print(f"  Computed rolling clutch FT% for {len(rolling_data)} games")

    # Sample distribution
    h_vals = [v.get("home_roll_clutch_ft_pct", 0) for v in rolling_data.values() if "home_roll_clutch_ft_pct" in v]
    if h_vals:
        print(f"  home_roll_clutch_ft_pct: mean={np.mean(h_vals):.4f} std={np.std(h_vals):.4f} "
              f"range=[{np.min(h_vals):.4f}, {np.max(h_vals):.4f}]")

    # Push rolling data
    patches = []
    for game_id, vals in rolling_data.items():
        patch = {"game_id": int(game_id)}
        patch.update(vals)
        patches.append(patch)

    if patches:
        print(f"\n  Pushing {len(patches)} rolling clutch patches...")
        success, errors = 0, 0
        for i, patch in enumerate(patches):
            game_id = patch.pop("game_id")
            try:
                url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
                r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
                if r.ok:
                    success += 1
                else:
                    errors += 1
            except:
                errors += 1

            if (i + 1) % 500 == 0 or i == len(patches) - 1:
                remaining = (len(patches) - i - 1) * 0.002 / 60
                print(f"    {i+1}/{len(patches)} | success:{success} errors:{errors}")

        print(f"  Rolling push complete: {success} success, {errors} errors")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
