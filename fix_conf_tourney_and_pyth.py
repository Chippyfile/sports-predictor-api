#!/usr/bin/env python3
"""
fix_conf_tourney_and_pyth.py — Fix two broken features in ncaa_historical
=========================================================================
1. is_conference_tournament: Flag games with ESPN season_type=22
2. pyth_residual: Recalculate as actual_win% - pythagorean_expected_win%

Usage:
  python3 -u fix_conf_tourney_and_pyth.py --check     # preview what would change
  python3 -u fix_conf_tourney_and_pyth.py              # fix and push to Supabase
"""
import sys, os, json, time, argparse, math
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fix_pyth_residual(df, check_only=False):
    """
    Pythagorean expectation: expected_win% = ppg^exp / (ppg^exp + opp_ppg^exp)
    Residual = actual_win% - expected_win%
    
    Positive = team overperforms (clutch/lucky)
    Negative = team underperforms
    """
    print("\n" + "=" * 70)
    print("  FIXING PYTH_RESIDUAL")
    print("=" * 70)

    EXP = 11.5  # Standard CBB Pythagorean exponent

    patches = []
    fixed = 0
    skipped = 0

    for _, row in df.iterrows():
        game_id = row["game_id"]
        patch = {}

        for side in ["home", "away"]:
            ppg = pd.to_numeric(row.get(f"{side}_ppg"), errors="coerce")
            opp_ppg = pd.to_numeric(row.get(f"{side}_opp_ppg"), errors="coerce")
            wins = pd.to_numeric(row.get(f"{side}_wins"), errors="coerce")
            losses = pd.to_numeric(row.get(f"{side}_losses"), errors="coerce")

            if pd.isna(ppg) or pd.isna(opp_ppg) or pd.isna(wins) or pd.isna(losses):
                continue
            if ppg <= 0 or opp_ppg <= 0:
                continue

            total_games = wins + losses
            if total_games == 0:
                continue

            actual_wp = wins / total_games
            ppg_exp = ppg ** EXP
            opp_exp = opp_ppg ** EXP
            expected_wp = ppg_exp / (ppg_exp + opp_exp)
            residual = round(actual_wp - expected_wp, 4)

            patch[f"{side}_pyth_residual"] = residual

        if patch:
            patches.append({"game_id": game_id, **patch})
            fixed += 1
        else:
            skipped += 1

    # Show sample
    print(f"  Computed: {fixed} games, skipped: {skipped}")
    if patches:
        sample = patches[:5]
        for p in sample:
            hr = p.get("home_pyth_residual", "N/A")
            ar = p.get("away_pyth_residual", "N/A")
            print(f"    {p['game_id']}: home={hr}, away={ar}")

        # Distribution check
        h_vals = [p["home_pyth_residual"] for p in patches if "home_pyth_residual" in p]
        if h_vals:
            print(f"\n  home_pyth_residual distribution:")
            print(f"    mean={np.mean(h_vals):.4f}  std={np.std(h_vals):.4f}")
            print(f"    min={np.min(h_vals):.4f}  max={np.max(h_vals):.4f}")

    if check_only:
        print(f"\n  CHECK ONLY — no data pushed.")
        return patches

    # Push
    if patches:
        push_patches(patches, "pyth_residual fix")

    return patches


def fix_conf_tourney(df, check_only=False):
    """
    Query ESPN API for each game's season type.
    ESPN season_type mapping for NCAAM:
      2 = regular season
      3 = postseason (conference tournament + NCAA)
      
    Within postseason, conference tournaments typically have:
      - slug containing 'conference' or specific conf tourney names
      - OR we can check the seasonType.type field from ESPN summary
      
    Actually the simplest: ESPN /scoreboard endpoint returns
    groups and seasonType. type=3 + before Selection Sunday = conf tourney.
    
    Approach: For games already flagged is_ncaa_tournament=False that are 
    in March, check if they're neutral-site OR query ESPN.
    
    Faster approach: ESPN game summary has seasonType info.
    """
    print("\n" + "=" * 70)
    print("  FIXING IS_CONFERENCE_TOURNAMENT")
    print("=" * 70)

    # Get March games that aren't NCAA tournament
    march_mask = df["game_date"].str.contains("-(03|04)-", na=False)
    not_ncaa = df["is_ncaa_tournament"].astype(str).isin(["False", "0", "false", ""])
    candidates = df[march_mask & not_ncaa].copy()
    print(f"  March non-NCAA games to check: {len(candidates)}")

    if len(candidates) == 0:
        print("  No candidates found.")
        return []

    # Query ESPN for season type
    # ESPN scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}
    patches = []
    conf_tourney_count = 0
    errors = 0
    cache_file = "conf_tourney_cache.json"

    # Load cache
    try:
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"  Loaded cache: {len(cache)} entries")
    except:
        cache = {}

    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows()):
        game_id = str(row["game_id"])

        if game_id in cache:
            is_conf = cache[game_id]
        else:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}"
                r = requests.get(url, headers=ESPN_HEADERS, timeout=10)
                if r.ok:
                    data = r.json()
                    # Check header -> season -> type
                    season_type = data.get("header", {}).get("season", {}).get("type", 2)
                    
                    # Check conferenceCompetition flag on competition object
                    comps = data.get("header", {}).get("competitions", [{}])
                    is_conf_comp = comps[0].get("conferenceCompetition", False) if comps else False
                    is_ncaa = row.get("is_ncaa_tournament", False)
                    if str(is_ncaa).lower() in ["true", "1"]:
                        is_conf = False
                    elif is_conf_comp:
                        is_conf = True
                    else:
                        is_conf = False

                    cache[game_id] = is_conf
                else:
                    cache[game_id] = False
                    errors += 1

                # Rate limit
                time.sleep(0.15)

            except Exception as e:
                cache[game_id] = False
                errors += 1

        if is_conf:
            patches.append({
                "game_id": int(game_id),
                "is_conference_tournament": True,
            })
            conf_tourney_count += 1

        if (i + 1) % 500 == 0 or i == total - 1:
            # Save cache periodically
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            elapsed_pct = (i + 1) / total * 100
            print(f"    {i+1}/{total} ({elapsed_pct:.1f}%) | conf_tourney: {conf_tourney_count} | errors: {errors}")

    # Final cache save
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    print(f"\n  Conference tournament games found: {conf_tourney_count}")
    print(f"  Errors: {errors}")

    if check_only:
        print(f"  CHECK ONLY — no data pushed.")
        return patches

    if patches:
        push_patches(patches, "is_conference_tournament fix")

    return patches


def push_patches(patches, label):
    """Push patches to Supabase."""
    print(f"\n  Pushing {len(patches)} patches ({label})...")
    success = 0
    errors = 0

    for i, patch in enumerate(patches):
        game_id = patch.pop("game_id")
        try:
            url = f"{SUPABASE_URL}/rest/v1/ncaa_historical?game_id=eq.{game_id}"
            r = requests.patch(url, headers=HEADERS, json=patch, timeout=30)
            if r.ok:
                success += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"    Error on {game_id}: {r.text[:100]}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Exception on {game_id}: {e}")

        if (i + 1) % 500 == 0 or i == len(patches) - 1:
            print(f"    {i+1}/{len(patches)} | success:{success} errors:{errors}")

    print(f"  PUSH COMPLETE: {success} success, {errors} errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Preview only, don't push")
    parser.add_argument("--pyth-only", action="store_true", help="Only fix pyth_residual")
    parser.add_argument("--conf-only", action="store_true", help="Only fix conf tournament")
    args = parser.parse_args()

    print("=" * 70)
    print("  FIX CONFERENCE TOURNAMENT + PYTHAGOREAN RESIDUAL")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    t0 = time.time()
    cols = "game_id,game_date,season,home_ppg,away_ppg,home_opp_ppg,away_opp_ppg,home_wins,away_wins,home_losses,away_losses,is_ncaa_tournament,is_conference_tournament,is_postseason,neutral_site,home_pyth_residual,away_pyth_residual"
    rows = sb_get("ncaa_historical", f"select={cols}&order=season.asc")
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} games in {time.time()-t0:.0f}s")

    if not args.conf_only:
        fix_pyth_residual(df, check_only=args.check)

    if not args.pyth_only:
        fix_conf_tourney(df, check_only=args.check)

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
