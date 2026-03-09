#!/usr/bin/env python3
"""
mlb_historical_backfill_stats.py
════════════════════════════════
Backfills missing bullpen_era and sp_ip columns in mlb_historical.

The MLB historical table already has: wOBA, FIP, SP FIP, K/9, BB/9, park_factor,
temp, wind, rest_days, travel, def_oaa. What's MISSING: bullpen_era, sp_ip.

Strategy:
  - bullpen_era: Approximate from team FIP and SP FIP.
    bullpen_era ≈ team_fip * 1.05 (bullpens typically run ~5% worse than team avg)
    If SP FIP is available: bullpen_era ≈ (team_fip * innings - sp_fip * sp_ip) / (innings - sp_ip)
  - sp_ip: Default to 5.5 (league average) since we don't have per-game SP IP
    in historical data. This is still better than nothing — the feature exists
    but won't have game-level variance.

For platoon_delta: historical data doesn't have L/R matchup info, so these
stay at 0. The has_market flag already tells the model when platoon data
is unavailable.

Run from sports-predictor-api root:
    SUPABASE_ANON_KEY="..." python3 mlb_historical_backfill_stats.py
"""
import os, sys, requests, time
import numpy as np

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY environment variable")
    sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def sb_get(table, params=""):
    all_data = []
    offset = 0
    limit = 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }, timeout=30)
        if not r.ok:
            print(f"  Error: {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data


def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok


def main():
    print("=" * 70)
    print("  MLB HISTORICAL BACKFILL — Bullpen ERA + SP IP")
    print("=" * 70)

    # Load all historical games
    print("\n  Loading mlb_historical...")
    rows = sb_get("mlb_historical",
                  "actual_home_runs=not.is.null&select=game_pk,home_fip,away_fip,home_sp_fip,away_sp_fip,home_bullpen_era,away_bullpen_era,home_sp_ip,away_sp_ip&order=game_pk.asc")
    print(f"  Loaded {len(rows)} games")

    if not rows:
        print("  No data!")
        return

    # Check how many already have bullpen_era
    has_bp = sum(1 for r in rows if r.get("home_bullpen_era") is not None)
    print(f"  Already have bullpen_era: {has_bp}/{len(rows)}")

    if has_bp == len(rows):
        print("  All rows already have bullpen_era — nothing to backfill")
        return

    # Compute bullpen ERA from team FIP and SP FIP
    updates = []
    for r in rows:
        if r.get("home_bullpen_era") is not None and r.get("away_bullpen_era") is not None:
            continue  # already populated

        game_pk = r.get("game_pk")
        if not game_pk:
            continue

        patch = {}

        # Home bullpen ERA approximation
        h_fip = r.get("home_fip")
        h_sp_fip = r.get("home_sp_fip")
        if r.get("home_bullpen_era") is None and h_fip is not None:
            try:
                h_fip = float(h_fip)
                if h_sp_fip is not None:
                    h_sp_fip = float(h_sp_fip)
                    # Estimate: bullpen = (team_total - starter_contribution) / bullpen_innings
                    # Assume SP throws 5.5 IP out of 9, bullpen throws 3.5
                    sp_ip = 5.5
                    bp_ip = 3.5
                    total_era_9 = h_fip * 9  # total "earned runs" proxy over 9 IP
                    sp_contribution = h_sp_fip * sp_ip
                    bp_era = (total_era_9 - sp_contribution) / bp_ip
                    bp_era = max(2.0, min(7.0, bp_era))  # clamp to realistic range
                    patch["home_bullpen_era"] = round(bp_era, 2)
                else:
                    # No SP FIP — use team FIP * 1.05 as bullpen proxy
                    patch["home_bullpen_era"] = round(h_fip * 1.05, 2)
                patch["home_sp_ip"] = 5.5  # league average default
            except (ValueError, TypeError):
                pass

        # Away bullpen ERA approximation
        a_fip = r.get("away_fip")
        a_sp_fip = r.get("away_sp_fip")
        if r.get("away_bullpen_era") is None and a_fip is not None:
            try:
                a_fip = float(a_fip)
                if a_sp_fip is not None:
                    a_sp_fip = float(a_sp_fip)
                    sp_ip = 5.5
                    bp_ip = 3.5
                    total_era_9 = a_fip * 9
                    sp_contribution = a_sp_fip * sp_ip
                    bp_era = (total_era_9 - sp_contribution) / bp_ip
                    bp_era = max(2.0, min(7.0, bp_era))
                    patch["away_bullpen_era"] = round(bp_era, 2)
                else:
                    patch["away_bullpen_era"] = round(a_fip * 1.05, 2)
                patch["away_sp_ip"] = 5.5
            except (ValueError, TypeError):
                pass

        if patch:
            updates.append((game_pk, patch))

    print(f"\n  Updates to apply: {len(updates)}")

    # Push to Supabase
    if updates:
        print(f"  Pushing to Supabase...")
        success = 0
        for i, (game_pk, patch) in enumerate(updates):
            if sb_patch("mlb_historical", "game_pk", game_pk, patch):
                success += 1
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(updates)} done ({success} successful)...")
            time.sleep(0.02)  # Light rate limiting

        print(f"\n  ✅ Done: {success}/{len(updates)} rows updated")
    else:
        print("  Nothing to update")

    print(f"\n{'='*70}")
    print(f"  IMPACT ASSESSMENT:")
    print(f"  Before: bullpen_era_diff = 0 for all {len(rows)} historical games")
    print(f"          (both sides defaulted to 4.10, diff always 0)")
    print(f"  After:  bullpen_era_diff has real variance from team/SP FIP gap")
    print(f"          sp_ip_diff still ~0 (no per-game SP IP available)")
    print(f"          platoon_diff still 0 (no L/R matchup data)")
    print(f"")
    print(f"  Next: Retrain MLB model:")
    print(f"    curl -X POST $RAILWAY_API/train/mlb | python3 -m json.tool")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
