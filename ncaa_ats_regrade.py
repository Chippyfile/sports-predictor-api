#!/usr/bin/env python3
"""
NCAA ATS Regrade Script
Fixes rl_correct using correct formula: margin + spread > 0
Also nulls out rl_correct for games without market spread.
Run from: ~/Desktop/sports-predictor-api
"""

import requests
import json
import time

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co/rest/v1/ncaa_predictions"
SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4YWFxdHF2bHdqdnl1ZWR5YXVvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTgwNjM1NSwiZXhwIjoyMDg3MzgyMzU1fQ.m9D8hP71LjKnYT3MPxHPtYS4aSD6TX5lMA7286T_L_U"

HEADERS = {
    "apikey": SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

def fetch_batch(params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(f"{SUPABASE_URL}?{params}", headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ❌ Fetch failed after {retries} attempts: {e}")
                return []

def patch_row(row_id, body, retries=3):
    for attempt in range(retries):
        try:
            r = requests.patch(
                f"{SUPABASE_URL}?id=eq.{row_id}",
                headers=HEADERS, json=body, timeout=15
            )
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                print(f"  ❌ Patch failed for id={row_id}: {e}")
                return False

def main():
    print("=" * 60)
    print("NCAA ATS REGRADE — Correct formula: margin + spread > 0")
    print("=" * 60)
    
    # PHASE 1: Fix games WITH market spread
    print("\nPhase 1: Correcting ATS grades for games with market spread...")
    offset = 0
    batch_size = 500
    fixed = 0
    total = 0
    unchanged = 0
    
    while True:
        params = (
            f"result_entered=eq.true&market_spread_home=not.is.null"
            f"&actual_home_score=not.is.null&actual_away_score=not.is.null"
            f"&select=id,actual_home_score,actual_away_score,market_spread_home,rl_correct"
            f"&order=id.asc&limit={batch_size}&offset={offset}"
        )
        rows = fetch_batch(params)
        if not rows:
            break
        
        for g in rows:
            total += 1
            margin = g["actual_home_score"] - g["actual_away_score"]
            spread = g["market_spread_home"]
            ats = margin + spread
            
            if ats > 0:
                new_rl = True
            elif ats < 0:
                new_rl = False
            else:
                new_rl = None  # push
            
            if new_rl != g["rl_correct"]:
                if patch_row(g["id"], {"rl_correct": new_rl}):
                    fixed += 1
                # Small delay to avoid rate limits
                if fixed % 50 == 0 and fixed > 0:
                    time.sleep(0.5)
            else:
                unchanged += 1
        
        print(f"  Batch {offset // batch_size + 1}: {total} games checked, {fixed} fixed, {unchanged} already correct")
        offset += batch_size
        time.sleep(0.3)  # Brief pause between batches
    
    print(f"\n  ✅ Phase 1 complete: {fixed} corrected, {unchanged} unchanged out of {total} with market spread")
    
    # PHASE 2: Null out rl_correct for games WITHOUT market spread
    print("\nPhase 2: Nulling rl_correct for games without market spread...")
    params2 = (
        "result_entered=eq.true&market_spread_home=is.null"
        "&rl_correct=not.is.null&select=id&limit=5000"
    )
    no_market = fetch_batch(params2)
    nulled = 0
    
    for g in no_market:
        if patch_row(g["id"], {"rl_correct": None}):
            nulled += 1
        if nulled % 100 == 0 and nulled > 0:
            print(f"  Nulled {nulled}...")
            time.sleep(0.5)
    
    print(f"\n  ✅ Phase 2 complete: {nulled} fallback grades removed")
    
    # SUMMARY
    print("\n" + "=" * 60)
    print(f"DONE: {fixed} ATS corrected, {nulled} fallbacks nulled, {total} total with market data")
    print("=" * 60)
    
    # Verify with Virginia
    print("\nVerification — Virginia Cavaliers (spread=-14.5, margin=+5):")
    verify = fetch_batch("home_team_name=like.*Virginia Cavaliers*&market_spread_home=eq.-14.5&select=home_team_name,market_spread_home,actual_home_score,actual_away_score,rl_correct&limit=1")
    if verify:
        g = verify[0]
        margin = g["actual_home_score"] - g["actual_away_score"]
        print(f"  rl_correct = {g['rl_correct']}  (should be False)")
        print(f"  margin+spread = {margin + g['market_spread_home']}")

if __name__ == "__main__":
    main()
