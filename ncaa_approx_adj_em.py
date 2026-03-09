#!/usr/bin/env python3
"""
ncaa_approx_adj_em.py — Approximate KenPom adj_em from PIT data
═══════════════════════════════════════════════════════════════
Only ~2,600 of 64,881 games have real adj_em (from current-season sync).
This fills the other 62K from data already in Supabase:

  adj_oe ≈ PPP × 100  (offensive efficiency per 100 possessions)
  adj_de ≈ opp_ppg / (tempo / 100) (defensive efficiency per 100 possessions)
  adj_em = adj_oe - adj_de

Also computes from Elo as a cross-check:
  adj_em_elo ≈ (elo - 1500) / 25

Run:
  SUPABASE_ANON_KEY="..." python3 ncaa_approx_adj_em.py
"""
import os, sys, json, time, requests
import numpy as np

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_ANON_KEY"); sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=60)
        if not r.ok: print(f"  Error: {r.text[:200]}"); break
        data = r.json()
        if not data: break
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data

def sb_patch(table, match_col, match_val, patch_data):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok


def main():
    print("=" * 70)
    print("  APPROXIMATE adj_em FROM PIT DATA")
    print("  Filling 62K games missing KenPom efficiency margins")
    print("=" * 70)

    # ── Load all games with PIT data but missing adj_em ──
    print("\n  Loading games...")
    # Get games that have PIT data (home_ppp populated) 
    all_games = sb_get("ncaa_historical",
                       "actual_home_score=not.is.null"
                       "&home_ppp=not.is.null"
                       "&select=game_id,home_ppp,away_ppp,home_opp_ppg,away_opp_ppg,"
                       "home_tempo,away_tempo,home_elo,away_elo,"
                       "home_adj_em,away_adj_em,home_adj_oe,away_adj_oe,"
                       "home_adj_de,away_adj_de,season"
                       "&order=season.asc,game_date.asc")
    
    print(f"  Loaded {len(all_games)} games with PIT data")

    # ── Compute league averages per season ──
    print("  Computing per-season league averages...")
    season_stats = {}
    from collections import defaultdict
    season_ppp = defaultdict(list)
    season_oppg = defaultdict(list)
    season_tempo = defaultdict(list)

    for g in all_games:
        s = g.get("season")
        if not s: continue
        h_ppp = g.get("home_ppp")
        a_ppp = g.get("away_ppp")
        h_oppg = g.get("home_opp_ppg")
        a_oppg = g.get("away_opp_ppg")
        h_tempo = g.get("home_tempo")
        a_tempo = g.get("away_tempo")
        
        if h_ppp and float(h_ppp) > 0: season_ppp[s].append(float(h_ppp))
        if a_ppp and float(a_ppp) > 0: season_ppp[s].append(float(a_ppp))
        if h_oppg and float(h_oppg) > 0: season_oppg[s].append(float(h_oppg))
        if a_oppg and float(a_oppg) > 0: season_oppg[s].append(float(a_oppg))
        if h_tempo and float(h_tempo) > 0: season_tempo[s].append(float(h_tempo))
        if a_tempo and float(a_tempo) > 0: season_tempo[s].append(float(a_tempo))

    for s in sorted(season_ppp.keys()):
        avg_ppp = np.mean(season_ppp[s])
        avg_oppg = np.mean(season_oppg[s])
        avg_tempo = np.mean(season_tempo[s])
        season_stats[s] = {
            "avg_ppp": avg_ppp,
            "avg_oe": avg_ppp * 100,      # per 100 possessions
            "avg_oppg": avg_oppg,
            "avg_tempo": avg_tempo,
            "avg_de": avg_oppg / max(avg_tempo, 1) * 100,  # defensive eff per 100 poss
        }
        print(f"    Season {s}: avg_oe={avg_ppp*100:.1f}, avg_de={avg_oppg/max(avg_tempo,1)*100:.1f}, tempo={avg_tempo:.1f}")

    # ── Compute approximate adj_em for each game ──
    print(f"\n  Computing approximate adj_em for each game...")
    updates = []
    already_has = 0
    computed = 0
    skipped = 0

    for g in all_games:
        gid = g.get("game_id")
        if not gid: continue

        # Skip if already has real adj_em
        h_em = g.get("home_adj_em")
        a_em = g.get("away_adj_em")
        if h_em is not None and a_em is not None and float(h_em or 0) != 0 and float(a_em or 0) != 0:
            already_has += 1
            continue

        season = g.get("season")
        ss = season_stats.get(season)
        if not ss:
            skipped += 1
            continue

        h_ppp = float(g.get("home_ppp") or 0)
        a_ppp = float(g.get("away_ppp") or 0)
        h_oppg = float(g.get("home_opp_ppg") or 0)
        a_oppg = float(g.get("away_opp_ppg") or 0)
        h_tempo = float(g.get("home_tempo") or 68)
        a_tempo = float(g.get("away_tempo") or 68)
        h_elo = float(g.get("home_elo") or 1500)
        a_elo = float(g.get("away_elo") or 1500)

        if h_ppp <= 0 or a_ppp <= 0:
            skipped += 1
            continue

        # ── Method 1: PPP-based (primary) ──
        # Offensive efficiency = PPP × 100 (points per 100 possessions)
        h_adj_oe = h_ppp * 100
        a_adj_oe = a_ppp * 100

        # Defensive efficiency = opponent PPG / (team tempo) × 100
        # This approximates points allowed per 100 possessions
        if h_tempo > 0 and h_oppg > 0:
            h_adj_de = h_oppg / h_tempo * 100
        else:
            h_adj_de = ss["avg_de"]

        if a_tempo > 0 and a_oppg > 0:
            a_adj_de = a_oppg / a_tempo * 100
        else:
            a_adj_de = ss["avg_de"]

        # Efficiency margin = offense - defense
        # Positive = team scores more than it allows per 100 possessions
        h_adj_em_ppp = h_adj_oe - h_adj_de
        a_adj_em_ppp = a_adj_oe - a_adj_de

        # ── Method 2: Elo-based (secondary, for cross-check) ──
        # Elo difference of 400 ≈ 10:1 expected win ratio
        # In NCAA, adj_em range is roughly -30 to +35
        # Elo range is roughly 1100 to 2300 (center 1500)
        # Scale: adj_em ≈ (elo - 1500) / 25
        h_adj_em_elo = (h_elo - 1500) / 25
        a_adj_em_elo = (a_elo - 1500) / 25

        # ── Blend: 70% PPP-based + 30% Elo-based ──
        # PPP is more direct but noisier early in season
        # Elo is smoother but less granular
        h_adj_em_final = 0.7 * h_adj_em_ppp + 0.3 * h_adj_em_elo
        a_adj_em_final = 0.7 * a_adj_em_ppp + 0.3 * a_adj_em_elo

        # Also compute adj_oe and adj_de since heuristic uses them
        patch = {
            "home_adj_em": round(h_adj_em_final, 2),
            "away_adj_em": round(a_adj_em_final, 2),
            "home_adj_oe": round(h_adj_oe, 2),
            "away_adj_oe": round(a_adj_oe, 2),
            "home_adj_de": round(h_adj_de, 2),
            "away_adj_de": round(a_adj_de, 2),
        }

        updates.append((gid, patch))
        computed += 1

    print(f"\n  Results:")
    print(f"    Already has real adj_em: {already_has}")
    print(f"    Computed approximate:    {computed}")
    print(f"    Skipped (no data):       {skipped}")

    # ── Sanity check ──
    if updates:
        sample = updates[len(updates)//2][1]
        print(f"\n  Sample values (mid-dataset):")
        print(f"    home_adj_em: {sample['home_adj_em']}")
        print(f"    away_adj_em: {sample['away_adj_em']}")
        print(f"    home_adj_oe: {sample['home_adj_oe']}")
        print(f"    home_adj_de: {sample['home_adj_de']}")

        # Distribution check
        all_em = [u[1]["home_adj_em"] for u in updates] + [u[1]["away_adj_em"] for u in updates]
        print(f"\n  adj_em distribution:")
        print(f"    min:    {min(all_em):.1f}")
        print(f"    25th:   {np.percentile(all_em, 25):.1f}")
        print(f"    median: {np.median(all_em):.1f}")
        print(f"    75th:   {np.percentile(all_em, 75):.1f}")
        print(f"    max:    {max(all_em):.1f}")
        print(f"    std:    {np.std(all_em):.1f}")

    # ── Push to Supabase ──
    print(f"\n  Pushing {len(updates)} updates to Supabase...")
    success = 0
    for i, (gid, patch) in enumerate(updates):
        if sb_patch("ncaa_historical", "game_id", gid, patch):
            success += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(updates)} ({success} ok)")
        time.sleep(0.02)

    print(f"\n{'='*70}")
    print(f"  adj_em APPROXIMATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Updated: {success}/{len(updates)}")
    print(f"")
    print(f"  WHAT THIS DID:")
    print(f"  - Computed adj_oe from PPP × 100 (points per 100 possessions)")
    print(f"  - Computed adj_de from opp_ppg / tempo × 100")
    print(f"  - Blended 70% PPP-based + 30% Elo-based for adj_em")
    print(f"  - Preserved real adj_em for {already_has} current-season games")
    print(f"")
    print(f"  NEXT: Retrain locally:")
    print(f"    python3 ncaa_train_local.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
