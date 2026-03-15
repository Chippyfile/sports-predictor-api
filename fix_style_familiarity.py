#!/usr/bin/env python3
"""
fix_style_familiarity.py — Fix cosine similarity bug (tempo dominating)
========================================================================
The style vector [tempo, efg_pct, to_rate, orb_pct, fta_rate, three_rate]
has tempo ~68 while others are 0.2-0.5, making cosine sim always ~1.0.

Fix: z-score normalize each component before cosine similarity.

Approach:
  1. Fetch all games chronologically with relevant style stats
  2. For each team, track cumulative style profile (running averages)
  3. For each team, track set of prior opponents
  4. For each game, compute: how similar is away team's style to the
     most similar prior opponent the home team has faced?
  5. Push corrected style_familiarity to Supabase

Usage:
  python3 -u fix_style_familiarity.py --check    # preview, no push
  python3 -u fix_style_familiarity.py             # compute + push
"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import requests

sys.path.insert(0, ".")
from config import SUPABASE_URL, SUPABASE_KEY
from db import sb_get

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

# Style vector components and their approximate league-wide stats for z-scoring
# These are used to normalize each dimension to roughly equal scale
STYLE_COMPONENTS = {
    "tempo":      {"mean": 70.0, "std": 5.0},
    "efg_pct":    {"mean": 0.50, "std": 0.04},
    "to_rate":    {"mean": 0.18, "std": 0.03},
    "orb_pct":    {"mean": 0.30, "std": 0.05},
    "fta_rate":   {"mean": 0.32, "std": 0.06},
    "three_rate": {"mean": 0.35, "std": 0.06},
}


def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / max(na * nb, 1e-9))


def normalize_style_vector(raw_vec):
    """Z-score normalize a style vector so all dimensions contribute equally."""
    keys = list(STYLE_COMPONENTS.keys())
    normalized = []
    for i, k in enumerate(keys):
        val = raw_vec[i] if i < len(raw_vec) else STYLE_COMPONENTS[k]["mean"]
        z = (val - STYLE_COMPONENTS[k]["mean"]) / max(STYLE_COMPONENTS[k]["std"], 1e-9)
        normalized.append(z)
    return normalized


def get_team_style(stats):
    """Build raw style vector from team's cumulative stats."""
    tempo = stats.get("tempo", 70.0)
    efg = stats.get("efg_pct", 0.50)
    to_rate = stats.get("turnovers", 13.0) / max(stats.get("tempo", 70.0), 1)
    orb = stats.get("orb_pct", 0.30)
    fta_rate = stats.get("fta_rate", 0.32)
    three_rate = stats.get("three_rate", 0.35)
    return [tempo, efg, to_rate, orb, fta_rate, three_rate]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  FIX STYLE_FAMILIARITY (normalized cosine similarity)")
    print("=" * 70)

    # Fetch games with style-relevant stats
    cols = ",".join([
        "game_id", "game_date", "season",
        "home_team_id", "away_team_id",
        "home_tempo", "away_tempo",
        "home_efg_pct", "away_efg_pct",
        "home_turnovers", "away_turnovers",
        "home_orb_pct", "away_orb_pct",
        "home_fta_rate", "away_fta_rate",
        "home_three_rate", "away_three_rate",
        "style_familiarity",
    ])

    print("\n  Loading data...")
    t0 = time.time()
    rows = sb_get("ncaa_historical", f"select={cols}&order=game_date.asc,game_id.asc")
    print(f"  Loaded {len(rows)} games in {time.time()-t0:.0f}s")

    if not rows:
        print("  ERROR: No data loaded")
        return

    # First pass: compute league-wide means/stds from actual data
    print("  Computing league-wide normalization stats...")
    all_tempos, all_efgs, all_to_rates, all_orbs, all_ftas, all_threes = [], [], [], [], [], []

    for r in rows:
        for side in ["home", "away"]:
            tempo = r.get(f"{side}_tempo")
            efg = r.get(f"{side}_efg_pct")
            to = r.get(f"{side}_turnovers")
            orb = r.get(f"{side}_orb_pct")
            fta = r.get(f"{side}_fta_rate")
            three = r.get(f"{side}_three_rate")

            if tempo is not None: all_tempos.append(float(tempo))
            if efg is not None: all_efgs.append(float(efg))
            if tempo and to is not None: all_to_rates.append(float(to) / max(float(tempo), 1))
            if orb is not None: all_orbs.append(float(orb))
            if fta is not None: all_ftas.append(float(fta))
            if three is not None: all_threes.append(float(three))

    # Update normalization constants from actual data
    if all_tempos:
        STYLE_COMPONENTS["tempo"]["mean"] = np.mean(all_tempos)
        STYLE_COMPONENTS["tempo"]["std"] = max(np.std(all_tempos), 0.01)
    if all_efgs:
        STYLE_COMPONENTS["efg_pct"]["mean"] = np.mean(all_efgs)
        STYLE_COMPONENTS["efg_pct"]["std"] = max(np.std(all_efgs), 0.001)
    if all_to_rates:
        STYLE_COMPONENTS["to_rate"]["mean"] = np.mean(all_to_rates)
        STYLE_COMPONENTS["to_rate"]["std"] = max(np.std(all_to_rates), 0.001)
    if all_orbs:
        STYLE_COMPONENTS["orb_pct"]["mean"] = np.mean(all_orbs)
        STYLE_COMPONENTS["orb_pct"]["std"] = max(np.std(all_orbs), 0.001)
    if all_ftas:
        STYLE_COMPONENTS["fta_rate"]["mean"] = np.mean(all_ftas)
        STYLE_COMPONENTS["fta_rate"]["std"] = max(np.std(all_ftas), 0.001)
    if all_threes:
        STYLE_COMPONENTS["three_rate"]["mean"] = np.mean(all_threes)
        STYLE_COMPONENTS["three_rate"]["std"] = max(np.std(all_threes), 0.001)

    print("  Normalization stats (from data):")
    for k, v in STYLE_COMPONENTS.items():
        print(f"    {k:<12}: mean={v['mean']:.4f}  std={v['std']:.4f}")

    # Second pass: process games chronologically
    print("\n  Computing style familiarity...")

    # Per-team state: running average of style stats + set of prior opponents
    team_stats = defaultdict(lambda: {
        "tempo": [], "efg_pct": [], "turnovers": [], "orb_pct": [],
        "fta_rate": [], "three_rate": [], "opponents": set()
    })

    patches = []
    unchanged = 0
    changed = 0
    no_data = 0

    for i, r in enumerate(rows):
        game_id = r["game_id"]
        h_tid = r.get("home_team_id")
        a_tid = r.get("away_team_id")

        if not h_tid or not a_tid:
            no_data += 1
            continue

        h_tid, a_tid = str(h_tid), str(a_tid)

        # Get away team's current style vector (from their running averages)
        a_stats = team_stats[a_tid]
        if a_stats["tempo"]:
            a_raw = [
                np.mean(a_stats["tempo"][-20:]),  # last 20 games rolling
                np.mean(a_stats["efg_pct"][-20:]),
                np.mean(a_stats["turnovers"][-20:]) / max(np.mean(a_stats["tempo"][-20:]), 1),
                np.mean(a_stats["orb_pct"][-20:]),
                np.mean(a_stats["fta_rate"][-20:]),
                np.mean(a_stats["three_rate"][-20:]),
            ]
            a_norm = normalize_style_vector(a_raw)
        else:
            a_norm = None

        # Get home team's prior opponents' style vectors
        h_stats = team_stats[h_tid]
        h_prior_opp_ids = h_stats["opponents"]

        if a_norm and h_prior_opp_ids:
            sims = []
            for opp_id in h_prior_opp_ids:
                opp_stats = team_stats[opp_id]
                if opp_stats["tempo"]:
                    opp_raw = [
                        np.mean(opp_stats["tempo"][-20:]),
                        np.mean(opp_stats["efg_pct"][-20:]),
                        np.mean(opp_stats["turnovers"][-20:]) / max(np.mean(opp_stats["tempo"][-20:]), 1),
                        np.mean(opp_stats["orb_pct"][-20:]),
                        np.mean(opp_stats["fta_rate"][-20:]),
                        np.mean(opp_stats["three_rate"][-20:]),
                    ]
                    opp_norm = normalize_style_vector(opp_raw)
                    sims.append(cosine_sim(a_norm, opp_norm))

            if sims:
                valid_sims = [s for s in sims if not np.isnan(s)]
                familiarity = round(max(valid_sims), 4) if valid_sims else 0.5
                familiarity = max(0.0, min(1.0, familiarity))  # clamp 0-1
            else:
                familiarity = 0.5
        else:
            familiarity = 0.5

        # Check if changed
        old_val = r.get("style_familiarity")
        if old_val is not None:
            try:
                old_val = round(float(old_val), 4)
            except:
                old_val = None

        if np.isnan(familiarity):
            familiarity = 0.5
        if old_val != familiarity:
            patches.append({"game_id": game_id, "style_familiarity": familiarity})
            changed += 1
        else:
            unchanged += 1

        # Update team stats with this game's data
        for side, tid in [("home", h_tid), ("away", a_tid)]:
            for stat in ["tempo", "efg_pct", "turnovers", "orb_pct", "fta_rate", "three_rate"]:
                val = r.get(f"{side}_{stat}")
                if val is not None:
                    try:
                        team_stats[tid][stat].append(float(val))
                    except:
                        pass

        # Track opponents
        team_stats[h_tid]["opponents"].add(a_tid)
        team_stats[a_tid]["opponents"].add(h_tid)

        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{len(rows)} | changed: {changed} | unchanged: {unchanged}")

    print(f"\n  Total: {changed} changed, {unchanged} unchanged, {no_data} no team data")

    # Distribution check
    if patches:
        vals = [p["style_familiarity"] for p in patches]
        print(f"\n  New style_familiarity distribution:")
        print(f"    mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        print(f"    min={np.min(vals):.4f}  max={np.max(vals):.4f}")
        print(f"    == 0.5 (default): {sum(1 for v in vals if v == 0.5)}")
        print(f"    == 1.0 (old bug): {sum(1 for v in vals if v == 1.0)}")

        # Bucket distribution
        buckets = defaultdict(int)
        for v in vals:
            buckets[round(v, 1)] += 1
        print(f"\n  Distribution by bucket:")
        for b in sorted(buckets):
            bar = "#" * (buckets[b] // 200)
            print(f"    {b:.1f}: {buckets[b]:>6} {bar}")

    if args.check:
        print(f"\n  CHECK ONLY — no data pushed.")
        return

    # Push
    if patches:
        print(f"\n  Pushing {len(patches)} patches...")
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
                    if errors <= 3:
                        print(f"    Error on {game_id}: {r.text[:100]}")
            except Exception as e:
                errors += 1

            if (i + 1) % 500 == 0 or i == len(patches) - 1:
                pct = (i + 1) / len(patches) * 100
                rate = (i + 1) / max(time.time() - t0, 1)
                remaining = (len(patches) - i - 1) / max(rate, 0.01) / 60
                print(f"    {i+1}/{len(patches)} ({pct:.1f}%) | success:{success} errors:{errors} | {remaining:.1f}min left")

        print(f"\n  PUSH COMPLETE: {success} success, {errors} errors")


if __name__ == "__main__":
    main()
