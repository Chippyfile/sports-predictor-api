#!/usr/bin/env python3
"""
March Madness 2026 Tournament Simulator
========================================
Uses the NCAA prediction model (/predict/ncaa/full endpoint) to simulate
the entire tournament via Monte Carlo methods.

Usage:
    python3 march_madness_sim.py [--sims 10000] [--api-url URL] [--output bracket_results.json]

Flow:
    1. Loads the 2026 bracket (hardcoded from Selection Sunday)
    2. For each matchup, calls /predict/ncaa/full to get predicted spread
    3. Runs N Monte Carlo simulations using Normal(spread, sigma)
    4. Tracks how often each team reaches each round
    5. Outputs probabilities + most likely bracket
"""

import argparse
import json
import requests
import sys
import time
import random
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════
# 2026 BRACKET DATA — Selection Sunday March 15, 2026
# ═══════════════════════════════════════════════════════════════
# Format: (seed, team_name, espn_team_id)
# ESPN team IDs are used to call /predict/ncaa/full
# NOTE: You may need to verify/update ESPN team IDs for smaller schools

EAST_REGION = [
    (1, "Duke", 150),
    (16, "Siena", 2547),
    (8, "Ohio State", 194),
    (9, "TCU", 2628),
    (5, "St. John's", 2599),
    (12, "Northern Iowa", 2460),
    (4, "Kansas", 2305),
    (13, "Cal Baptist", 2856),
    (6, "Louisville", 97),
    (11, "South Florida", 58),
    (3, "Michigan State", 127),
    (14, "North Dakota State", 2449),
    (7, "UCLA", 26),
    (10, "UCF", 2116),
    (2, "UConn", 41),
    (15, "Furman", 231),
]

WEST_REGION = [
    (1, "Arizona", 12),
    (16, "LIU", 112),
    (8, "Villanova", 2918),
    (9, "Utah State", 328),
    (5, "Wisconsin", 275),
    (12, "High Point", 2272),
    (4, "Arkansas", 8),
    (13, "Hawaii", 62),
    (6, "BYU", 252),
    (11, "Texas", 251),         # First Four winner (Texas vs NC State)
    (3, "Gonzaga", 2250),
    (14, "Kennesaw State", 338),
    (7, "Miami FL", 2390),
    (10, "Missouri", 142),
    (2, "Purdue", 2509),
    (15, "Queens", 2818),
]

MIDWEST_REGION = [
    (1, "Michigan", 130),
    (16, "UMBC", 2692),         # First Four winner (UMBC vs Howard)
    (8, "Georgia", 61),
    (9, "Saint Louis", 139),
    (5, "Texas Tech", 2641),
    (12, "Akron", 2006),
    (4, "Alabama", 333),
    (13, "Hofstra", 2275),
    (6, "Tennessee", 2633),
    (11, "SMU", 2567),          # First Four winner (SMU vs Miami OH)
    (3, "Virginia", 258),
    (14, "Wright State", 2750),
    (7, "Kentucky", 96),
    (10, "Santa Clara", 2491),
    (2, "Iowa State", 66),
    (15, "Tennessee State", 2634),
]

SOUTH_REGION = [
    (1, "Florida", 57),
    (16, "Lehigh", 2329),       # First Four winner (Prairie View vs Lehigh)
    (8, "Clemson", 228),
    (9, "Iowa", 2294),
    (5, "Vanderbilt", 238),
    (12, "McNeese", 2377),
    (4, "Nebraska", 158),
    (13, "Troy", 2653),
    (6, "North Carolina", 153),
    (11, "VCU", 2670),
    (3, "Illinois", 356),
    (14, "Penn", 219),
    (7, "Saint Mary's", 2608),
    (10, "Texas A&M", 245),
    (2, "Houston", 248),
    (15, "Idaho", 70),
]

ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG", "CHAMP"]

# ═══════════════════════════════════════════════════════════════
# API PREDICTION WRAPPER
# ═══════════════════════════════════════════════════════════════

class PredictionCache:
    """Cache API predictions to avoid redundant calls for the same matchup."""
    def __init__(self):
        self.cache = {}

    def key(self, team_a_id, team_b_id):
        return (min(team_a_id, team_b_id), max(team_a_id, team_b_id))

    def get(self, team_a_id, team_b_id):
        return self.cache.get(self.key(team_a_id, team_b_id))

    def set(self, team_a_id, team_b_id, result):
        self.cache[self.key(team_a_id, team_b_id)] = result


def fetch_prediction(api_url, home_team, away_team, cache, game_date="2026-03-20"):
    """
    Call /predict/ncaa/full for a matchup.
    Returns: (predicted_margin_for_home, sigma)
    """
    home_seed, home_name, home_id = home_team
    away_seed, away_name, away_id = away_team

    # Check cache
    cached = cache.get(home_id, away_id)
    if cached:
        margin, sigma = cached
        # Flip margin if teams are swapped vs cached version
        if cached == (margin, sigma):
            return margin, sigma

    payload = {
        "home_team_id": home_id,
        "away_team_id": away_id,
        "game_date": game_date,
        "neutral_site": True,       # All tournament games are neutral
        "home_team_name": home_name,
        "away_team_name": away_name,
    }

    try:
        resp = requests.post(
            f"{api_url}/predict/ncaa/full",
            json=payload,
            timeout=30
        )
        data = resp.json()

        if "error" in data:
            print(f"  API error for {home_name} vs {away_name}: {data['error']}")
            # Fallback: use seed difference as rough spread
            margin = (away_seed - home_seed) * 1.5
            sigma = 11.0
        else:
            margin = data.get("ml_margin", 0)
            sigma = data.get("sigma", 11.0)
            win_prob = data.get("ml_win_prob_home", 0.5)
            coverage = data.get("feature_coverage", "?")
            print(f"  {home_name} ({home_seed}) vs {away_name} ({away_seed}): "
                  f"spread={margin:+.1f}, win_prob={win_prob:.1%}, "
                  f"coverage={coverage}")

    except Exception as e:
        print(f"  Request failed for {home_name} vs {away_name}: {e}")
        margin = (away_seed - home_seed) * 1.5
        sigma = 11.0

    cache.set(home_id, away_id, (margin, sigma))
    return margin, sigma


def seed_based_spread(home_seed, away_seed):
    """Fallback spread estimate based on seed difference."""
    # Historical average margin by seed matchup
    # Rough approximation: each seed line ≈ 1.5 points
    return (away_seed - home_seed) * 1.4


# ═══════════════════════════════════════════════════════════════
# TOURNAMENT SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_game(margin, sigma):
    """
    Simulate a single game.
    margin: predicted spread (positive = home favored)
    sigma: standard deviation of prediction error
    Returns: positive if home wins, negative if away wins
    """
    return random.gauss(margin, sigma)


def simulate_region(region, api_url, cache, use_api=True):
    """
    Fetch predictions for all possible matchups in a region.
    Returns a dict of predictions keyed by (team_a_id, team_b_id).
    """
    predictions = {}

    # First round matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    first_round_pairs = [
        (region[0], region[1]),   # 1 vs 16
        (region[2], region[3]),   # 8 vs 9
        (region[4], region[5]),   # 5 vs 12
        (region[6], region[7]),   # 4 vs 13
        (region[8], region[9]),   # 6 vs 11
        (region[10], region[11]), # 3 vs 14
        (region[12], region[13]), # 7 vs 10
        (region[14], region[15]), # 2 vs 15
    ]

    print(f"\n  Fetching first-round predictions...")
    for home, away in first_round_pairs:
        if use_api:
            margin, sigma = fetch_prediction(api_url, home, away, cache)
        else:
            margin = seed_based_spread(home[0], away[0])
            sigma = 11.0
            print(f"  {home[1]} ({home[0]}) vs {away[1]} ({away[0]}): "
                  f"seed-based spread={margin:+.1f}")
        predictions[(home[2], away[2])] = (margin, sigma)
        time.sleep(0.1)  # Rate limiting

    return predictions


def run_tournament_sim(regions, api_url, n_sims=10000, use_api=True):
    """
    Run Monte Carlo simulation of the entire tournament.

    regions: dict with keys 'east', 'west', 'midwest', 'south'
    Returns: dict of team -> {round: probability}
    """
    cache = PredictionCache()
    all_predictions = {}

    # Phase 1: Fetch all first-round predictions via API
    print("\n" + "="*60)
    print("PHASE 1: Fetching model predictions for all matchups")
    print("="*60)

    for region_name, region_teams in regions.items():
        print(f"\n{'─'*40}")
        print(f"  {region_name.upper()} REGION")
        print(f"{'─'*40}")
        preds = simulate_region(region_teams, api_url, cache, use_api)
        all_predictions.update(preds)

    # Phase 2: For later-round matchups, we'll fetch on-the-fly during simulation
    # or use a heuristic based on the teams that advance

    print("\n" + "="*60)
    print(f"PHASE 2: Running {n_sims:,} Monte Carlo simulations")
    print("="*60)

    # Track results
    team_round_counts = defaultdict(lambda: defaultdict(int))
    champion_counts = defaultdict(int)
    final_four_counts = defaultdict(int)

    # All teams start in R64
    all_teams = {}
    for region_name, region_teams in regions.items():
        for team in region_teams:
            seed, name, team_id = team
            all_teams[team_id] = team
            team_round_counts[team_id]["R64"] = n_sims  # Everyone starts

    def get_prediction(team_a, team_b):
        """Get or compute prediction for any matchup."""
        a_id, b_id = team_a[2], team_b[2]
        key1 = (a_id, b_id)
        key2 = (b_id, a_id)

        if key1 in all_predictions:
            return all_predictions[key1]
        elif key2 in all_predictions:
            margin, sigma = all_predictions[key2]
            return (-margin, sigma)  # Flip perspective
        else:
            # For later rounds, use seed-based estimate
            # Could also call API here but would be slow
            margin = seed_based_spread(team_a[0], team_b[0])
            sigma = 11.0
            all_predictions[key1] = (margin, sigma)
            return (margin, sigma)

    def sim_bracket(region_teams):
        """Simulate a single region bracket, return list of advancing teams per round."""
        # R64 → R32
        r32 = []
        for i in range(0, 16, 2):
            team_a = region_teams[i]
            team_b = region_teams[i + 1]
            margin, sigma = get_prediction(team_a, team_b)
            result = simulate_game(margin, sigma)
            winner = team_a if result > 0 else team_b
            r32.append(winner)

        # R32 → S16
        s16 = []
        for i in range(0, 8, 2):
            team_a = r32[i]
            team_b = r32[i + 1]
            margin, sigma = get_prediction(team_a, team_b)
            result = simulate_game(margin, sigma)
            winner = team_a if result > 0 else team_b
            s16.append(winner)

        # S16 → E8
        e8 = []
        for i in range(0, 4, 2):
            team_a = s16[i]
            team_b = s16[i + 1]
            margin, sigma = get_prediction(team_a, team_b)
            result = simulate_game(margin, sigma)
            winner = team_a if result > 0 else team_b
            e8.append(winner)

        # E8 → F4 (regional final)
        team_a = e8[0]
        team_b = e8[1]
        margin, sigma = get_prediction(team_a, team_b)
        result = simulate_game(margin, sigma)
        f4_winner = team_a if result > 0 else team_b

        return r32, s16, e8, [f4_winner]

    # Run simulations
    for sim in range(n_sims):
        if sim % 1000 == 0 and sim > 0:
            print(f"  Completed {sim:,} / {n_sims:,} simulations...")

        # Simulate each region
        region_winners = {}
        for region_name, region_teams in regions.items():
            r32, s16, e8, f4 = sim_bracket(region_teams)

            for team in r32:
                team_round_counts[team[2]]["R32"] += 1
            for team in s16:
                team_round_counts[team[2]]["S16"] += 1
            for team in e8:
                team_round_counts[team[2]]["E8"] += 1
            for team in f4:
                team_round_counts[team[2]]["F4"] += 1
                final_four_counts[team[2]] += 1

            region_winners[region_name] = f4[0]

        # Final Four: East vs South, West vs Midwest
        semi1_a = region_winners["east"]
        semi1_b = region_winners["south"]
        margin1, sigma1 = get_prediction(semi1_a, semi1_b)
        result1 = simulate_game(margin1, sigma1)
        finalist1 = semi1_a if result1 > 0 else semi1_b
        team_round_counts[finalist1[2]]["NCG"] += 1

        semi2_a = region_winners["west"]
        semi2_b = region_winners["midwest"]
        margin2, sigma2 = get_prediction(semi2_a, semi2_b)
        result2 = simulate_game(margin2, sigma2)
        finalist2 = semi2_a if result2 > 0 else semi2_b
        team_round_counts[finalist2[2]]["NCG"] += 1

        # Championship game
        margin_final, sigma_final = get_prediction(finalist1, finalist2)
        result_final = simulate_game(margin_final, sigma_final)
        champion = finalist1 if result_final > 0 else finalist2
        team_round_counts[champion[2]]["CHAMP"] += 1
        champion_counts[champion[2]] += 1

    # Phase 3: Compile results
    print(f"\n  Completed all {n_sims:,} simulations!")

    results = []
    for team_id, team_info in all_teams.items():
        seed, name, tid = team_info
        region = None
        for rname, rteams in regions.items():
            if any(t[2] == team_id for t in rteams):
                region = rname
                break

        team_result = {
            "team_id": team_id,
            "team_name": name,
            "seed": seed,
            "region": region,
            "probabilities": {}
        }
        for round_name in ROUNDS:
            count = team_round_counts[team_id].get(round_name, 0)
            team_result["probabilities"][round_name] = round(count / n_sims * 100, 2)

        results.append(team_result)

    # Sort by championship probability descending
    results.sort(key=lambda x: x["probabilities"].get("CHAMP", 0), reverse=True)

    return results


# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════

def print_results(results, n_sims):
    """Pretty-print tournament simulation results."""
    print("\n" + "="*90)
    print(f"  2026 MARCH MADNESS SIMULATION RESULTS ({n_sims:,} simulations)")
    print("="*90)

    # Title contenders (>1% championship probability)
    print(f"\n{'─'*90}")
    print(f"  {'TEAM':<25} {'SEED':>4} {'REGION':<10} {'R32':>6} {'S16':>6} "
          f"{'E8':>6} {'F4':>6} {'NCG':>6} {'CHAMP':>6}")
    print(f"{'─'*90}")

    for r in results:
        if r["probabilities"].get("CHAMP", 0) >= 0.5:
            p = r["probabilities"]
            print(f"  {r['team_name']:<25} {r['seed']:>4} {r['region']:<10} "
                  f"{p.get('R32', 0):>5.1f}% {p.get('S16', 0):>5.1f}% "
                  f"{p.get('E8', 0):>5.1f}% {p.get('F4', 0):>5.1f}% "
                  f"{p.get('NCG', 0):>5.1f}% {p.get('CHAMP', 0):>5.1f}%")

    # Most likely champion
    champ = results[0]
    print(f"\n{'─'*90}")
    print(f"  PREDICTED CHAMPION: {champ['team_name']} ({champ['seed']} seed, "
          f"{champ['region'].upper()}) — {champ['probabilities']['CHAMP']:.1f}% probability")
    print(f"{'─'*90}")

    # Most likely Final Four
    f4_teams = sorted(results, key=lambda x: x["probabilities"].get("F4", 0), reverse=True)[:8]
    print(f"\n  MOST LIKELY FINAL FOUR:")
    for t in f4_teams[:4]:
        print(f"    {t['team_name']} ({t['seed']}, {t['region'].upper()}) — "
              f"{t['probabilities']['F4']:.1f}%")

    # Cinderella watch (seeds 10+ with >5% Sweet 16)
    cinderellas = [r for r in results
                   if r["seed"] >= 10 and r["probabilities"].get("S16", 0) > 5]
    if cinderellas:
        cinderellas.sort(key=lambda x: x["probabilities"]["S16"], reverse=True)
        print(f"\n  CINDERELLA WATCH (10+ seeds with >5% Sweet 16 chance):")
        for c in cinderellas:
            print(f"    {c['team_name']} ({c['seed']} seed) — "
                  f"S16: {c['probabilities']['S16']:.1f}%, "
                  f"E8: {c['probabilities'].get('E8', 0):.1f}%")

    # Upset alerts (first round upsets >30%)
    print(f"\n  FIRST-ROUND UPSET ALERTS (lower seed wins >30%):")
    for r in results:
        if r["seed"] >= 9:
            r32_prob = r["probabilities"].get("R32", 0)
            if r32_prob > 30:
                print(f"    {r['team_name']} ({r['seed']} seed) beats their opponent "
                      f"— {r32_prob:.1f}%")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="March Madness 2026 Tournament Simulator")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--api-url", type=str,
                        default="https://sports-predictor-api-production.up.railway.app",
                        help="Base URL for the prediction API")
    parser.add_argument("--output", type=str, default="bracket_results.json",
                        help="Output JSON file path")
    parser.add_argument("--seed-only", action="store_true",
                        help="Use seed-based spreads only (no API calls)")
    args = parser.parse_args()

    print("="*60)
    print("  2026 MARCH MADNESS TOURNAMENT SIMULATOR")
    print("  Powered by NCAA Prediction Model")
    print(f"  Simulations: {args.sims:,}")
    print(f"  API: {args.api_url}")
    print("="*60)

    regions = {
        "east": EAST_REGION,
        "west": WEST_REGION,
        "midwest": MIDWEST_REGION,
        "south": SOUTH_REGION,
    }

    results = run_tournament_sim(
        regions,
        args.api_url,
        n_sims=args.sims,
        use_api=not args.seed_only
    )

    # Print results
    print_results(results, args.sims)

    # Save to JSON
    output = {
        "meta": {
            "simulations": args.sims,
            "api_url": args.api_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "NCAA Stacked Ensemble (XGB+CatBoost+MLP)",
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    # Also save a simplified version for the React frontend
    frontend_data = {
        "meta": output["meta"],
        "teams": [
            {
                "name": r["team_name"],
                "seed": r["seed"],
                "region": r["region"],
                "r32": r["probabilities"].get("R32", 0),
                "s16": r["probabilities"].get("S16", 0),
                "e8": r["probabilities"].get("E8", 0),
                "f4": r["probabilities"].get("F4", 0),
                "ncg": r["probabilities"].get("NCG", 0),
                "champ": r["probabilities"].get("CHAMP", 0),
            }
            for r in results
        ]
    }
    frontend_path = args.output.replace(".json", "_frontend.json")
    with open(frontend_path, "w") as f:
        json.dump(frontend_data, f, indent=2)
    print(f"  Frontend data saved to {frontend_path}")


if __name__ == "__main__":
    main()
