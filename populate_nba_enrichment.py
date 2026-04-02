#!/usr/bin/env python3
"""
populate_nba_enrichment.py — Populate nba_team_enrichment for all 30 teams.

Step 1: Verify table exists (prints SQL if not)
Step 2: Check nba_game_stats has data
Step 3: Run recompute_all_enrichment()
Step 4: Verify results

Usage:
    python3 populate_nba_enrichment.py          # Check + populate
    python3 populate_nba_enrichment.py --check  # Check only, don't write
"""
import sys, os, json
sys.path.insert(0, ".")

CHECK_ONLY = "--check" in sys.argv

import requests
from config import SUPABASE_URL, SUPABASE_KEY

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

print("=" * 60)
print("  NBA Team Enrichment — Populate All 30 Teams")
print("=" * 60)

# ── Step 1: Check table exists ──
print("\n1. Checking nba_team_enrichment table...")
resp = requests.get(
    f"{SUPABASE_URL}/rest/v1/nba_team_enrichment?select=team_abbr&limit=1",
    headers=headers, timeout=10
)
if resp.status_code == 200:
    existing = resp.json()
    print(f"   Table exists. Current rows: ", end="")
    count_resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/nba_team_enrichment?select=team_abbr",
        headers=headers, timeout=10
    )
    teams = [r["team_abbr"] for r in (count_resp.json() or [])]
    print(f"{len(teams)} teams: {', '.join(sorted(teams)) if teams else '(empty)'}")
else:
    print(f"   Table missing or error: {resp.status_code}")
    print(f"\n   Run this SQL in Supabase SQL Editor:\n")
    print("""
CREATE TABLE IF NOT EXISTS nba_team_enrichment (
    team_abbr TEXT PRIMARY KEY,
    updated_date TEXT,
    games_counted INT DEFAULT 0,
    scoring_var FLOAT DEFAULT 0,
    consistency FLOAT DEFAULT 0,
    ceiling FLOAT DEFAULT 0,
    floor FLOAT DEFAULT 0,
    bimodal FLOAT DEFAULT 0,
    score_kurtosis FLOAT DEFAULT 0,
    scoring_entropy FLOAT DEFAULT 0,
    def_stability FLOAT DEFAULT 0,
    opp_suppression FLOAT DEFAULT 0,
    three_value FLOAT DEFAULT 0,
    ts_regression FLOAT DEFAULT 0,
    three_pt_regression FLOAT DEFAULT 0,
    pace_leverage FLOAT DEFAULT 0,
    pace_control FLOAT DEFAULT 0,
    margin_accel FLOAT DEFAULT 0,
    momentum_halflife FLOAT DEFAULT 0,
    win_aging FLOAT DEFAULT 0,
    pyth_residual FLOAT DEFAULT 0,
    pyth_luck FLOAT DEFAULT 0,
    recovery_idx FLOAT DEFAULT 0,
    ft_trip_rate FLOAT DEFAULT 0,
    avg_ft_trip_rate FLOAT DEFAULT 0,
    three_fg_rate FLOAT DEFAULT 0,
    avg_three_fg_rate FLOAT DEFAULT 0,
    oreb FLOAT DEFAULT 0,
    avg_oreb FLOAT DEFAULT 0,
    avg_margin FLOAT DEFAULT 0,
    margin_trend FLOAT DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT now()
);
""")
    print("   Then re-run this script.")
    sys.exit(1)

# ── Step 2: Check nba_game_stats has data ──
print("\n2. Checking nba_game_stats data...")
stats_resp = requests.get(
    f"{SUPABASE_URL}/rest/v1/nba_game_stats?select=team_abbr&limit=500",
    headers=headers, timeout=10
)
if stats_resp.ok:
    stats_rows = stats_resp.json() or []
    from collections import Counter
    team_counts = Counter(r["team_abbr"] for r in stats_rows)
    n_teams_with_data = sum(1 for c in team_counts.values() if c >= 5)
    print(f"   {len(stats_rows)} rows, {len(team_counts)} teams, {n_teams_with_data} with 5+ games")
    if n_teams_with_data < 10:
        low_teams = [t for t, c in team_counts.items() if c < 5]
        print(f"   ⚠ Low data teams: {low_teams[:10]}")
    # Show top/bottom
    top5 = team_counts.most_common(5)
    bot5 = team_counts.most_common()[-5:]
    print(f"   Top: {', '.join(f'{t}({c})' for t,c in top5)}")
    print(f"   Low: {', '.join(f'{t}({c})' for t,c in bot5)}")
else:
    print(f"   ⚠ nba_game_stats not accessible: {stats_resp.status_code}")
    print(f"   Enrichment needs game-level stats. Run nba_game_stats backfill first.")
    sys.exit(1)

if CHECK_ONLY:
    print("\n--check mode: stopping here.")
    sys.exit(0)

# ── Step 3: Compute + save enrichment for all 30 teams ──
print("\n3. Computing enrichment for all 30 teams...")
from nba_enrichment import recompute_all_enrichment
n_success = recompute_all_enrichment()

# ── Step 4: Verify ──
print(f"\n4. Verification...")
verify = requests.get(
    f"{SUPABASE_URL}/rest/v1/nba_team_enrichment?select=team_abbr,ceiling,floor,scoring_entropy,momentum_halflife,games_counted&order=team_abbr",
    headers=headers, timeout=10
)
if verify.ok:
    rows = verify.json() or []
    print(f"   {len(rows)} teams in nba_team_enrichment:")
    for r in rows:
        print(f"   {r['team_abbr']:4s}  games={r.get('games_counted',0):2d}  "
              f"ceil={r.get('ceiling',0):+6.1f}  floor={r.get('floor',0):+6.1f}  "
              f"entropy={r.get('scoring_entropy',0):.3f}  momentum={r.get('momentum_halflife',0):+.2f}")

print(f"\nDone: {n_success}/30 teams enriched")
if n_success >= 25:
    print("✓ Ready — next prediction should show 50+/55 coverage")
elif n_success >= 15:
    print("⚠ Partial — some teams missing game stats data")
else:
    print("✗ Low coverage — check nba_game_stats table")
