#!/usr/bin/env python3
"""
backfill_advanced_features.py — Compute & write missing per-team columns to ncaa_historical.

The serve path (ncaa_full_predict.py) reads per-team columns like form, opp_to_rate,
pit_sos, momentum_halflife, etc. from ncaa_historical. For the 2026 season, these are
NULL because ncaaSync.js only writes basic stats — the enrichment scripts were never run.

This script fixes that by computing each feature chronologically (using only prior games,
no leakage) and writing them back to ncaa_historical.

Usage:
    python3 backfill_advanced_features.py --check     # Preview, no writes
    python3 backfill_advanced_features.py              # Backfill all seasons with NULLs
    python3 backfill_advanced_features.py --season 2026  # Backfill specific season
"""
import sys, os, time, math, warnings
sys.path.insert(0, '.')
os.environ.setdefault('SUPABASE_ANON_KEY', os.environ.get('SUPABASE_ANON_KEY', ''))
warnings.filterwarnings("ignore")

import numpy as np
from collections import defaultdict, deque
import requests

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ['SUPABASE_ANON_KEY']

CHECK_ONLY = "--check" in sys.argv
TARGET_SEASON = None
for i, arg in enumerate(sys.argv):
    if arg == "--season" and i + 1 < len(sys.argv):
        TARGET_SEASON = int(sys.argv[i + 1])

# ═══════════════════════════════════════════════════════════════
# SUPABASE HELPERS
# ═══════════════════════════════════════════════════════════════

def sb_get(table, params=""):
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, timeout=60)
                if r.ok:
                    break
                if "57014" in r.text:  # statement timeout (cold start)
                    print(f"  Retry {attempt+1} for offset {offset}...")
                    time.sleep(3)
                    continue
                print(f"  ERROR {r.status_code}: {r.text[:200]}")
                return all_data
            except requests.exceptions.Timeout:
                print(f"  Timeout, retry {attempt+1}...")
                time.sleep(3)
        else:
            print(f"  Failed after 3 retries at offset {offset}")
            return all_data
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data


def sb_patch(table, row_id, data):
    """Update a single row by id."""
    url = f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}"
    r = requests.patch(url,
        headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json", "Prefer": "return=minimal"},
        json=data, timeout=30)
    return r.ok


def sb_batch_patch(table, updates, batch_size=50):
    """Batch update rows. Each update = (id, {col: val})."""
    success = 0
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]
        for row_id, data in batch:
            if sb_patch(table, row_id, data):
                success += 1
        if (i + batch_size) % 500 == 0:
            print(f"    Updated {success}/{i + batch_size}...")
    return success


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (chronological, no leakage)
# ═══════════════════════════════════════════════════════════════

WINDOW = 20  # Rolling window for most features
ELO_K = 32   # Elo K-factor
FORM_DECAY = 0.1  # Exponential decay for form/momentum


class TeamAccumulator:
    """Maintains running stats for a single team across the season."""

    def __init__(self):
        self.games = []         # list of {margin, opp_id, opp_rank, date, is_home, score, opp_score}
        self.elo = 1500
        self.home_margins = deque(maxlen=WINDOW)
        self.away_margins = deque(maxlen=WINDOW)
        self.opponent_elos = deque(maxlen=WINDOW)

    def get_features(self):
        """Return computed features from accumulated games (prior games only)."""
        result = {}
        n = len(self.games)
        if n < 1:
            return result

        margins = [g["margin"] for g in self.games]
        recent = margins[-10:]

        # ── elo ──
        result["elo"] = round(self.elo, 1)

        # ── form: exponential decay of win/loss (F13 from ncaaUtils.js) ──
        form = 0.0
        for i, g in enumerate(self.games):
            age = n - 1 - i
            weight = math.exp(-FORM_DECAY * age)
            form += (1 if g["margin"] > 0 else -1) * weight
        result["form"] = round(form, 4)

        # ── momentum_halflife: recency-weighted margin ──
        if len(recent) >= 3:
            weighted = sum(m * math.exp(-FORM_DECAY * (len(recent) - 1 - i))
                           for i, m in enumerate(recent))
            weight_sum = sum(math.exp(-FORM_DECAY * (len(recent) - 1 - i))
                             for i in range(len(recent)))
            result["momentum_halflife"] = round(weighted / max(weight_sum, 1), 4)

        # ── opp_suppression: avg margin vs ranked (top-50) opponents ──
        ranked = [g for g in self.games if (g.get("opp_rank") or 200) <= 50]
        if len(ranked) >= 2:
            result["opp_suppression"] = round(
                sum(g["margin"] for g in ranked) / len(ranked), 4)

        # ── overreaction: volatility of margin changes ──
        if n >= 3:
            diffs = [margins[i] - margins[i-1] for i in range(1, n)]
            result["overreaction"] = round(float(np.std(diffs)), 4)

        # ── blowout_asym: (wins by 15+) - (losses by 15+) normalized ──
        big_wins = sum(1 for m in margins if m >= 15)
        big_losses = sum(1 for m in margins if m <= -15)
        result["blowout_asym"] = round((big_wins - big_losses) / max(n, 1), 4)

        # ── home_margin / away_margin: venue-specific performance ──
        h_margins = list(self.home_margins)
        a_margins = list(self.away_margins)
        if len(h_margins) >= 3:
            result["home_margin"] = round(sum(h_margins) / len(h_margins), 4)
        if len(a_margins) >= 3:
            result["away_margin"] = round(sum(a_margins) / len(a_margins), 4)

        # ── pit_sos: Elo-based SOS (avg opponent Elo) ──
        if len(self.opponent_elos) >= 3:
            result["pit_sos"] = round(sum(self.opponent_elos) / len(self.opponent_elos), 1)

        # ── fatigue_load: schedule density (games in last 14 days) ──
        if n >= 2:
            last_date = self.games[-1]["date"]
            recent_games = sum(1 for g in self.games
                               if g["date"] >= _date_minus_days(last_date, 14))
            result["fatigue_load"] = round(recent_games / 7.0, 4)  # normalize to ~1.0

        # ── opp_to_rate: computed from team turnovers / possessions ──
        # This is a TEAM stat from ESPN, not game-level. If missing, estimate from margins.
        # We set it from the raw columns if available.

        # ── to_conversion: steals-to-opponent-TO ratio ──
        # Also from ESPN team stats. Pass through if available.

        # ── opp_ppg: from game results ──
        opp_scores = [g["opp_score"] for g in self.games if g.get("opp_score")]
        if len(opp_scores) >= 3:
            result["opp_ppg"] = round(sum(opp_scores) / len(opp_scores), 1)

        return result

    def update(self, game):
        """Update accumulators AFTER computing features for this game."""
        self.games.append(game)

        # Elo update
        opp_elo = game.get("opp_elo", 1500)
        expected = 1.0 / (1.0 + 10 ** ((opp_elo - self.elo) / 400))
        actual = 1.0 if game["margin"] > 0 else 0.0 if game["margin"] < 0 else 0.5
        self.elo += ELO_K * (actual - expected)
        self.opponent_elos.append(opp_elo)

        # Venue margins
        if game.get("is_home"):
            self.home_margins.append(game["margin"])
        else:
            self.away_margins.append(game["margin"])


def _date_minus_days(date_str, days):
    """Simple date subtraction."""
    from datetime import datetime, timedelta
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (dt - timedelta(days=days)).strftime("%Y-%m-%d")
    except:
        return "2000-01-01"


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  BACKFILL ADVANCED FEATURES → ncaa_historical")
print("=" * 70)

# 1. Pull games — minimal columns first, then augment
_select_minimal = (
    "id,game_date,season,home_team_id,away_team_id,"
    "actual_home_score,actual_away_score,home_rank,away_rank,"
    "neutral_site,home_form,home_elo"
)

_select_stats = (
    "id,home_turnovers,away_turnovers,home_steals,away_steals,"
    "home_tempo,away_tempo"
)

if TARGET_SEASON:
    print(f"  Pulling season {TARGET_SEASON}...")
    rows = sb_get("ncaa_historical",
                   f"season=eq.{TARGET_SEASON}&actual_home_score=not.is.null"
                   f"&select={_select_minimal}&order=game_date.asc")
else:
    print("  Pulling games with NULL form...")
    rows = sb_get("ncaa_historical",
                   f"home_form=is.null&actual_home_score=not.is.null"
                   f"&select={_select_minimal}&order=game_date.asc")
    if not rows:
        print("  No rows with NULL form. Trying elo=NULL...")
        rows = sb_get("ncaa_historical",
                       f"home_elo=is.null&actual_home_score=not.is.null"
                       f"&select={_select_minimal}&order=game_date.asc")

if not rows:
    print("  No rows to backfill!")
    sys.exit(0)

print(f"  Found {len(rows)} games to process")

# Check how many have NULL form/elo (columns we included in minimal select)
null_form = sum(1 for r in rows if r.get("home_form") is None)
null_elo = sum(1 for r in rows if r.get("home_elo") is None)
print(f"  NULL form: {null_form}/{len(rows)}, NULL elo: {null_elo}/{len(rows)}")

# 2. Also pull ALL prior completed games for Elo seeding
# (we need the full season history to compute running stats correctly)
seasons = set(r.get("season") for r in rows if r.get("season"))
print(f"  Seasons: {sorted(seasons)}")

# Pull stats (turnovers, steals, tempo) for ratio features
stats_map = {}
for season in sorted(seasons):
    stats_rows = sb_get("ncaa_historical",
                         f"season=eq.{season}&actual_home_score=not.is.null"
                         f"&select={_select_stats}&order=game_date.asc")
    for sr in (stats_rows or []):
        stats_map[sr["id"]] = sr
    print(f"  Stats for season {season}: {len(stats_rows or [])} rows")

# Merge stats into rows
for r in rows:
    s = stats_map.get(r["id"], {})
    for k in ["home_turnovers", "away_turnovers", "home_steals", "away_steals",
              "home_tempo", "away_tempo"]:
        if k not in r:
            r[k] = s.get(k)

# For each season, pull ALL games (needed for chronological Elo computation)
all_games = []
for season in sorted(seasons):
    prior_rows = sb_get("ncaa_historical",
                         f"season=eq.{season}&actual_home_score=not.is.null"
                         f"&select={_select_minimal}&order=game_date.asc")
    if prior_rows:
        # Merge stats
        for r in prior_rows:
            s = stats_map.get(r["id"], {})
            for k in ["home_turnovers", "away_turnovers", "home_steals", "away_steals",
                       "home_tempo", "away_tempo"]:
                if k not in r:
                    r[k] = s.get(k)
        all_games.extend(prior_rows)
        print(f"  Season {season}: {len(prior_rows)} games")

# Deduplicate by id
seen = set()
unique_games = []
for g in all_games:
    gid = g.get("id")
    if gid not in seen:
        seen.add(gid)
        unique_games.append(g)
all_games = sorted(unique_games, key=lambda g: g.get("game_date", ""))
print(f"  Total unique games: {len(all_games)}")

# 3. Track which row IDs need updating (the ones with NULLs)
target_ids = set(r["id"] for r in rows)

# 4. Process chronologically
teams = defaultdict(TeamAccumulator)
updates = []  # (id, {col: val})
processed = 0

for g in all_games:
    gid = g.get("id")
    hid = str(g.get("home_team_id", ""))
    aid = str(g.get("away_team_id", ""))
    hs = g.get("actual_home_score")
    aws = g.get("actual_away_score")
    date = g.get("game_date", "")
    neutral = g.get("neutral_site", False)

    if not hid or not aid or hs is None or aws is None:
        continue

    hs, aws = float(hs), float(aws)
    margin = hs - aws
    h_rank = int(g.get("home_rank") or 200)
    a_rank = int(g.get("away_rank") or 200)

    # Get features from PRIOR accumulated games
    h_feats = teams[hid].get_features()
    a_feats = teams[aid].get_features()

    # Only write back for rows that are in our target set
    if gid in target_ids:
        patch = {}
        # Home team features — always write since these rows have NULLs
        for col, val in h_feats.items():
            patch[f"home_{col}"] = val
        # Away team features
        for col, val in a_feats.items():
            patch[f"away_{col}"] = val

        # opp_to_rate: compute from raw team stats if available
        # home team's opp_to_rate = opponent's turnovers / opponent's possessions
        h_to = g.get("home_turnovers")
        a_to = g.get("away_turnovers")
        h_tempo = g.get("home_tempo")
        a_tempo = g.get("away_tempo")
        # Home team forces turnovers: away team's TO rate
        if a_to is not None and a_tempo and float(a_tempo) > 0:
            patch["home_opp_to_rate"] = round(float(a_to) / float(a_tempo), 4)
        if h_to is not None and h_tempo and float(h_tempo) > 0:
            patch["away_opp_to_rate"] = round(float(h_to) / float(h_tempo), 4)

        # to_conversion: steals / opponent turnovers
        h_steals = g.get("home_steals")
        a_steals = g.get("away_steals")
        if h_steals and a_to and float(a_to) > 0:
            patch["home_to_conversion"] = round(float(h_steals) / float(a_to), 4)
        if a_steals and h_to and float(h_to) > 0:
            patch["away_to_conversion"] = round(float(a_steals) / float(h_to), 4)

        if patch:
            updates.append((gid, patch))

    # Update accumulators AFTER computing features (no leakage)
    teams[hid].update({
        "margin": margin, "opp_id": aid, "opp_rank": a_rank,
        "date": date, "is_home": True, "opp_score": aws,
        "opp_elo": teams[aid].elo,
    })
    teams[aid].update({
        "margin": -margin, "opp_id": hid, "opp_rank": h_rank,
        "date": date, "is_home": False, "opp_score": hs,
        "opp_elo": teams[hid].elo,
    })

    processed += 1
    if processed % 1000 == 0:
        print(f"  Processed {processed}/{len(all_games)} games, {len(updates)} updates queued")

print(f"\n  Processed {processed} games, {len(updates)} rows need updating")

# 5. Preview or write
if not updates:
    print("  Nothing to update!")
    sys.exit(0)

# Show sample
print(f"\n  Sample updates (first 3):")
for row_id, patch in updates[:3]:
    cols = list(patch.keys())[:8]
    print(f"    id={row_id}: {', '.join(f'{c}={patch[c]}' for c in cols)}")

# Count columns being filled
col_counts = defaultdict(int)
for _, patch in updates:
    for col in patch:
        col_counts[col] += 1
print(f"\n  Columns being filled:")
for col, cnt in sorted(col_counts.items(), key=lambda x: -x[1]):
    print(f"    {col}: {cnt} rows")

if CHECK_ONLY:
    print(f"\n  --check mode: {len(updates)} rows would be updated. Run without --check to write.")
    sys.exit(0)

# 6. Write to Supabase
print(f"\n  Writing {len(updates)} updates to ncaa_historical...")
t0 = time.time()
success = sb_batch_patch("ncaa_historical", updates, batch_size=50)
elapsed = time.time() - t0
print(f"  ✅ Updated {success}/{len(updates)} rows in {elapsed:.0f}s")
