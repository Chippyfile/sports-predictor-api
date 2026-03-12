#!/usr/bin/env python3
"""
ncaa_rolling_tendencies.py
══════════════════════════
Compute per-team rolling averages from ESPN extraction columns
(PBP, player, clutch, garbage time) and write them back to
ncaa_historical as roll_* columns.

These are SAFE for ML training because they use only data from
PRIOR games — never the current game. The rolling window looks
back N games for each team at the point in time of each game.

Rolling features computed (per team, per game):
  PBP:     roll_largest_run, roll_drought_rate, roll_lead_changes,
           roll_time_with_lead_pct
  Player:  roll_star1_share, roll_top3_share, roll_bench_share,
           roll_minutes_hhi, roll_players_used
  Clutch:  roll_clutch_ft_pct
  Garbage: roll_garbage_pct

Usage:
    python3 ncaa_rolling_tendencies.py [--window 8] [--dry-run]

Prerequisites:
    - ESPN extraction complete (ncaa_espn_extract_all.py Phase 2 pushed)
    - venue/refs extraction complete (ncaa_reextract_venue_refs.py)
    - SUPABASE_ANON_KEY set in environment
"""
import os, sys, json, time, argparse
import numpy as np
import requests
from collections import defaultdict

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


def sb_get_all(table, params=""):
    """Fetch all rows with pagination (1000 per request)."""
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
            print(f"  sb_get error: {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
        if offset % 10000 == 0:
            print(f"    fetched {offset} rows...")
    return all_data


def sb_patch(table, match_col, match_val, patch_data, session=None, retries=3):
    """Update a single row in Supabase with retry logic."""
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    _s = session or requests
    for attempt in range(retries):
        try:
            r = _s.patch(url, headers=HEADERS, json=patch_data, timeout=20)
            return r.ok
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
            else:
                return False


def safe_float(val, default=None):
    """Convert to float safely, return default if not possible."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def compute_rolling_tendencies(games, window=8):
    """
    Compute rolling team tendency features for all games.

    For each game, look at each team's prior N games (sorted by date)
    and compute rolling averages of PBP/player/clutch/garbage features.

    Returns: dict of game_id -> {roll columns for home side, roll columns for away side}
    """
    # ── Step 1: Build per-team game history ──
    # Each entry: (game_date, game_id, side, raw_features)
    team_history = defaultdict(list)

    for g in games:
        gid = g.get("game_id")
        gdate = g.get("game_date", "")
        if not gid or not gdate:
            continue

        home_id = g.get("home_team_id") or g.get("home_team_abbr", "")
        away_id = g.get("away_team_id") or g.get("away_team_abbr", "")

        if not home_id or not away_id:
            continue

        # Extract raw per-game features for home team's performance IN THIS GAME
        # These are the team's own stats from this specific game
        home_feats = {
            "largest_run": safe_float(g.get("home_largest_run")),
            "drought_count": safe_float(g.get("home_drought_count")),
            "lead_changes": safe_float(g.get("lead_changes")),
            "time_with_lead_pct": safe_float(g.get("home_time_with_lead_pct")),
            "star1_pts_share": safe_float(g.get("home_star1_pts_share")),
            "top3_pts_share": safe_float(g.get("home_top3_pts_share")),
            "bench_pts_share": safe_float(g.get("home_bench_pts_share")),
            "minutes_hhi": safe_float(g.get("home_minutes_hhi")),
            "players_used": safe_float(g.get("home_players_used")),
            "clutch_ftm": safe_float(g.get("home_clutch_ftm")),
            "clutch_fta": safe_float(g.get("home_clutch_fta")),
            "is_garbage_time_game": safe_float(g.get("is_garbage_time_game")),
        }

        away_feats = {
            "largest_run": safe_float(g.get("away_largest_run")),
            "drought_count": safe_float(g.get("away_drought_count")),
            "lead_changes": safe_float(g.get("lead_changes")),
            "time_with_lead_pct": safe_float(g.get("home_time_with_lead_pct"), 0.5),
            "star1_pts_share": safe_float(g.get("away_star1_pts_share")),
            "top3_pts_share": safe_float(g.get("away_top3_pts_share")),
            "bench_pts_share": safe_float(g.get("away_bench_pts_share")),
            "minutes_hhi": safe_float(g.get("away_minutes_hhi")),
            "players_used": safe_float(g.get("away_players_used")),
            "clutch_ftm": safe_float(g.get("away_clutch_ftm")),
            "clutch_fta": safe_float(g.get("away_clutch_fta")),
            "is_garbage_time_game": safe_float(g.get("is_garbage_time_game")),
        }

        # ── ATS (Against The Spread) result per team ──
        # espn_spread uses Vegas convention: negative = home favored
        # actual_margin + espn_spread > 0 → home covered, < 0 → away covered
        _espn_sp = safe_float(g.get("espn_spread"))
        _h_score = safe_float(g.get("actual_home_score"))
        _a_score = safe_float(g.get("actual_away_score"))
        if _espn_sp is not None and _h_score is not None and _a_score is not None:
            actual_margin = _h_score - _a_score
            ats_margin = actual_margin + _espn_sp  # >0 = home covers, <0 = away covers
            if abs(ats_margin) < 0.5:  # push
                home_feats["ats_covered"] = None
                away_feats["ats_covered"] = None
            else:
                home_feats["ats_covered"] = 1 if ats_margin > 0 else 0
                away_feats["ats_covered"] = 1 if ats_margin < 0 else 0
            home_feats["ats_margin"] = ats_margin
            away_feats["ats_margin"] = -ats_margin
        else:
            home_feats["ats_covered"] = None
            away_feats["ats_covered"] = None
            home_feats["ats_margin"] = None
            away_feats["ats_margin"] = None

        # For away team's time_with_lead, invert (1 - home_time_with_lead_pct)
        h_twl = safe_float(g.get("home_time_with_lead_pct"), 0.5)
        away_feats["time_with_lead_pct"] = 1.0 - h_twl if h_twl is not None else None

        team_history[str(home_id)].append((gdate, gid, "home", home_feats))
        team_history[str(away_id)].append((gdate, gid, "away", away_feats))

    # ── Step 2: Sort each team's history by date ──
    for tid in team_history:
        team_history[tid].sort(key=lambda x: x[0])

    # ── Step 3: Compute rolling averages ──
    # For each game, look back at the team's prior N games
    def _rolling_avg(values, default):
        """Average of non-None values, or default."""
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else default

    # Build game_id -> {home_roll_*, away_roll_*}
    result = {}

    for tid, history in team_history.items():
        for idx, (gdate, gid, side, feats) in enumerate(history):
            # Look back at prior games (not including current)
            start = max(0, idx - window)
            prior = history[start:idx]

            if len(prior) < 2:
                # Not enough history — use defaults (will be filled by raw_cols defaults)
                continue

            prior_feats = [p[3] for p in prior]

            # Compute rolling averages
            roll = {}
            roll[f"{side}_roll_largest_run"] = round(
                _rolling_avg([f["largest_run"] for f in prior_feats], 8.0), 2)
            roll[f"{side}_roll_drought_rate"] = round(
                _rolling_avg([f["drought_count"] for f in prior_feats], 1.5), 2)
            roll[f"{side}_roll_lead_changes"] = round(
                _rolling_avg([f["lead_changes"] for f in prior_feats], 8.0), 2)
            roll[f"{side}_roll_time_with_lead_pct"] = round(
                _rolling_avg([f["time_with_lead_pct"] for f in prior_feats], 0.5), 3)
            roll[f"{side}_roll_star1_share"] = round(
                _rolling_avg([f["star1_pts_share"] for f in prior_feats], 0.25), 3)
            roll[f"{side}_roll_top3_share"] = round(
                _rolling_avg([f["top3_pts_share"] for f in prior_feats], 0.55), 3)
            roll[f"{side}_roll_bench_share"] = round(
                _rolling_avg([f["bench_pts_share"] for f in prior_feats], 0.20), 3)
            roll[f"{side}_roll_minutes_hhi"] = round(
                _rolling_avg([f["minutes_hhi"] for f in prior_feats], 0.20), 3)
            roll[f"{side}_roll_players_used"] = round(
                _rolling_avg([f["players_used"] for f in prior_feats], 8.0), 1)

            # Clutch FT%: sum FTM / sum FTA across window
            clutch_ftm = sum(f["clutch_ftm"] for f in prior_feats if f["clutch_ftm"] is not None)
            clutch_fta = sum(f["clutch_fta"] for f in prior_feats if f["clutch_fta"] is not None)
            roll[f"{side}_roll_clutch_ft_pct"] = round(
                clutch_ftm / max(clutch_fta, 1), 3)

            # Garbage time %: fraction of prior games that were garbage time
            garbage_games = [f["is_garbage_time_game"] for f in prior_feats
                            if f["is_garbage_time_game"] is not None]
            roll[f"{side}_roll_garbage_pct"] = round(
                sum(garbage_games) / max(len(garbage_games), 1), 3)

            # ── ATS (Against The Spread) rolling record ──
            # Win rate: fraction of prior games with spread data where team covered
            ats_results = [f["ats_covered"] for f in prior_feats if f.get("ats_covered") is not None]
            if len(ats_results) >= 3:
                roll[f"{side}_roll_ats_pct"] = round(sum(ats_results) / len(ats_results), 3)
                roll[f"{side}_roll_ats_n"] = len(ats_results)
            else:
                roll[f"{side}_roll_ats_pct"] = 0.5  # neutral default
                roll[f"{side}_roll_ats_n"] = 0

            # Average ATS margin: how much the team beats/misses the spread by
            ats_margins = [f["ats_margin"] for f in prior_feats if f.get("ats_margin") is not None]
            if len(ats_margins) >= 3:
                roll[f"{side}_roll_ats_margin"] = round(sum(ats_margins) / len(ats_margins), 2)
            else:
                roll[f"{side}_roll_ats_margin"] = 0.0

            # Merge into result
            if gid not in result:
                result[gid] = {}
            result[gid].update(roll)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute rolling team tendencies")
    parser.add_argument("--window", type=int, default=8, help="Rolling window size (games)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't push to Supabase")
    parser.add_argument("--season", type=int, default=None, help="Process only this season")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  NCAA ROLLING TENDENCIES (window={args.window})")
    print("=" * 70)

    # ── Fetch all games with ESPN extraction data ──
    select_cols = (
        "game_id,game_date,season,"
        "home_team_id,away_team_id,home_team_abbr,away_team_abbr,"
        "home_largest_run,away_largest_run,"
        "home_drought_count,away_drought_count,"
        "lead_changes,home_time_with_lead_pct,"
        "home_star1_pts_share,away_star1_pts_share,"
        "home_top3_pts_share,away_top3_pts_share,"
        "home_bench_pts_share,away_bench_pts_share,"
        "home_minutes_hhi,away_minutes_hhi,"
        "home_players_used,away_players_used,"
        "home_clutch_ftm,home_clutch_fta,"
        "away_clutch_ftm,away_clutch_fta,"
        "is_garbage_time_game,"
        "espn_spread,actual_home_score,actual_away_score"
    )

    params = f"select={select_cols}&order=game_date.asc"
    if args.season:
        params += f"&season=eq.{args.season}"

    print(f"  Fetching games from Supabase...")
    games = sb_get_all("ncaa_historical", params)
    print(f"  Fetched {len(games)} games")

    if not games:
        print("  No games found. Exiting.")
        return

    # ── Compute rolling tendencies ──
    print(f"  Computing rolling tendencies (window={args.window})...")
    t0 = time.time()
    tendencies = compute_rolling_tendencies(games, window=args.window)
    elapsed = time.time() - t0
    print(f"  Computed tendencies for {len(tendencies)} games in {elapsed:.1f}s")

    # ── Stats on coverage ──
    has_home = sum(1 for v in tendencies.values() if "home_roll_star1_share" in v)
    has_away = sum(1 for v in tendencies.values() if "away_roll_star1_share" in v)
    print(f"  Coverage: {has_home} games with home rolling, {has_away} with away rolling")

    if args.dry_run:
        print("  DRY RUN — not pushing to Supabase")
        # Print a sample
        sample_ids = list(tendencies.keys())[:3]
        for gid in sample_ids:
            print(f"\n  {gid}:")
            for k, v in sorted(tendencies[gid].items()):
                print(f"    {k}: {v}")
        return

    # ── Push to Supabase ──
    # Valid column names for Supabase (must match SQL migration)
    valid_cols = {
        "home_roll_largest_run", "away_roll_largest_run",
        "home_roll_drought_rate", "away_roll_drought_rate",
        "home_roll_lead_changes", "away_roll_lead_changes",
        "home_roll_time_with_lead_pct", "away_roll_time_with_lead_pct",
        "home_roll_star1_share", "away_roll_star1_share",
        "home_roll_top3_share", "away_roll_top3_share",
        "home_roll_bench_share", "away_roll_bench_share",
        "home_roll_minutes_hhi", "away_roll_minutes_hhi",
        "home_roll_players_used", "away_roll_players_used",
        "home_roll_clutch_ft_pct", "away_roll_clutch_ft_pct",
        "home_roll_garbage_pct", "away_roll_garbage_pct",
        "home_roll_ats_pct", "away_roll_ats_pct",
        "home_roll_ats_n", "away_roll_ats_n",
        "home_roll_ats_margin", "away_roll_ats_margin",
    }

    print(f"\n  Pushing to Supabase ({len(tendencies)} games)...")
    success = 0
    errors = 0
    skipped = 0
    t0 = time.time()

    # Use persistent session for connection pooling (much faster for 64K requests)
    session = requests.Session()
    session.headers.update({
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    })

    for i, (gid, roll_data) in enumerate(tendencies.items()):
        patch = {k: v for k, v in roll_data.items() if k in valid_cols and v is not None}
        if not patch:
            skipped += 1
            continue

        if sb_patch("ncaa_historical", "game_id", gid, patch, session=session):
            success += 1
        else:
            errors += 1

        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1}/{len(tendencies)} | success:{success} err:{errors} | {rate:.0f}/s")

    session.close()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ROLLING TENDENCIES COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"  Updated: {success} | Errors: {errors} | Skipped: {skipped}")
    print(f"  Window: {args.window} games")
    print(f"\n  NEXT: Run SQL migration to add roll_* columns, then retrain model")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
