#!/usr/bin/env python3
"""
ncaa_historical_backfill_stats.py
═════════════════════════════════
Backfills the 52 missing columns in ncaa_historical.

Two-phase approach:
  Phase 1: Fetch ESPN team stats for each season, update raw stat columns
           (fgpct, threepct, assists, turnovers, steals, blocks, etc.)
  Phase 2: Run the heuristic backfill to compute derived columns
           (spread_home, win_pct_home, pred_home_score, etc.)

Run from sports-predictor-api root:
    python3 ncaa_historical_backfill_stats.py

Prerequisites:
    1. Run ncaa_historical_migration.sql in Supabase SQL Editor first
    2. Ensure SUPABASE_ANON_KEY is set in environment
"""
import os, sys, json, time, requests
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

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"


def sb_get(table, params=""):
    """Fetch from Supabase with pagination."""
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
    return all_data


def sb_patch(table, match_col, match_val, patch_data):
    """Update a single row in Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/{table}?{match_col}=eq.{match_val}"
    r = requests.patch(url, headers=HEADERS, json=patch_data, timeout=15)
    return r.ok


def sb_patch_batch(table, match_col, updates):
    """Batch update rows. updates = list of (match_val, patch_data)."""
    success = 0
    for match_val, patch_data in updates:
        if sb_patch(table, match_col, match_val, patch_data):
            success += 1
    return success


def fetch_espn_team_stats(team_id, season):
    """Fetch team stats from ESPN for a given season."""
    url = f"{ESPN_BASE}/teams/{team_id}/statistics?season={season}"
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None


def parse_espn_stats(stats_json):
    """Parse ESPN team stats JSON into a flat dict."""
    if not stats_json:
        return None

    result = {}
    try:
        # Navigate to the stats categories
        splits = stats_json.get("results", stats_json)
        if isinstance(splits, dict):
            cats = splits.get("splits", {}).get("categories", [])
        elif isinstance(splits, list):
            cats = []
            for item in splits:
                cats.extend(item.get("splits", {}).get("categories", []))
        else:
            cats = []

        stat_map = {}
        for cat in cats:
            for stat in cat.get("stats", []):
                name = stat.get("name", "").lower().replace(" ", "_")
                val = stat.get("value", stat.get("displayValue", 0))
                try:
                    stat_map[name] = float(val)
                except (ValueError, TypeError):
                    stat_map[name] = 0

        # Map to our column names
        result["fgpct"] = stat_map.get("field_goal_pct", stat_map.get("fgpct", 0)) / 100 \
            if stat_map.get("field_goal_pct", stat_map.get("fgpct", 0)) > 1 else \
            stat_map.get("field_goal_pct", stat_map.get("fgpct", 0))
        result["threepct"] = stat_map.get("three_point_field_goal_pct", stat_map.get("3pt_pct", 0))
        if result["threepct"] > 1:
            result["threepct"] /= 100
        result["ftpct"] = stat_map.get("free_throw_pct", stat_map.get("ftpct", 0))
        if result["ftpct"] > 1:
            result["ftpct"] /= 100
        result["assists"] = stat_map.get("assists_per_game", stat_map.get("assists", 14))
        result["turnovers"] = stat_map.get("turnovers_per_game", stat_map.get("turnovers", 12))
        result["steals"] = stat_map.get("steals_per_game", stat_map.get("steals", 7))
        result["blocks"] = stat_map.get("blocks_per_game", stat_map.get("blocked_shots", 3.5))

        # Derived
        fga = stat_map.get("field_goals_attempted_per_game",
                           stat_map.get("fga", 60))
        fta = stat_map.get("free_throws_attempted_per_game",
                           stat_map.get("fta", 20))
        off_reb = stat_map.get("offensive_rebounds_per_game",
                               stat_map.get("offensive_rebounds", 10))
        total_reb = stat_map.get("rebounds_per_game",
                                 stat_map.get("total_rebounds", 35))

        result["orb_pct"] = off_reb / max(total_reb, 1)
        result["fta_rate"] = fta / max(fga, 1)
        result["ato_ratio"] = result["assists"] / max(result["turnovers"], 1)

    except Exception as e:
        print(f"    Parse error: {e}")
        return None

    return result


def compute_form_and_sos(team_id, season, game_date, all_games):
    """
    Compute form score and SOS from games prior to game_date.
    Uses simple last-10 game form and opponent win% as SOS proxy.
    """
    team_games = []
    for g in all_games:
        if g.get("season") != season:
            continue
        if g.get("game_date", "") >= game_date:
            continue
        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id:
            is_home = (g.get("home_team_id") == team_id)
            margin = (g["actual_home_score"] - g["actual_away_score"]) * (1 if is_home else -1)
            team_games.append(margin)

    if not team_games:
        return 0.0, 0.5

    # Form: weighted average of last 10 margins, normalized
    last10 = team_games[-10:]
    weights = np.linspace(0.5, 1.0, len(last10))
    form = float(np.average(last10, weights=weights))
    form = max(-15, min(15, form))  # clamp
    form_score = form / 15.0  # normalize to [-1, 1]

    # SOS proxy: win rate
    wins = sum(1 for m in team_games if m > 0)
    sos = wins / len(team_games) if team_games else 0.5

    return round(form_score, 4), round(sos, 4)


def compute_rest_days(team_id, game_date, all_games):
    """Compute rest days since last game for a team."""
    from datetime import datetime
    gd = datetime.strptime(game_date, "%Y-%m-%d") if isinstance(game_date, str) else game_date
    prev_dates = []
    for g in all_games:
        d = g.get("game_date", "")
        if not d or d >= game_date:
            continue
        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id:
            prev_dates.append(d)
    if not prev_dates:
        return 5  # default
    last = max(prev_dates)
    last_dt = datetime.strptime(last, "%Y-%m-%d") if isinstance(last, str) else last
    return max(0, (gd - last_dt).days)


# ═══════════════════════════════════════════════════════════════
# MAIN BACKFILL
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  NCAA HISTORICAL BACKFILL — Phase 1: Stats + Context")
    print("=" * 70)

    # ── Load all historical games ──
    print("\n  Loading ncaa_historical...")
    all_games = sb_get("ncaa_historical",
                       "actual_home_score=not.is.null&select=*&order=game_date.asc")
    print(f"  Loaded {len(all_games)} games")

    if not all_games:
        print("  No games found!")
        return

    # ── Get unique team IDs and seasons ──
    team_ids = set()
    seasons = set()
    for g in all_games:
        if g.get("home_team_id"):
            team_ids.add(int(g["home_team_id"]))
        if g.get("away_team_id"):
            team_ids.add(int(g["away_team_id"]))
        if g.get("season"):
            seasons.add(int(g["season"]))

    print(f"  Unique teams: {len(team_ids)}")
    print(f"  Seasons: {sorted(seasons)}")

    # ── Phase 1: Fetch ESPN stats per team per season ──
    print(f"\n  Phase 1: Fetching ESPN team stats...")
    stats_cache = {}  # (team_id, season) -> stats dict
    total_teams = len(team_ids) * len(seasons)
    fetched = 0

    for season in sorted(seasons):
        print(f"\n  Season {season}:")
        season_fetched = 0
        for tid in sorted(team_ids):
            stats_json = fetch_espn_team_stats(tid, season)
            parsed = parse_espn_stats(stats_json)
            if parsed:
                stats_cache[(tid, season)] = parsed
                season_fetched += 1
            fetched += 1
            if fetched % 50 == 0:
                print(f"    Fetched {fetched}/{total_teams} team-seasons...")
            time.sleep(0.15)  # Rate limit
        print(f"    Got stats for {season_fetched}/{len(team_ids)} teams")

    print(f"\n  Total team-season stats cached: {len(stats_cache)}")

    # ── Phase 2: Build updates for each game ──
    print(f"\n  Phase 2: Computing updates for {len(all_games)} games...")
    updates = []
    skipped = 0

    for i, g in enumerate(all_games):
        game_id = g.get("game_id") or g.get("id")
        if not game_id:
            skipped += 1
            continue

        season = int(g.get("season", 0))
        h_tid = int(g.get("home_team_id", 0))
        a_tid = int(g.get("away_team_id", 0))
        game_date = g.get("game_date", "")

        h_stats = stats_cache.get((h_tid, season), {})
        a_stats = stats_cache.get((a_tid, season), {})

        # Compute form and SOS
        h_form, h_sos = compute_form_and_sos(h_tid, season, game_date, all_games)
        a_form, a_sos = compute_form_and_sos(a_tid, season, game_date, all_games)

        # Compute rest days
        h_rest = compute_rest_days(h_tid, game_date, all_games)
        a_rest = compute_rest_days(a_tid, game_date, all_games)

        # Compute W/L from prior games
        h_wins = sum(1 for gg in all_games
                     if gg.get("season") == season and gg.get("game_date", "") < game_date
                     and ((gg.get("home_team_id") == h_tid and (gg.get("actual_home_score", 0) or 0) > (gg.get("actual_away_score", 0) or 0))
                          or (gg.get("away_team_id") == h_tid and (gg.get("actual_away_score", 0) or 0) > (gg.get("actual_home_score", 0) or 0))))
        h_total = sum(1 for gg in all_games
                      if gg.get("season") == season and gg.get("game_date", "") < game_date
                      and (gg.get("home_team_id") == h_tid or gg.get("away_team_id") == h_tid))
        h_losses = h_total - h_wins

        a_wins = sum(1 for gg in all_games
                     if gg.get("season") == season and gg.get("game_date", "") < game_date
                     and ((gg.get("home_team_id") == a_tid and (gg.get("actual_home_score", 0) or 0) > (gg.get("actual_away_score", 0) or 0))
                          or (gg.get("away_team_id") == a_tid and (gg.get("actual_away_score", 0) or 0) > (gg.get("actual_home_score", 0) or 0))))
        a_total = sum(1 for gg in all_games
                      if gg.get("season") == season and gg.get("game_date", "") < game_date
                      and (gg.get("home_team_id") == a_tid or gg.get("away_team_id") == a_tid))
        a_losses = a_total - a_wins

        # Determine tournament context from date/game properties
        is_postseason = g.get("is_postseason", False)
        month = int(game_date.split("-")[1]) if game_date else 1
        is_early = month in (11, 12) and not is_postseason
        is_march = month == 3

        patch = {
            # Shooting
            "home_fgpct": h_stats.get("fgpct"),
            "away_fgpct": a_stats.get("fgpct"),
            "home_threepct": h_stats.get("threepct"),
            "away_threepct": a_stats.get("threepct"),
            "home_ftpct": h_stats.get("ftpct"),
            "away_ftpct": a_stats.get("ftpct"),
            # Ball handling
            "home_assists": h_stats.get("assists"),
            "away_assists": a_stats.get("assists"),
            "home_turnovers": h_stats.get("turnovers"),
            "away_turnovers": a_stats.get("turnovers"),
            # Four Factors
            "home_orb_pct": h_stats.get("orb_pct"),
            "away_orb_pct": a_stats.get("orb_pct"),
            "home_fta_rate": h_stats.get("fta_rate"),
            "away_fta_rate": a_stats.get("fta_rate"),
            "home_ato_ratio": h_stats.get("ato_ratio"),
            "away_ato_ratio": a_stats.get("ato_ratio"),
            # Defense
            "home_steals": h_stats.get("steals"),
            "away_steals": a_stats.get("steals"),
            "home_blocks": h_stats.get("blocks"),
            "away_blocks": a_stats.get("blocks"),
            # Context
            "home_form": h_form,
            "away_form": a_form,
            "home_sos": h_sos,
            "away_sos": a_sos,
            "home_wins": h_wins,
            "away_wins": a_wins,
            "home_losses": h_losses,
            "away_losses": a_losses,
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            # Tournament context
            "is_early_season": is_early,
            "is_ncaa_tournament": is_postseason and is_march,
            "importance_multiplier": 1.3 if is_postseason else (0.8 if is_early else 1.0),
        }

        # Remove None values (keep only fields we have data for)
        patch = {k: v for k, v in patch.items() if v is not None}

        if patch:
            updates.append((game_id, patch))

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(all_games)} games...")

    print(f"  Updates ready: {len(updates)}, skipped: {skipped}")

    # ── Phase 3: Push updates to Supabase ──
    print(f"\n  Phase 3: Pushing updates to Supabase...")
    batch_size = 50
    total_success = 0

    # Determine which column to match on
    # Check if game_id is the primary identifier
    sample = all_games[0]
    match_col = "game_id" if "game_id" in sample and sample["game_id"] else "id"

    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        success = sb_patch_batch("ncaa_historical", match_col, batch)
        total_success += success
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(updates):
            print(f"    Updated {total_success}/{len(updates)} rows...")

    print(f"\n  ✅ Phase 1 complete: {total_success}/{len(updates)} rows updated")

    # ── Phase 4: Run heuristic backfill for derived columns ──
    print(f"\n  Phase 4: Heuristic backfill (spread, win_pct, pred scores)...")
    print(f"  This runs via Railway endpoint. Execute:")
    print(f"    curl -s -X POST $RAILWAY_API/train/ncaa | python3 -m json.tool")
    print(f"  The training function calls _ncaa_backfill_heuristic internally,")
    print(f"  which computes pred_home_score, pred_away_score, spread_home,")
    print(f"  win_pct_home from the stats we just backfilled.")
    print(f"\n  ALTERNATIVELY, to backfill heuristic columns directly:")
    print(f"  Add spread_home/win_pct_home computation to this script")
    print(f"  (uses the same _ncaa_backfill_heuristic logic from sports/ncaa.py)")

    print(f"\n{'='*70}")
    print(f"  DONE — Next steps:")
    print(f"  1. Retrain: curl -X POST $RAILWAY_API/train/ncaa")
    print(f"  2. Backtest: curl -X POST $RAILWAY_API/backtest/ncaa -d '{{\"min_train\":200}}'")
    print(f"  3. Compare ML accuracy to previous 73.8%")
    print(f"  4. If improved, run mega_sweep.py for optimal config")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
