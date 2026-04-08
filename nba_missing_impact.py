"""
nba_missing_impact.py — Lineup-adjusted NBA predictions via BPM/VORP
=====================================================================
Computes missing player impact from ESPN injury reports + nba_player_impact table.
Called by predict_nba_full() to adjust margin and win probability.

Features produced (9 total):
  home_missing_margin, away_missing_margin, missing_margin_diff,
  home_missing_bpm, away_missing_bpm,
  home_missing_vorp, away_missing_vorp,
  home_missing_minutes, away_missing_minutes
"""

import requests as _req
import os, time
from difflib import get_close_matches
from datetime import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
_SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY or ''}",
}

# ESPN team IDs (abbr to ESPN numeric ID)
ESPN_TEAM_IDS = {
    "ATL": 1, "BOS": 2, "BKN": 17, "CHA": 30, "CHI": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10, "IND": 11,
    "LAC": 12, "LAL": 13, "MEM": 29, "MIA": 14, "MIL": 15, "MIN": 16,
    "NOP": 3, "NYK": 18, "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21,
    "POR": 22, "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
}

# ═══════════════════════════════════════════════════════════
# PLAYER IMPACT CACHE (loaded once per Railway container)
# ═══════════════════════════════════════════════════════════

_player_impact_cache = {}
_player_impact_cache_time = 0
_CACHE_TTL = 3600 * 6  # 6 hours

def _load_player_impact(season=2026):
    """Load player impact data from Supabase into memory."""
    global _player_impact_cache_time

    if season in _player_impact_cache and (time.time() - _player_impact_cache_time) < _CACHE_TTL:
        return _player_impact_cache[season]

    if not SUPABASE_KEY:
        print("  [impact] No SUPABASE_KEY -- skipping player impact")
        return []

    try:
        r = _req.get(
            f"{SUPABASE_URL}/rest/v1/nba_player_impact?season=eq.{season}"
            "&select=player_name,team,bpm,vorp,mpg,minutes_share,impact_score,margin_impact,bpm_weighted"
            "&limit=800",
            headers=_SB_HEADERS, timeout=15,
        )
        if r.ok:
            rows = r.json()
            _player_impact_cache[season] = rows
            _player_impact_cache_time = time.time()
            print(f"  [impact] Loaded {len(rows)} players from nba_player_impact")
            return rows
        else:
            print(f"  [impact] Supabase load failed: {r.status_code}")
            return []
    except Exception as e:
        print(f"  [impact] Supabase error: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# ESPN INJURY REPORT
# ═══════════════════════════════════════════════════════════

def get_out_players(team_abbr, game_id=None, game_date=None):
    """
    Get list of OUT players from ESPN.

    Strategy:
      1. If game_id provided, check game summary injuries (most accurate, game-day)
      2. Fallback: team page injury report

    Returns list of player names confirmed OUT.
    """
    out_players = []

    # ── Method 1: Game summary (game-day decisions) ──
    if game_id:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
            r = _req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.ok:
                data = r.json()
                espn_id = str(ESPN_TEAM_IDS.get(team_abbr, ""))

                # Roster section: players with status OUT/INACTIVE
                for team_data in data.get("rosters", []):
                    tid = str(team_data.get("team", {}).get("id", ""))
                    if tid != espn_id:
                        continue
                    for player in team_data.get("roster", []):
                        status = (player.get("status") or "").upper()
                        if status in ("OUT", "O", "INACTIVE"):
                            name = player.get("athlete", {}).get("displayName")
                            if name:
                                out_players.append(name)

                # Injuries section (separate from roster)
                for inj_group in data.get("injuries", []):
                    tid = str(inj_group.get("team", {}).get("id", ""))
                    if tid != espn_id:
                        continue
                    for item in inj_group.get("injuries", []):
                        status = (item.get("status") or item.get("type", {}).get("description", "")).upper()
                        if "OUT" in status:
                            name = item.get("athlete", {}).get("displayName")
                            if name and name not in out_players:
                                out_players.append(name)

                if out_players:
                    print(f"  [impact] {team_abbr} OUT (game summary): {out_players}")
                    return out_players
        except Exception as e:
            print(f"  [impact] Game summary fetch failed for {team_abbr}: {e}")

    # ── Method 2: Team page injuries ──
    espn_id = ESPN_TEAM_IDS.get(team_abbr)
    if not espn_id:
        return []

    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_id}?enable=roster,injuries"
        r = _req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if not r.ok:
            return []

        data = r.json()
        # Navigate nested injury structure — ESPN format varies
        injuries_raw = data.get("team", {}).get("injuries", [])
        for entry in injuries_raw:
            # Can be a category group (e.g., {"title": "Out", "injuries": [...]})
            items = entry.get("injuries", [entry]) if isinstance(entry, dict) else [entry]
            for item in items:
                desc = (item.get("type", {}).get("description", "")
                        or item.get("status", "")
                        or item.get("type", {}).get("abbreviation", "")).upper()
                if "OUT" in desc or desc in ("O", "OFS"):
                    name = item.get("athlete", {}).get("displayName")
                    if name and name not in out_players:
                        out_players.append(name)

        if out_players:
            print(f"  [impact] {team_abbr} OUT (team page): {out_players}")
    except Exception as e:
        print(f"  [impact] Team page fetch error for {team_abbr}: {e}")

    return out_players


# ═══════════════════════════════════════════════════════════
# IMPACT COMPUTATION
# ═══════════════════════════════════════════════════════════

def _sum_impact(team_abbr, out_names, all_players):
    """Sum impact metrics for OUT players on a team."""
    team_players = [p for p in all_players if p.get("team") == team_abbr]
    empty = {"margin": 0.0, "bpm_w": 0.0, "vorp": 0.0, "minutes": 0.0,
             "matched": [], "unmatched": []}
    if not team_players or not out_names:
        return empty

    team_names = [p["player_name"] for p in team_players]
    totals = {"margin": 0.0, "bpm_w": 0.0, "vorp": 0.0, "minutes": 0.0}
    matched, unmatched = [], []

    for name in out_names:
        # Exact (case-insensitive)
        hit = next((p for p in team_players if p["player_name"].lower() == name.lower()), None)
        # Fuzzy
        if not hit:
            close = get_close_matches(name, team_names, n=1, cutoff=0.55)
            if close:
                hit = next((p for p in team_players if p["player_name"] == close[0]), None)
                if hit:
                    print(f"  [impact] Fuzzy matched '{name}' -> '{hit['player_name']}'")
        if hit:
            totals["margin"] += float(hit.get("margin_impact", 0))
            totals["bpm_w"] += float(hit.get("bpm_weighted", 0))
            totals["vorp"] += float(hit.get("vorp", 0))
            totals["minutes"] += float(hit.get("mpg", 0))
            matched.append(hit["player_name"])
        else:
            unmatched.append(name)

    return {k: round(v, 3) for k, v in totals.items()} | {"matched": matched, "unmatched": unmatched}


def _zero_features():
    return {
        "home_missing_margin": 0.0, "away_missing_margin": 0.0, "missing_margin_diff": 0.0,
        "home_missing_bpm": 0.0, "away_missing_bpm": 0.0,
        "home_missing_vorp": 0.0, "away_missing_vorp": 0.0,
        "home_missing_minutes": 0.0, "away_missing_minutes": 0.0,
        "_home_out": [], "_away_out": [],
        "_home_matched": [], "_away_matched": [],
        "_home_unmatched": [], "_away_unmatched": [],
    }


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def compute_missing_impact(home_abbr, away_abbr, game_id=None, game_date=None,
                           home_out=None, away_out=None, season=2026):
    """
    Compute missing player impact for a single NBA game.

    Returns dict with 9 numeric features + diagnostic metadata (_prefixed).
    """
    players = _load_player_impact(season)
    if not players:
        print("  [impact] No player impact data -- returning zeros")
        return _zero_features()

    # Fetch OUT players from ESPN if not explicitly provided
    if home_out is None:
        home_out = get_out_players(home_abbr, game_id=game_id, game_date=game_date)
    if away_out is None:
        away_out = get_out_players(away_abbr, game_id=game_id, game_date=game_date)

    hi = _sum_impact(home_abbr, home_out, players)
    ai = _sum_impact(away_abbr, away_out, players)
    diff = round(hi["margin"] - ai["margin"], 3)

    print(f"  [impact] {home_abbr} OUT({len(home_out)}): margin={hi['margin']:+.2f} "
          f"| {away_abbr} OUT({len(away_out)}): margin={ai['margin']:+.2f} "
          f"| NET: {diff:+.2f} pts")

    return {
        "home_missing_margin": hi["margin"], "away_missing_margin": ai["margin"],
        "missing_margin_diff": diff,
        "home_missing_bpm": hi["bpm_w"], "away_missing_bpm": ai["bpm_w"],
        "home_missing_vorp": hi["vorp"], "away_missing_vorp": ai["vorp"],
        "home_missing_minutes": hi["minutes"], "away_missing_minutes": ai["minutes"],
        "_home_out": home_out, "_away_out": away_out,
        "_home_matched": hi["matched"], "_away_matched": ai["matched"],
        "_home_unmatched": hi.get("unmatched", []), "_away_unmatched": ai.get("unmatched", []),
    }


# ═══════════════════════════════════════════════════════════
# APPLY TO PREDICTION (pre-v28 direct adjustment)
# ═══════════════════════════════════════════════════════════

def adjust_prediction(margin, win_prob, impact_features, sigma=7.0):
    """
    Post-prediction margin adjustment using missing player impact.

    Use this UNTIL v28 is trained with impact features natively.
    After v28, the model handles it internally and this is unnecessary.

    Returns: (adjusted_margin, adjusted_win_prob, adjustment_pts)
    """
    import math

    diff = impact_features.get("missing_margin_diff", 0)
    if abs(diff) < 0.5:
        return margin, win_prob, 0.0

    # Positive diff = home missing more impact -> margin shifts toward away
    adj_margin = round(margin - diff, 2)
    adj_wp = 1.0 / (1.0 + math.exp(-adj_margin / sigma))
    adj_wp = round(max(0.05, min(0.95, adj_wp)), 4)

    print(f"  [impact] Adjusted: margin {margin:.1f} -> {adj_margin:.1f} "
          f"({diff:+.1f}), wp {win_prob:.3f} -> {adj_wp:.3f}")

    return adj_margin, adj_wp, round(diff, 2)
