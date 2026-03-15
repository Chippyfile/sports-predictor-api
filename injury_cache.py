"""
injury_cache.py — Auto-caching NCAAB injury data from Covers.com

Caches injury metrics in memory. Refreshes every 4 hours.
Import and call get_team_injuries(team_name) from predict_ncaa.

Usage in prediction flow:
    from injury_cache import get_team_injuries
    
    home_inj = get_team_injuries("Alabama Crimson Tide")
    # Returns: {"missing_starters": 1, "injury_penalty": 0.5}
    
    game["home_injury_penalty"] = home_inj["injury_penalty"]
    game["away_injury_penalty"] = away_inj["injury_penalty"]
    game["injury_diff"] = home_inj["injury_penalty"] - away_inj["injury_penalty"]
    game["home_missing_starters"] = home_inj["missing_starters"]
    game["away_missing_starters"] = away_inj["missing_starters"]
"""

import os
import re
import json
import time
import requests
import threading
from datetime import datetime, timezone

COVERS_URL = "https://www.covers.com/sport/basketball/ncaab/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours
MAPPING_FILE = "covers_to_espn_mapping.json"

# Penalty weights
STATUS_WEIGHT = {
    "Out": 1.0, "Doubtful": 0.75, "Questionable": 0.5,
    "Day-To-Day": 0.25, "Probable": 0.1,
}
POS_WEIGHT = {"G": 1.0, "F": 1.0, "C": 1.2}

# Module-level cache
_cache = {
    "metrics": {},           # ESPN team name → {missing_starters, injury_penalty}
    "last_updated": 0,       # timestamp
    "lock": threading.Lock(),
}

# Load team name mapping once at import
_mapping = {}            # Covers name → ESPN name
_reverse_mapping = {}    # ESPN name (lowercase) → Covers name


def _load_mapping():
    """Load Covers → ESPN team name mapping."""
    global _mapping, _reverse_mapping
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE) as f:
            _mapping = json.load(f)
        _reverse_mapping = {v.lower(): k for k, v in _mapping.items()}
        print(f"  [injury_cache] Loaded {len(_mapping)} team mappings")
    else:
        print(f"  [injury_cache] ⚠️ {MAPPING_FILE} not found — injuries disabled")


def _scrape_and_compute():
    """Scrape Covers.com and compute per-team injury metrics."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [injury_cache] ⚠️ beautifulsoup4 not installed — injuries disabled")
        return {}

    try:
        r = requests.get(COVERS_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [injury_cache] ⚠️ Scrape failed: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")

    injuries_by_covers = {}
    for t in tables:
        el = t
        for _ in range(6):
            el = el.parent if el.parent else el
        img = el.find("img", alt=True)
        if not img:
            continue
        team_name = img["alt"].strip()

        rows = t.find_all("tr")
        players = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            player_text = cells[0].get_text(strip=True)
            pos_text = cells[1].get_text(strip=True)
            status_text = cells[2].get_text(strip=True)
            if pos_text == status_text or "no injuries" in player_text.lower() or player_text == "Player":
                continue

            match = re.match(
                r"(Out|Questionable|Day-To-Day|Doubtful|Probable)\s*-\s*(.+?)(?:\s*\(.+?\))?$",
                status_text
            )
            availability = match.group(1) if match else "Unknown"
            injury_type = match.group(2).strip() if match else "Unknown"

            # Skip redshirts
            if "redshirt" in injury_type.lower():
                continue

            players.append({"pos": pos_text, "availability": availability})

        if players:
            injuries_by_covers[team_name] = players

    # Convert Covers names → ESPN names and compute metrics
    metrics = {}
    for covers_name, players in injuries_by_covers.items():
        espn_name = _mapping.get(covers_name)
        if not espn_name:
            continue

        missing = sum(1 for p in players if p["availability"] == "Out")
        penalty = sum(
            STATUS_WEIGHT.get(p["availability"], 0.5) * POS_WEIGHT.get(p["pos"], 1.0)
            for p in players
        )
        metrics[espn_name.lower()] = {
            "missing_starters": missing,
            "injury_penalty": round(penalty, 2),
        }

    print(f"  [injury_cache] Refreshed: {len(metrics)} teams with injuries "
          f"({sum(m['missing_starters'] for m in metrics.values())} total out)")
    return metrics


def _refresh_if_needed():
    """Refresh cache if stale. Thread-safe."""
    now = time.time()
    if now - _cache["last_updated"] < CACHE_TTL_SECONDS:
        return  # Cache still fresh

    with _cache["lock"]:
        # Double-check after acquiring lock
        if now - _cache["last_updated"] < CACHE_TTL_SECONDS:
            return
        _cache["metrics"] = _scrape_and_compute()
        _cache["last_updated"] = time.time()


def get_team_injuries(espn_team_name):
    """
    Get injury metrics for a team by ESPN name.
    Auto-refreshes from Covers.com every 4 hours.
    
    Returns: {"missing_starters": int, "injury_penalty": float}
    """
    _refresh_if_needed()
    return _cache["metrics"].get(
        espn_team_name.lower(),
        {"missing_starters": 0, "injury_penalty": 0.0}
    )


def inject_injuries(game):
    """
    Inject injury data into a game dict before prediction.
    Call this in predict_ncaa before building features.
    
    Modifies game dict in-place, adding:
        home_injury_penalty, away_injury_penalty, injury_diff,
        home_missing_starters, away_missing_starters
    """
    home_name = game.get("home_team_name", "")
    away_name = game.get("away_team_name", "")

    home_inj = get_team_injuries(home_name)
    away_inj = get_team_injuries(away_name)

    game["home_injury_penalty"] = home_inj["injury_penalty"]
    game["away_injury_penalty"] = away_inj["injury_penalty"]
    game["injury_diff"] = home_inj["injury_penalty"] - away_inj["injury_penalty"]
    game["home_missing_starters"] = home_inj["missing_starters"]
    game["away_missing_starters"] = away_inj["missing_starters"]

    return game


# Load mapping on import
_load_mapping()
