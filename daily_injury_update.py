"""
daily_injury_update.py — Scrape injuries from Covers.com, map to ESPN team names,
and push injury_penalty + missing_starters to Supabase for today's games.

Usage:
    python3 daily_injury_update.py           # Scrape + push for today's games
    python3 daily_injury_update.py --dry-run # Scrape + show what would be pushed
    python3 daily_injury_update.py --all     # Push for all current season games

Run daily before game predictions are generated.
"""

import os
import sys
import json
import requests
import re
from datetime import datetime, timezone
from bs4 import BeautifulSoup

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY', '')
MAPPING_FILE = 'covers_to_espn_mapping.json'
COVERS_URL = "https://www.covers.com/sport/basketball/ncaab/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Injury penalty weights
STATUS_WEIGHT = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.5,
    "Day-To-Day": 0.25,
    "Probable": 0.1,
}
POS_WEIGHT = {"G": 1.0, "F": 1.0, "C": 1.2}


def load_mapping():
    """Load Covers → ESPN team name mapping."""
    if not os.path.exists(MAPPING_FILE):
        print(f"  ❌ {MAPPING_FILE} not found. Run build_team_mapping.py first.")
        sys.exit(1)
    with open(MAPPING_FILE) as f:
        return json.load(f)


def scrape_injuries():
    """Scrape all NCAAB injuries from Covers.com."""
    r = requests.get(COVERS_URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")

    all_injuries = {}
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
            if pos_text == status_text:
                continue
            if "no injuries" in player_text.lower():
                continue
            if player_text == "Player":
                continue

            status_match = re.match(
                r"(Out|Questionable|Day-To-Day|Doubtful|Probable)\s*-\s*(.+?)(?:\s*\(\s*(.+?)\s*\))?$",
                status_text
            )
            if status_match:
                availability = status_match.group(1)
                injury_type = status_match.group(2).strip()
            else:
                availability = status_text.split("-")[0].strip() if "-" in status_text else status_text
                injury_type = "Unknown"

            players.append({
                "player": player_text,
                "pos": pos_text,
                "availability": availability,
                "injury_type": injury_type,
            })

        if players:
            all_injuries[team_name] = players

    return all_injuries


def compute_metrics(injuries):
    """Compute injury_penalty and missing_starters per team."""
    metrics = {}
    for team, players in injuries.items():
        # Filter out redshirts
        real = [p for p in players if "redshirt" not in p["injury_type"].lower()]
        missing = sum(1 for p in real if p["availability"] == "Out")
        penalty = sum(
            STATUS_WEIGHT.get(p["availability"], 0.5) * POS_WEIGHT.get(p["pos"], 1.0)
            for p in real
        )
        metrics[team] = {
            "missing_starters": missing,
            "injury_penalty": round(penalty, 2),
        }
    return metrics


def get_todays_games():
    """Fetch today's games from ncaa_predictions."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_date=eq.{today}&select=id,game_id,home_team_name,away_team_name"
    r = requests.get(url, headers={
        "apikey": KEY, "Authorization": f"Bearer {KEY}"
    }, timeout=30)
    if not r.ok:
        print(f"  ❌ Failed to fetch today's games: {r.status_code}")
        return []
    return r.json()


def get_season_games():
    """Fetch all current season games from ncaa_predictions."""
    all_data, offset, limit = [], 0, 1000
    while True:
        url = f"{SUPABASE_URL}/rest/v1/ncaa_predictions?select=id,game_id,home_team_name,away_team_name&limit={limit}&offset={offset}"
        r = requests.get(url, headers={
            "apikey": KEY, "Authorization": f"Bearer {KEY}"
        }, timeout=30)
        if not r.ok:
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data


def build_reverse_mapping(mapping):
    """Build ESPN name → Covers name reverse lookup."""
    reverse = {}
    for covers, espn in mapping.items():
        reverse[espn.lower()] = covers
    return reverse


def push_injuries(games, metrics, mapping, dry_run=False):
    """Match games to injury data and push to Supabase."""
    # Build ESPN → Covers reverse mapping
    reverse = build_reverse_mapping(mapping)

    updates = []
    for game in games:
        home = game.get("home_team_name", "")
        away = game.get("away_team_name", "")
        game_id = game.get("game_id") or game.get("id")

        # Look up Covers team name for home/away
        home_covers = reverse.get(home.lower())
        away_covers = reverse.get(away.lower())

        home_m = metrics.get(home_covers, {"missing_starters": 0, "injury_penalty": 0.0})
        away_m = metrics.get(away_covers, {"missing_starters": 0, "injury_penalty": 0.0})

        injury_diff = home_m["injury_penalty"] - away_m["injury_penalty"]

        updates.append({
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "home_missing_starters": home_m["missing_starters"],
            "away_missing_starters": away_m["missing_starters"],
            "home_injury_penalty": home_m["injury_penalty"],
            "away_injury_penalty": away_m["injury_penalty"],
            "injury_diff": round(injury_diff, 2),
        })

    if dry_run:
        print(f"\n  DRY RUN — would update {len(updates)} games:")
        # Show games with actual injuries
        with_injuries = [u for u in updates if u["home_injury_penalty"] > 0 or u["away_injury_penalty"] > 0]
        print(f"  Games with injury impact: {len(with_injuries)}")
        for u in sorted(with_injuries, key=lambda x: abs(x["injury_diff"]), reverse=True)[:20]:
            print(f"    {u['home_team'][:25]:25s} vs {u['away_team'][:25]:25s}  "
                  f"home_pen={u['home_injury_penalty']:5.2f}  away_pen={u['away_injury_penalty']:5.2f}  "
                  f"diff={u['injury_diff']:+5.2f}")
        return updates

    # Push to Supabase
    success, errors = 0, 0
    for u in updates:
        patch = {
            "home_injury_penalty": u["home_injury_penalty"],
            "away_injury_penalty": u["away_injury_penalty"],
            "injury_diff": u["injury_diff"],
            "home_missing_starters": u["home_missing_starters"],
            "away_missing_starters": u["away_missing_starters"],
        }
        gid = u["game_id"]
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/ncaa_predictions?game_id=eq.{gid}",
            headers={
                "apikey": KEY, "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json", "Prefer": "return=minimal",
            },
            json=patch,
            timeout=10,
        )
        if resp.ok:
            success += 1
        else:
            errors += 1

    print(f"  Pushed: {success} success, {errors} errors")
    return updates


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    all_games = "--all" in sys.argv

    print("=" * 60)
    print("  DAILY INJURY UPDATE")
    print("=" * 60)

    if not KEY:
        print("  ❌ SUPABASE_ANON_KEY not set")
        sys.exit(1)

    # 1. Load mapping
    print(f"\n  Loading team mapping...")
    mapping = load_mapping()
    print(f"  {len(mapping)} team mappings loaded")

    # 2. Scrape injuries
    print(f"  Scraping Covers.com...")
    injuries = scrape_injuries()
    total_injured = sum(len(v) for v in injuries.values())
    print(f"  {total_injured} injured players across {len(injuries)} teams")

    # 3. Compute metrics
    metrics = compute_metrics(injuries)
    teams_with_impact = sum(1 for m in metrics.values() if m["injury_penalty"] > 0)
    print(f"  {teams_with_impact} teams with injury impact")

    # 4. Get games
    if all_games:
        print(f"  Fetching all season games...")
        games = get_season_games()
    else:
        print(f"  Fetching today's games...")
        games = get_todays_games()
    print(f"  {len(games)} games found")

    if not games:
        print("  No games to update.")
        sys.exit(0)

    # 5. Match and push
    print(f"  Matching injuries to games...")
    updates = push_injuries(games, metrics, mapping, dry_run=dry_run)

    with_impact = [u for u in updates if u["home_injury_penalty"] > 0 or u["away_injury_penalty"] > 0]
    print(f"\n  Summary: {len(with_impact)}/{len(updates)} games have injury impact")

    # Save snapshot
    snapshot = {
        "scraped_at": datetime.now(timezone.utc).isoformat() + "Z",
        "total_injured": total_injured,
        "teams_with_injuries": len(injuries),
        "games_updated": len(updates),
        "games_with_impact": len(with_impact),
    }
    with open("injury_update_log.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"  Done. Log saved to injury_update_log.json")
