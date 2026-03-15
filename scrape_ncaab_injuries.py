"""
Scrape NCAAB injuries from Covers.com
Run daily to populate injury columns in ncaa_historical/predictions.

Output: JSON with team → [player, pos, status, injury_type, date] mappings
Also computes: missing_starters count and injury_penalty per team.
"""

import requests
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

URL = "https://www.covers.com/sport/basketball/ncaab/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def scrape_injuries():
    """Scrape all NCAAB injuries from Covers.com."""
    r = requests.get(URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table")
    
    all_injuries = {}
    
    for t in tables:
        # Walk up 6 levels to find team container
        el = t
        for _ in range(6):
            el = el.parent if el.parent else el
        
        # Get team name from img alt
        img = el.find("img", alt=True)
        if not img:
            continue
        team_name = img["alt"].strip()
        
        # Parse table rows
        rows = t.find_all("tr")
        players = []
        
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            
            player_text = cells[0].get_text(strip=True)
            pos_text = cells[1].get_text(strip=True)
            status_text = cells[2].get_text(strip=True)
            
            # Skip description rows (they repeat text across all columns)
            if pos_text == status_text:
                continue
            # Skip "No injuries to report"
            if "no injuries" in player_text.lower():
                continue
            # Skip header rows
            if player_text == "Player":
                continue
            
            # Parse status: "Out - Knee ( Thu, Mar 5)" or "Questionable - Ankle ( Wed, Feb 18)"
            status_match = re.match(r"(Out|Questionable|Day-To-Day|Doubtful|Probable)\s*-\s*(.+?)(?:\s*\(\s*(.+?)\s*\))?$", status_text)
            if status_match:
                availability = status_match.group(1)
                injury_type = status_match.group(2).strip()
                date_str = status_match.group(3).strip() if status_match.group(3) else ""
            else:
                availability = status_text
                injury_type = "Unknown"
                date_str = ""
            
            players.append({
                "player": player_text,
                "pos": pos_text,
                "availability": availability,
                "injury_type": injury_type,
                "date_reported": date_str,
            })
        
        if players:
            all_injuries[team_name] = players
    
    return all_injuries


def compute_injury_metrics(injuries):
    """
    Compute per-team injury metrics for model features.
    
    Returns dict: team_name -> {missing_starters, injury_penalty}
    
    Penalty weights:
    - Out: 1.0 per player (full impact)
    - Doubtful: 0.75
    - Questionable: 0.5
    - Day-To-Day: 0.25
    - Probable: 0.1
    
    Position weights (starters matter more):
    - G (Guard): 1.0
    - F (Forward): 1.0  
    - C (Center): 1.2 (harder to replace)
    """
    STATUS_WEIGHT = {
        "Out": 1.0,
        "Doubtful": 0.75,
        "Questionable": 0.5,
        "Day-To-Day": 0.25,
        "Probable": 0.1,
    }
    POS_WEIGHT = {"G": 1.0, "F": 1.0, "C": 1.2}
    
    metrics = {}
    for team, players in injuries.items():
        # Filter out redshirts — they're not real injuries
        real_injuries = [p for p in players if "redshirt" not in p["injury_type"].lower()]
        
        missing = sum(1 for p in real_injuries if p["availability"] == "Out")
        penalty = sum(
            STATUS_WEIGHT.get(p["availability"], 0.5) * POS_WEIGHT.get(p["pos"], 1.0)
            for p in real_injuries
        )
        
        metrics[team] = {
            "missing_starters": missing,
            "injury_penalty": round(penalty, 2),
            "total_injured": len(real_injuries),
            "players": real_injuries,
        }
    
    return metrics


if __name__ == "__main__":
    print(f"Scraping NCAAB injuries from Covers.com...")
    injuries = scrape_injuries()
    
    total_players = sum(len(v) for v in injuries.values())
    teams_with_injuries = len(injuries)
    
    print(f"Found {total_players} injured players across {teams_with_injuries} teams")
    
    metrics = compute_injury_metrics(injuries)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  INJURY REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Sort by penalty descending
    sorted_teams = sorted(metrics.items(), key=lambda x: x[1]["injury_penalty"], reverse=True)
    
    print(f"\n  Top 20 most impacted teams:")
    for team, m in sorted_teams[:20]:
        print(f"    {team:25s} penalty={m['injury_penalty']:5.2f}  out={m['missing_starters']}  total={m['total_injured']}")
    
    # Save
    output = {
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "source": "covers.com",
        "teams_with_injuries": teams_with_injuries,
        "total_injured_players": total_players,
        "metrics": metrics,
    }
    
    with open("ncaab_injuries.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Saved to ncaab_injuries.json")
