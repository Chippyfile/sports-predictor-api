"""
nba_ref_scraper.py — Scrape referee assignments from official.nba.com
Posted ~9AM ET each game day. No external deps beyond requests + re.

Usage:
    from nba_ref_scraper import get_refs_for_game
    refs = get_refs_for_game("IND", "LAL")
    # {"ref_1": "Nick Buchert", "ref_2": "Gediminas Petraitis", "ref_3": "Dannica Baroody"}
"""
import re, requests
from datetime import datetime, timezone, timedelta
from functools import lru_cache

URL = "https://official.nba.com/referee-assignments/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# NBA team city/name → abbreviation mapping for matching page text
# Maps any substring ESPN/nba.com uses → abbr (covers variants like "L.A. Lakers")
CITY_TO_ABBR = {
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "charlotte": "CHA",
    "chicago": "CHI", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN",
    "detroit": "DET", "golden state": "GSW", "houston": "HOU", "indiana": "IND",
    "l.a. clippers": "LAC", "l.a. lakers": "LAL",
    "la clippers": "LAC", "la lakers": "LAL",
    "los angeles clippers": "LAC", "los angeles lakers": "LAL",
    "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN",
    "new orleans": "NOP", "new york": "NYK", "oklahoma city": "OKC",
    "orlando": "ORL", "philadelphia": "PHI", "phoenix": "PHX",
    "portland": "POR", "sacramento": "SAC", "san antonio": "SAS",
    "toronto": "TOR", "utah": "UTA", "washington": "WAS",
}

# Cache keyed by date string so it refreshes daily
_cache = {}

def _fetch_assignments():
    """Fetch and parse today's assignments. Returns list of dicts."""
    today = datetime.now(timezone(timedelta(hours=-4))).strftime("%Y-%m-%d")  # ET
    if today in _cache:
        return _cache[today]

    try:
        r = requests.get(URL, headers=HEADERS, timeout=10)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"  [ref_scraper] fetch failed: {e}")
        return []

    # Extract table rows: <tr><td>...</td></tr>
    # Each row: game | crew_chief | referee | umpire | alternate
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    assignments = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) < 4:
            continue
        # Strip HTML tags and clean up
        def clean(html_cell):
            # Remove tags, decode entities, strip whitespace
            text = re.sub(r'<[^>]+>', '', html_cell)
            text = text.replace('&amp;', '&').replace('&#8211;', '-').strip()
            # Extract just the name (before the (#number) part)
            m = re.match(r'^(.*?)\s*\(#\d+\)', text)
            return m.group(1).strip() if m else text.strip()

        game = clean(cells[0])
        refs = [clean(cells[i]) for i in range(1, min(4, len(cells)))]
        refs = [r for r in refs if r]  # remove empty

        if ' @ ' in game:
            away_city, home_city = game.split(' @ ', 1)
            assignments.append({
                "game": game,
                "away_city": away_city.strip().lower(),
                "home_city": home_city.strip().lower(),
                "refs": refs,
            })

    _cache[today] = assignments
    print(f"  [ref_scraper] loaded {len(assignments)} games for {today}")
    return assignments


def get_refs_for_game(home_abbr: str, away_abbr: str) -> dict:
    """
    Look up referee assignments for a game.
    Returns {"ref_1": name, "ref_2": name, "ref_3": name} or {}
    """
    assignments = _fetch_assignments()
    if not assignments:
        return {}

    def _city_to_abbr(city_str):
        """Convert a city/team string from the page to an abbreviation."""
        s = city_str.strip().lower()
        # Direct lookup
        for key, abbr in CITY_TO_ABBR.items():
            if key in s:
                return abbr
        return s

    for a in assignments:
        h = _city_to_abbr(a["home_city"])
        aw = _city_to_abbr(a["away_city"])
        if h == home_abbr.upper() and aw == away_abbr.upper():
            refs = a["refs"]
            result = {}
            for i, name in enumerate(refs[:3], 1):
                result[f"ref_{i}"] = name
            print(f"  [ref_scraper] {away_abbr}@{home_abbr}: {refs}")
            return result

    print(f"  [ref_scraper] no match for {away_abbr}@{home_abbr} (available: {[a['game'] for a in assignments]})")
    return {}


if __name__ == "__main__":
    # Test
    print("Testing ref scraper...")
    refs = get_refs_for_game("CHA", "SAC")
    print(f"SAC@CHA: {refs}")
    refs = get_refs_for_game("NYK", "NOP")
    print(f"NOP@NYK: {refs}")
