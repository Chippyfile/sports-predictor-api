import sys, json, requests

game_id = sys.argv[1] if len(sys.argv) > 1 else "401746082"
url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}"

r = requests.get(url)
data = r.json()

print(f"TOP-LEVEL KEYS: {list(data.keys())}\n")

if "gameInfo" in data:
    gi = data["gameInfo"]
    print(f"gameInfo keys: {list(gi.keys())}")
    if "venue" in gi:
        print(f"  venue: {json.dumps(gi['venue'], indent=2)}")
    if "officials" in gi:
        print(f"  officials: {json.dumps(gi['officials'], indent=2)}")
    if "attendance" in gi:
        print(f"  attendance: {gi['attendance']}")

if "header" in data:
    h = data["header"]
    if "competitions" in h and len(h["competitions"]) > 0:
        comp = h["competitions"][0]
        print(f"\nheader.competitions[0] keys: {list(comp.keys())}")
        if "venue" in comp:
            print(f"  header venue: {json.dumps(comp['venue'], indent=2)}")
        if "officials" in comp:
            print(f"  header officials: {json.dumps(comp['officials'], indent=2)}")

print("\n=== FULL STRUCTURE ===")
for k, v in data.items():
    if isinstance(v, dict):
        print(f"{k}: {list(v.keys())}")
    elif isinstance(v, list) and len(v) > 0:
        print(f"{k}: list[{len(v)}] -> {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}")
    else:
        print(f"{k}: {type(v).__name__}")
