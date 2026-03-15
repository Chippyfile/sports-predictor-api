#!/usr/bin/env python3
"""Patch ncaaSync.js to paginate the regrade query"""

with open('/Users/peterkm2/mlb-predictor/src/sports/ncaa/ncaaSync.js') as f:
    txt = f.read()

old = """  const allGraded = await supabaseQuery(
    `/ncaa_predictions?result_entered=eq.true&select=id,win_pct_home,spread_home,market_spread_home,market_ou_total,actual_home_score,actual_away_score,ou_total,pred_home_score,pred_away_score,home_team_id,away_team_id,home_adj_em,away_adj_em&limit=10000`
  );"""

new = """  // Paginate to get all graded records (Supabase caps at 1000/request)
  let allGraded = [];
  let offset = 0;
  const pageSize = 1000;
  while (true) {
    const page = await supabaseQuery(
      `/ncaa_predictions?result_entered=eq.true&select=id,win_pct_home,spread_home,market_spread_home,market_ou_total,actual_home_score,actual_away_score,ou_total,pred_home_score,pred_away_score,home_team_id,away_team_id,home_adj_em,away_adj_em&limit=${pageSize}&offset=${offset}&order=id.asc`
    );
    if (!page || !page.length) break;
    allGraded = allGraded.concat(page);
    onProgress?.(`Loading ${allGraded.length} records...`);
    if (page.length < pageSize) break;
    offset += pageSize;
  }"""

if old in txt:
    txt = txt.replace(old, new)
    with open('/Users/peterkm2/mlb-predictor/src/sports/ncaa/ncaaSync.js', 'w') as f:
        f.write(txt)
    print("Patched: added pagination to regrade")
else:
    print("ERROR: could not find old pattern")
    # Show what's there
    import re
    m = re.search(r'const allGraded.*?;', txt, re.DOTALL)
    if m:
        print(f"Found: {m.group()[:200]}")
