#!/usr/bin/env python3
"""
mlb_push_enriched.py — Push enriched parquet columns back to Supabase mlb_historical
Run AFTER adding columns via SQL (ALTER TABLE).
Uses batch PATCH by game_pk for speed.
"""
import pandas as pd, requests, os, time, math

ENRICHED_COLS = [
    "temp_f", "wind_mph", "wind_out_flag",
    "home_sp_ip", "away_sp_ip",
    "home_platoon_delta", "away_platoon_delta",
    "pred_home_runs", "pred_away_runs",
    "pyth_residual_diff", "babip_luck_diff", "scoring_entropy_diff",
    "first_inn_rate_diff", "clutch_divergence_diff", "opp_adj_form_diff",
    "ump_run_env", "series_game_num",
    "scoring_entropy_combined", "first_inn_rate_combined",
    "sp_relative_fip_diff", "temp_x_park",
]

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
headers = {
    "apikey": key, "Authorization": f"Bearer {key}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

print("Loading parquet...")
df = pd.read_parquet("mlb_training_data.parquet")
print(f"  {len(df)} rows, pushing {len(ENRICHED_COLS)} columns")

# Batch update: group by chunks of 100 for progress reporting
BATCH_LOG = 500
updated = 0
errors = 0
t0 = time.time()

for i, (_, row) in enumerate(df.iterrows()):
    gpk = row.get("game_pk")
    if pd.isna(gpk):
        continue
    
    patch = {}
    for col in ENRICHED_COLS:
        val = row.get(col)
        if pd.notna(val):
            patch[col] = round(float(val), 4)
    
    if not patch:
        continue
    
    r = requests.patch(
        f"{url}/rest/v1/mlb_historical?game_pk=eq.{int(gpk)}",
        headers=headers, json=patch, timeout=10,
    )
    if r.ok:
        updated += 1
    else:
        errors += 1
        if errors <= 3:
            print(f"  ❌ game_pk={int(gpk)}: {r.status_code} {r.text[:100]}")
    
    if (i + 1) % BATCH_LOG == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(df) - i - 1) / rate
        print(f"  {i+1}/{len(df)} ({updated} updated, {errors} errors, {rate:.0f}/s, ETA {eta/60:.0f}m)")

elapsed = time.time() - t0
print(f"\n✅ Done in {elapsed/60:.1f}m — {updated} updated, {errors} errors")
