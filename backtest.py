# save as backtest.py and run with: python backtest.py
import os
from supabase import create_client

SUPABASE_URL = "https://lxaaqtqvlwjvyuedyauo.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")  # This will use your Railway env var

if not SUPABASE_KEY:
    print("Error: SUPABASE_ANON_KEY not set in environment")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Now this will work
result = supabase.table("mlb_historical") \
    .select("game_date,home_team,away_team,home_woba,away_woba,home_sp_fip,away_sp_fip,home_bullpen_era,away_bullpen_era,park_factor,temp_f,wind_mph,wind_out_flag,home_rest_days,away_rest_days,home_travel,away_travel,actual_home_runs,actual_away_runs") \
    .filter("season", "gte", 2024) \
    .order("random") \
    .limit(25) \
    .execute()

print(result.data)
