"""
NBA Training Data Dump — Parquet Cache
Mirrors dump_training_data.py (NCAA pattern).

Usage:
  python dump_nba_training_data.py              # Pull fresh + save parquet
  python dump_nba_training_data.py --check      # Show cache status only

Creates:
  nba_training_data.parquet    — all nba_historical + completed nba_predictions
  nba_predictions_all.parquet  — all nba_predictions (including pending)
"""

import os, sys, time, requests
import pandas as pd

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lxaaqtqvlwjvyuedyauo.supabase.co")
KEY = os.environ.get("SUPABASE_ANON_KEY", "")

if not KEY:
    print("ERROR: Set SUPABASE_ANON_KEY environment variable")
    print("  export SUPABASE_ANON_KEY='your-key-here'")
    sys.exit(1)


def sb_get(table, params=""):
    """Paginated Supabase fetch."""
    all_data, offset, limit = [], 0, 1000
    while True:
        sep = "&" if params else ""
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}{sep}limit={limit}&offset={offset}"
        r = requests.get(url, headers={
            "apikey": KEY, "Authorization": f"Bearer {KEY}"
        }, timeout=60)
        if not r.ok:
            print(f"  Error {r.status_code}: {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_data


def dump():
    """Pull all NBA data from Supabase and save to parquet."""
    t0 = time.time()
    
    # ── 1. nba_historical (training corpus) ──
    print("  Fetching nba_historical...")
    hist_rows = sb_get("nba_historical",
                       "is_outlier_season=eq.false&select=*&order=game_date.asc")
    hist_df = pd.DataFrame(hist_rows) if hist_rows else pd.DataFrame()
    print(f"    nba_historical: {len(hist_df)} rows")
    
    # ── 2. nba_predictions (current season) ──
    print("  Fetching nba_predictions...")
    pred_rows = sb_get("nba_predictions", "select=*&order=game_date.asc")
    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
    print(f"    nba_predictions: {len(pred_df)} rows")
    
    # Save predictions separately (includes pending games)
    if len(pred_df) > 0:
        pred_df.to_parquet("nba_predictions_all.parquet", index=False)
        print(f"    Saved nba_predictions_all.parquet ({len(pred_df)} rows)")
    
    # ── 3. Combine for training (only completed games) ──
    # Add season to predictions if missing
    if len(pred_df) > 0:
        if "season" not in pred_df.columns:
            pred_df["season"] = pred_df["game_date"].apply(
                lambda d: int(d[:4]) + 1 if int(d[5:7]) >= 10 else int(d[:4])
            )
        # Filter to completed games
        completed = pred_df[pred_df.get("result_entered", False) == True].copy()
        if "actual_home_score" in completed.columns:
            completed = completed[completed["actual_home_score"].notna()]
        print(f"    Completed predictions: {len(completed)} rows")
    else:
        completed = pd.DataFrame()
    
    # Combine historical + completed predictions
    if len(hist_df) > 0 and len(completed) > 0:
        combined = pd.concat([hist_df, completed], ignore_index=True)
    elif len(hist_df) > 0:
        combined = hist_df
    else:
        combined = completed
    
    # Deduplicate
    if "game_date" in combined.columns and "home_team" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=["game_date", "home_team", "away_team"], keep="last"
        )
        if before != len(combined):
            print(f"    Deduped: {before} → {len(combined)} rows")
    
    combined = combined.sort_values("game_date").reset_index(drop=True)
    
    # Save training data
    combined.to_parquet("nba_training_data.parquet", index=False)
    
    elapsed = time.time() - t0
    print(f"\n  ✅ Saved nba_training_data.parquet")
    print(f"     {len(combined)} rows, {len(combined.columns)} columns")
    print(f"     Size: {os.path.getsize('nba_training_data.parquet') / 1024:.0f} KB")
    print(f"     Time: {elapsed:.0f}s")
    
    return combined


def load_cached():
    """Load cached parquet if it exists."""
    path = "nba_training_data.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        mod_time = os.path.getmtime(path)
        age_hours = (time.time() - mod_time) / 3600
        print(f"  Loaded {len(df)} rows from {path} (cached {age_hours:.1f}h ago)")
        return df
    return None


def check_cache():
    """Show cache status."""
    files = [
        "nba_training_data.parquet",
        "nba_predictions_all.parquet",
        "nba_elo_ratings.json",
        "nba_elo_snapshots.parquet",
        "nba_model_local.pkl",
    ]
    print("\n  NBA Local Cache Status:")
    print("  " + "-" * 50)
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            mod = time.ctime(os.path.getmtime(f))
            if f.endswith(".parquet"):
                df = pd.read_parquet(f)
                print(f"  ✅ {f:40s} {size/1024:>7.0f} KB  {len(df):>6} rows  {mod}")
            else:
                print(f"  ✅ {f:40s} {size/1024:>7.0f} KB  {mod}")
        else:
            print(f"  ❌ {f:40s} [not found]")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NBA Training Data Dump")
    print("=" * 60)
    
    if "--check" in sys.argv:
        check_cache()
    else:
        df = dump()
        
        # Column summary
        print(f"\n  Column Coverage:")
        print(f"  {'Column':<30s} {'Non-null':>10s} {'Coverage':>10s} {'Sample':>20s}")
        print(f"  {'-'*70}")
        for col in sorted(df.columns):
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            sample = str(df[col].dropna().iloc[0])[:20] if non_null > 0 else "[empty]"
            if pct < 100:
                print(f"  {col:<30s} {non_null:>10d} {pct:>9.1f}% {sample:>20s}")
        
        # Also check cache status
        check_cache()
