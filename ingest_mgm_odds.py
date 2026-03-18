"""
MGM Sharp Money Data Ingest
Merges all_odds.csv into nba_training_data.parquet.

Key features computed:
  sharp_spread_signal: stake% - wager% for home spread (positive = sharps on home)
  sharp_ml_signal: stake% - wager% for home ML
  sharp_ou_signal: stake% - wager% for over
  public_home_spread_pct: % of bets on home spread (public sentiment)
  public_home_ml_pct: % of bets on home ML
  mgm_spread_home: MGM closing spread
  mgm_ou: MGM closing O/U
  mgm_home_ml: MGM home moneyline
  mgm_away_ml: MGM away moneyline

Usage:
  python ingest_mgm_odds.py                    # Merge into parquet
  python ingest_mgm_odds.py --preview          # Show merge stats only
"""

import os, sys
import pandas as pd
import numpy as np

# Team name mapping: MGM city names → our abbreviations
MGM_TO_ABBR = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS",
}


def load_mgm(csv_path="all_odds.csv"):
    """Load and normalize MGM betting data."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} games from {csv_path}")

    # Map team names to abbreviations
    df["home_abbr"] = df["home_team"].map(MGM_TO_ABBR)
    df["away_abbr"] = df["away_team"].map(MGM_TO_ABBR)

    unmapped = df[df["home_abbr"].isna()]["home_team"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unmapped teams: {unmapped}")

    # Normalize date (remove time portion: "2021-10-19-10:00" → "2021-10-19")
    df["game_date_clean"] = df["game_date"].str[:10]

    # Compute sharp money features
    # Stake% = proportion of total dollar amount (big bets = sharp money)
    # Wager% = proportion of total number of bets (many small bets = public)
    # Gap = stake% - wager% → positive means sharps are on that side

    # Spread sharp signal (home perspective)
    df["sharp_spread_signal"] = (
        pd.to_numeric(df["spread_home_stake_percentage"], errors="coerce") -
        pd.to_numeric(df["spread_home_wager_percentage"], errors="coerce")
    ).round(2)

    # ML sharp signal (home perspective)
    df["sharp_ml_signal"] = (
        pd.to_numeric(df["money_home_stake_percentage"], errors="coerce") -
        pd.to_numeric(df["money_home_wager_percentage"], errors="coerce")
    ).round(2)

    # O/U sharp signal (over perspective)
    df["sharp_ou_signal"] = (
        pd.to_numeric(df["total_over_stake_percentage"], errors="coerce") -
        pd.to_numeric(df["total_over_wager_percentage"], errors="coerce")
    ).round(2)

    # Public betting percentages (raw wager%)
    df["public_home_spread_pct"] = pd.to_numeric(df["spread_home_wager_percentage"], errors="coerce")
    df["public_home_ml_pct"] = pd.to_numeric(df["money_home_wager_percentage"], errors="coerce")
    df["public_over_pct"] = pd.to_numeric(df["total_over_wager_percentage"], errors="coerce")

    # Raw odds
    df["mgm_spread_home"] = pd.to_numeric(df["spread_home_points"], errors="coerce")
    df["mgm_ou"] = pd.to_numeric(df["total_over_points"], errors="coerce")
    df["mgm_home_ml"] = pd.to_numeric(df["money_home_odds"], errors="coerce")
    df["mgm_away_ml"] = pd.to_numeric(df["money_away_odds"], errors="coerce")
    df["mgm_home_spread_odds"] = pd.to_numeric(df["spread_home_odds"], errors="coerce")
    df["mgm_away_spread_odds"] = pd.to_numeric(df["spread_away_odds"], errors="coerce")

    # Juice imbalance: difference in spread odds between sides
    # Standard is -110/-110. If home is -115 and away is -105, book is shading home.
    df["spread_juice_imbalance"] = (
        df["mgm_home_spread_odds"].fillna(-110) - df["mgm_away_spread_odds"].fillna(-110)
    ).round(1)

    # Select output columns
    out_cols = [
        "game_date_clean", "home_abbr", "away_abbr",
        "sharp_spread_signal", "sharp_ml_signal", "sharp_ou_signal",
        "public_home_spread_pct", "public_home_ml_pct", "public_over_pct",
        "mgm_spread_home", "mgm_ou", "mgm_home_ml", "mgm_away_ml",
        "mgm_home_spread_odds", "mgm_away_spread_odds",
        "spread_juice_imbalance",
    ]

    result = df[out_cols].copy()
    result = result.rename(columns={"game_date_clean": "game_date", "home_abbr": "home_team", "away_abbr": "away_team"})

    print(f"\n  Sharp money stats:")
    print(f"    sharp_spread_signal: mean={result['sharp_spread_signal'].mean():.2f}, std={result['sharp_spread_signal'].std():.2f}")
    print(f"    sharp_ml_signal: mean={result['sharp_ml_signal'].mean():.2f}, std={result['sharp_ml_signal'].std():.2f}")
    print(f"    sharp_ou_signal: mean={result['sharp_ou_signal'].mean():.2f}, std={result['sharp_ou_signal'].std():.2f}")
    print(f"    Non-null coverage: {result['sharp_spread_signal'].notna().sum()}/{len(result)} ({result['sharp_spread_signal'].notna().mean()*100:.0f}%)")

    return result


def merge_into_training(mgm_df, parquet_path="nba_training_data.parquet"):
    """Merge MGM data into training parquet."""
    training = pd.read_parquet(parquet_path)
    print(f"\n  Training data: {len(training)} rows, {len(training.columns)} cols")

    # Drop existing MGM columns if re-running
    mgm_cols = [c for c in training.columns if c.startswith("sharp_") or c.startswith("mgm_") or c.startswith("public_") or c == "spread_juice_imbalance"]
    if mgm_cols:
        training = training.drop(columns=mgm_cols)
        print(f"  Dropped {len(mgm_cols)} existing MGM columns")

    # Merge on game_date + home_team + away_team
    merged = training.merge(
        mgm_df,
        on=["game_date", "home_team", "away_team"],
        how="left"
    )

    # Stats
    matched = merged["sharp_spread_signal"].notna().sum()
    print(f"  Matched: {matched}/{len(merged)} games ({matched/len(merged)*100:.1f}%)")

    # Save
    merged.to_parquet(parquet_path, index=False)
    print(f"  ✅ Saved to {parquet_path} ({len(merged)} rows, {len(merged.columns)} cols, {os.path.getsize(parquet_path)/1024:.0f} KB)")

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("  MGM Sharp Money Data Ingest")
    print("=" * 60)

    csv_path = "all_odds.csv"
    if not os.path.exists(csv_path):
        # Check uploads directory
        alt = "/mnt/user-data/uploads/all_odds.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            print(f"  ERROR: {csv_path} not found")
            sys.exit(1)

    mgm_df = load_mgm(csv_path)

    if "--preview" in sys.argv:
        print("\n  Preview mode — not merging")
        print(mgm_df.head(10).to_string())
    else:
        merged = merge_into_training(mgm_df)

        # Verify sharp money features
        print(f"\n  Verification:")
        for col in ["sharp_spread_signal", "sharp_ml_signal", "sharp_ou_signal",
                     "public_home_spread_pct", "mgm_spread_home"]:
            non_null = merged[col].notna().sum()
            print(f"    {col:30s}: {non_null:>5d} non-null ({non_null/len(merged)*100:.0f}%)")
