"""
Quick check: Are pyth_residual, luck, and recent_form_diff real or zeros in training data?
Run from ~/Desktop/sports-predictor-api/
"""
import pandas as pd

df = pd.read_parquet("ncaa_training_data.parquet")
print(f"Total rows: {len(df)}")

for col in ["home_pyth_residual", "away_pyth_residual",
            "home_luck", "away_luck",
            "recent_form_diff"]:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        print(f"\n{col}:")
        print(f"  non-null: {s.notna().sum()}")
        print(f"  zeros:    {(s == 0).sum()}")
        print(f"  nonzero:  {(s != 0).sum()}")
        print(f"  mean:     {s.mean():.6f}")
        print(f"  std:      {s.std():.6f}")
        print(f"  sample:   {s.dropna()[s.dropna() != 0].head(5).tolist()}")
    else:
        print(f"\n{col}: NOT IN PARQUET")
