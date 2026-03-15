#!/usr/bin/env python3
"""
fix_select_cols.py — Replace select=* with explicit column list in retrain_and_upload.py
========================================================================================
Extracts all columns referenced by ncaa_build_features raw_cols defaults,
adds meta columns needed for training, and patches the Supabase query.

Usage:
  python3 fix_select_cols.py --dry-run   # show the select list
  python3 fix_select_cols.py              # patch retrain_and_upload.py
"""
import re, sys, argparse


def extract_raw_cols_from_ncaa():
    """Parse sports/ncaa.py to find all column names in raw_cols defaults."""
    with open("sports/ncaa.py") as f:
        src = f.read()

    # Find the raw_cols dict section
    match = re.search(r'raw_cols\s*=\s*\{(.*?)\}', src, re.DOTALL)
    if not match:
        print("ERROR: Could not find raw_cols in sports/ncaa.py")
        sys.exit(1)

    raw_section = match.group(1)
    # Extract all quoted key names
    cols = re.findall(r'"(\w+)"\s*:', raw_section)
    return sorted(set(cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1. Get columns from feature builder
    raw_cols = extract_raw_cols_from_ncaa()

    # 2. Add meta columns needed by retrain pipeline
    meta_cols = [
        "game_id", "game_date", "season",
        "actual_home_score", "actual_away_score",
        "home_team_id", "away_team_id",
        "home_team_abbr", "away_team_abbr",
        "home_conference", "away_conference",
        "neutral_site",
        # Referee columns (used by ref profile loader)
        "referee_1", "referee_2", "referee_3",
    ]

    all_cols = sorted(set(raw_cols + meta_cols))

    print(f"  Raw cols from feature builder: {len(raw_cols)}")
    print(f"  Meta cols for training:        {len(meta_cols)}")
    print(f"  Total unique columns:          {len(all_cols)}")

    select_str = ",".join(all_cols)
    print(f"  Select string length:          {len(select_str)} chars")

    if args.dry_run:
        print(f"\n  SELECT LIST:\n  {select_str}")
        print(f"\n  DRY RUN — no files modified.")
        return

    # 3. Patch retrain_and_upload.py
    filepath = "retrain_and_upload.py"
    with open(filepath) as f:
        txt = f.read()

    old_pattern = 'actual_home_score=not.is.null&select=*&order=season.asc'
    new_query = f'actual_home_score=not.is.null&select={select_str}&order=season.asc'

    if old_pattern not in txt:
        # Maybe already partially fixed?
        if 'select=*' in txt:
            # Find and replace any select=*
            txt_new = txt.replace('select=*', f'select={select_str}')
            if txt_new == txt:
                print("  ERROR: Could not find select=* in retrain_and_upload.py")
                sys.exit(1)
            txt = txt_new
            print("  Replaced select=* with explicit column list")
        else:
            print("  WARNING: select=* not found — may already be patched")
            return
    else:
        txt = txt.replace(old_pattern, new_query)
        print("  Replaced select=* with explicit column list")

    # Also revert limit back to 1000 since payload is now small enough
    if 'limit = 500' in txt:
        # This is in db.py, not retrain — skip
        pass

    # Backup + write
    with open(filepath + ".pre_select_fix", "w") as f:
        f.write(open(filepath).read())
    with open(filepath, "w") as f:
        f.write(txt)
    print(f"  Backup: {filepath}.pre_select_fix")
    print(f"  Patched: {filepath}")

    # Also fix db.py limit back to 1000
    try:
        with open("db.py") as f:
            db_txt = f.read()
        if 'limit = 500' in db_txt:
            db_txt = db_txt.replace('limit = 500', 'limit = 1000')
            with open("db.py", "w") as f:
                f.write(db_txt)
            print("  Restored db.py limit to 1000 (smaller payloads now)")
    except Exception as e:
        print(f"  Note: Could not update db.py limit: {e}")

    print(f"\n  DONE. Now retrain: python3 -u retrain_and_upload.py")
    print(f"  Should load all 64,881 rows.")


if __name__ == "__main__":
    main()
