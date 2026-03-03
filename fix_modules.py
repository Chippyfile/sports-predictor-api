#!/usr/bin/env python3
import os, re

for fn in ['backtests.py', 'cron.py', 'ncaa_ratings.py']:
    if not os.path.exists(fn):
        print(f'  SKIP {fn}')
        continue
    lines = open(fn).readlines()
    out = []
    for line in lines:
        if line.strip().startswith('@app.route('):
            continue
        if line.strip().startswith('from flask import'):
            continue
        out.append(line)
    open(fn, 'w').writelines(out)
    print(f'  {fn}: {len(lines)} -> {len(out)} lines')
print('Done. Now: cp slim_main.py main.py')
