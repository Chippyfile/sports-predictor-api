#!/usr/bin/env python3
"""Split main.py into modules. Run from sports-predictor-api dir."""
import os, shutil

SRC = "main.py"
if not os.path.exists(SRC):
    print("ERROR: main.py not found"); exit(1)

shutil.copy2(SRC, "main_monolith.py")
print("Backed up -> main_monolith.py")

with open(SRC) as f:
    lines = f.read().split("\n")

def find_line(m):
    for i,l in enumerate(lines):
        if m in l: return i+1
    return None

def xr(s,e): return "\n".join(lines[s-1:e])

# Locate all section boundaries
M = {}
for m in [
    "def sb_get(","def save_model(","def load_model(",
    "def _time_series_oof(","def _time_series_oof_proba(",
    "def _fit_negbin_k(","class StackedRegressor:","class StackedClassifier:",
    "def mlb_build_features(","def _mlb_merge_historical(",
    "def train_mlb(","def predict_mlb(",
    "def nba_build_features(","def train_nba(","def predict_nba(",
    "def ncaa_build_features(","def _ncaa_season_weight(",
    "def _flush_ncaa_batch(","def _ncaa_backfill_heuristic(",
    "def _ncaa_merge_historical(","def train_ncaa(","def predict_ncaa(",
    "def nfl_build_features(","def train_nfl(","def predict_nfl(",
    "def ncaaf_build_features(","def train_ncaaf(","def predict_ncaaf(",
    "def _negbin_draw(","def monte_carlo(",
    "def _histogram(","def accuracy_report(",
    "def index():","def health():",
    "def _active_sports(","def _log_training(",
    "def _should_promote(","def cron_auto_train(",
    "def _espn_cbb_get(",
    "def route_model_info(",
    "def route_compute_ncaa_efficiency(",
]:
    ln = find_line(m)
    if ln: M[m] = ln
print(f"Located {len(M)} markers")

open("config.py","w").write("import os\n\nSUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://lxaaqtqvlwjvyuedyauo.supabase.co')\nSUPABASE_KEY = os.environ.get('SUPABASE_ANON_KEY') or os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or ''\nMODEL_DIR = 'models'\nos.makedirs(MODEL_DIR, exist_ok=True)\n")
print("  config.py")

sb=M["def sb_get("]; sv=M["def save_model("]; ts=M["def _time_series_oof("]
open("db.py","w").write("import os,json,requests,joblib,io,base64\nimport numpy as np\nfrom datetime import datetime\nfrom config import SUPABASE_URL, SUPABASE_KEY, MODEL_DIR\n\n" + xr(sb, ts-2) + "\n")
print("  db.py")

nb=M["def _fit_negbin_k("]; sr=M["class StackedRegressor:"]; mf=M["def mlb_build_features("]
ac=M["def accuracy_report("]; hi=M.get("def _histogram(",ac-10); ix=M["def index():"]
open("ml_utils.py","w").write("import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import TimeSeriesSplit\nfrom sklearn.metrics import mean_absolute_error, brier_score_loss\ntry:\n    from xgboost import XGBRegressor, XGBClassifier\n    HAS_XGB = True\nexcept ImportError:\n    HAS_XGB = False\nfrom db import sb_get\n\n" + xr(ts,nb-2) + "\n\n" + xr(sr,mf-2) + "\n\n" + xr(hi,ix-2) + "\n")
print("  ml_utils.py")

os.makedirs("sports", exist_ok=True)
open("sports/__init__.py","w").write("")

nf=M["def nba_build_features("]
open("sports/mlb.py","w").write("import numpy as np, pandas as pd, traceback as _tb, shap\nfrom datetime import datetime\nfrom sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier\nfrom sklearn.linear_model import RidgeCV, LogisticRegression, ElasticNetCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.isotonic import IsotonicRegression\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom scipy.stats import nbinom\nfrom db import sb_get, save_model, load_model\nfrom ml_utils import HAS_XGB, _time_series_oof, _time_series_oof_proba, StackedRegressor, StackedClassifier\nif HAS_XGB:\n    from xgboost import XGBRegressor, XGBClassifier\n\n" + xr(nb,sr-2) + "\n\n" + xr(mf,nf-2) + "\n")
print("  sports/mlb.py")

nc=M["def ncaa_build_features("]
open("sports/nba.py","w").write("import numpy as np, pandas as pd, traceback as _tb, shap\nfrom datetime import datetime\nfrom sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.linear_model import RidgeCV, LogisticRegression, Ridge\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.isotonic import IsotonicRegression\nfrom sklearn.model_selection import cross_val_score, cross_val_predict\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom db import sb_get, save_model, load_model\nfrom ml_utils import HAS_XGB, _time_series_oof, _time_series_oof_proba, StackedRegressor, StackedClassifier\nif HAS_XGB:\n    from xgboost import XGBRegressor, XGBClassifier\n\n" + xr(nf,nc-2) + "\n")
print("  sports/nba.py")

nl=M["def nfl_build_features("]
open("sports/ncaa.py","w").write("import numpy as np, pandas as pd, traceback as _tb, shap, requests, time as _time\nfrom datetime import datetime\nfrom sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier\nfrom sklearn.linear_model import RidgeCV, LogisticRegression, Ridge, ElasticNetCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.isotonic import IsotonicRegression\nfrom sklearn.model_selection import cross_val_score, cross_val_predict\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom sklearn.metrics import mean_absolute_error, brier_score_loss\nfrom db import sb_get, save_model, load_model\nfrom config import SUPABASE_URL, SUPABASE_KEY\nfrom ml_utils import HAS_XGB, _time_series_oof, _time_series_oof_proba, StackedRegressor, StackedClassifier\nif HAS_XGB:\n    from xgboost import XGBRegressor, XGBClassifier\n\n" + xr(nc,nl-2) + "\n")
print("  sports/ncaa.py")

nca=M["def ncaaf_build_features("]
open("sports/nfl.py","w").write("import numpy as np, pandas as pd, shap\nfrom datetime import datetime\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom db import sb_get, save_model, load_model\n\n" + xr(nl,nca-2) + "\n")
print("  sports/nfl.py")

nd=M["def _negbin_draw("]
open("sports/ncaaf.py","w").write("import numpy as np, pandas as pd, shap\nfrom datetime import datetime\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom db import sb_get, save_model, load_model\n\n" + xr(nca,nd-2) + "\n")
print("  sports/ncaaf.py")

open("monte_carlo.py","w").write("import numpy as np\nfrom scipy.stats import nbinom\n\n" + xr(nd,ac-2) + "\n")
print("  monte_carlo.py")

at=M["def _active_sports("]; es=M["def _espn_cbb_get("]
open("cron.py","w").write("import json, traceback, time as _time, requests\nfrom datetime import datetime\nfrom flask import jsonify, request\nfrom config import SUPABASE_URL, SUPABASE_KEY\nfrom db import sb_get, save_model, load_model\n\n" + xr(at,es-2) + "\n")
print("  cron.py")

rc=M.get("def route_compute_ncaa_efficiency(")
if rc:
    open("ncaa_ratings.py","w").write("import requests, time as _time, traceback, json\nimport numpy as np, pandas as pd\nfrom datetime import datetime\nfrom config import SUPABASE_URL, SUPABASE_KEY\nfrom db import sb_get\n\n" + xr(es,rc-2) + "\n")
    print("  ncaa_ratings.py")

mi=M.get("def route_model_info(")
if mi and at:
    open("backtests.py","w").write("import numpy as np, pandas as pd, traceback, shap\nfrom datetime import datetime\nfrom flask import jsonify, request\nfrom sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier\nfrom sklearn.linear_model import RidgeCV, LogisticRegression, Ridge\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.isotonic import IsotonicRegression\nfrom sklearn.model_selection import cross_val_predict\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom sklearn.metrics import mean_absolute_error, brier_score_loss\nfrom db import sb_get, load_model\nfrom ml_utils import StackedRegressor, StackedClassifier\n\n" + xr(mi,at-2) + "\n")
    print("  backtests.py")

print("\n" + "=" * 50)
print("SPLIT COMPLETE")
print("=" * 50)
print("Now replace main.py with slim version.")
print("See main_slim_template.py or write your own.")
for fn in ["config.py","db.py","ml_utils.py","monte_carlo.py","cron.py","ncaa_ratings.py","backtests.py","sports/mlb.py","sports/nba.py","sports/ncaa.py","sports/nfl.py","sports/ncaaf.py"]:
    if os.path.exists(fn): print(f"  {sum(1 for _ in open(fn)):5d} lines  {fn}")
