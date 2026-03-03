import os,json,requests,joblib,io,base64
import numpy as np
from datetime import datetime
from config import SUPABASE_URL, SUPABASE_KEY, MODEL_DIR

def sb_get(table, params=""):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    
    all_data = []
    offset = 0
    limit = 1000  # Supabase default max per request
    
    while True:
        # Add range headers for pagination
        headers["Range"] = f"{offset}-{offset + limit - 1}"
        
        url = f"{SUPABASE_URL}/rest/v1/{table}?{params}"
        print(f"Fetching from Supabase: {url} (rows {offset}-{offset + limit - 1})")
        
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if not r.ok:
                print(f"Error response: {r.text[:200]}")
                break
                
            data = r.json()
            if not data:  # No more data
                break
                
            all_data.extend(data)
            print(f"Got {len(data)} rows, total so far: {len(all_data)}")
            
            # Check if we got less than the limit (end of data)
            if len(data) < limit:
                break
                
            offset += limit
            
        except Exception as e:
            print(f"Exception in sb_get: {str(e)}")
            break
    
    print(f"Total rows fetched: {len(all_data)}")
    return all_data


# ── Model cache ───────────────────────────────────────────────────────────────
# Models persisted to Supabase (model_store table) to survive Railway container
# restarts. Local disk + in-memory dict serve as fast cache layers.
#
# Required Supabase table (run once in SQL editor):
#   CREATE TABLE IF NOT EXISTS model_store (
#     name TEXT PRIMARY KEY,
#     data TEXT NOT NULL,
#     metadata JSONB DEFAULT '{}',
#     updated_at TIMESTAMPTZ DEFAULT now()
#   );
_models = {}

def save_model(name, obj):
    """Save model to local disk + in-memory cache + Supabase for persistence."""
    import base64, io
    # 1. Local disk + memory (fast access within current container)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(obj, path)
    _models[name] = obj

    # 2. Supabase persistence (survives container restarts)
    try:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        metadata = {
            "trained_at": obj.get("trained_at") if isinstance(obj, dict) else None,
            "mae_cv": obj.get("mae_cv") if isinstance(obj, dict) else None,
            "n_train": obj.get("n_train") if isinstance(obj, dict) else None,
            "model_type": obj.get("model_type", "") if isinstance(obj, dict) else "",
            "size_bytes": len(buf.getvalue()),
        }
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/model_store",
            headers=headers,
            json={"name": name, "data": b64, "metadata": metadata,
                  "updated_at": datetime.utcnow().isoformat()},
            timeout=60,
        )
        if resp.ok:
            print(f"  [model] {name} saved to Supabase ({len(buf.getvalue())/1024:.0f} KB)")
        else:
            print(f"  [model] Supabase save failed for {name}: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  [model] Supabase save error for {name}: {e}")

def load_model(name):
    """Load model: in-memory -> local disk -> Supabase fallback."""
    import base64, io
    # 1. In-memory cache (fastest)
    if name in _models:
        return _models[name]

    # 2. Local disk (survives within same container lifecycle)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        obj = joblib.load(path)
        _models[name] = obj
        return obj

    # 3. Supabase (survives container restarts)
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/model_store?name=eq.{name}&select=data",
            headers=headers, timeout=30,
        )
        if resp.ok:
            rows = resp.json()
            if rows and rows[0].get("data"):
                raw = base64.b64decode(rows[0]["data"])
                obj = joblib.load(io.BytesIO(raw))
                # Cache locally for fast subsequent access
                _models[name] = obj
                joblib.dump(obj, path)
                print(f"  [model] {name} restored from Supabase ({len(raw)/1024:.0f} KB)")
                return obj
    except Exception as e:
        print(f"  [model] Supabase load error for {name}: {e}")

    return None

# ═══════════════════════════════════════════════════════════════
# Time-Series CV Utility
