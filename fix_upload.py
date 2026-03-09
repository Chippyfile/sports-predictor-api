import sys, os, joblib, io, base64, requests, pickle, numpy as np
from datetime import datetime, timezone
sys.path.insert(0, '.')
from ml_utils import StackedRegressor, StackedClassifier

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = os.environ.get('SUPABASE_ANON_KEY')

class Remapper(pickle.Unpickler):
    def find_class(self, module, name):
        # Only remap our custom classes from __main__ to ml_utils
        if module == '__main__' and name == 'StackedRegressor':
            return StackedRegressor
        if module == '__main__' and name == 'StackedClassifier':
            return StackedClassifier
        if module == 'ncaa_train_local' and name == 'StackedRegressor':
            return StackedRegressor
        if module == 'ncaa_train_local' and name == 'StackedClassifier':
            return StackedClassifier
        return super().find_class(module, name)

print('Loading model with class remapping...')
with open('ncaa_model_local.pkl', 'rb') as f:
    bundle = Remapper(f).load()
print(f'  n_train: {bundle["n_train"]}')
print(f'  reg class: {type(bundle["reg"]).__module__}.{type(bundle["reg"]).__name__}')
print(f'  clf class: {type(bundle["clf"]).__module__}.{type(bundle["clf"]).__name__}')

print('Compressing...')
buf = io.BytesIO()
joblib.dump(bundle, buf, compress=3)
compressed = buf.getvalue()
print(f'  Size: {len(compressed)/1024:.0f} KB')

b64 = base64.b64encode(compressed).decode('ascii')
print('Uploading...')
resp = requests.post(
    f'{SUPABASE_URL}/rest/v1/model_store',
    headers={'apikey': KEY, 'Authorization': f'Bearer {KEY}',
             'Content-Type': 'application/json',
             'Prefer': 'resolution=merge-duplicates'},
    json={'name': 'ncaa', 'data': b64,
          'metadata': {'trained_at': bundle.get('trained_at'),
                       'mae_cv': bundle.get('mae_cv'),
                       'n_train': bundle.get('n_train'),
                       'model_type': 'StackedEnsemble_LOCAL_FULL',
                       'size_bytes': len(compressed)},
          'updated_at': datetime.now(timezone.utc).isoformat()},
    timeout=300,
)
if resp.ok:
    print(f'  ✅ Upload successful')
else:
    print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')
