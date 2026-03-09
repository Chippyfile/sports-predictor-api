import joblib, io, base64, requests, sys
from datetime import datetime

SUPABASE_URL = 'https://lxaaqtqvlwjvyuedyauo.supabase.co'
KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4YWFxdHF2bHdqdnl1ZWR5YXVvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4MDYzNTUsImV4cCI6MjA4NzM4MjM1NX0.UItPw2j2oo5F2_zJZmf43gmZnNHVQ5FViQgbd4QEii0'

# Must define classes before unpickling
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression

class StackedRegressor:
    def __init__(self, base_models, meta_model, base_scalers=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_scalers = base_scalers or [None] * len(base_models)
    def predict(self, X):
        base_preds = []
        for model, scaler in zip(self.base_models, self.base_scalers):
            X_in = scaler.transform(X) if scaler else X
            base_preds.append(model.predict(X_in))
        return self.meta_model.predict(np.column_stack(base_preds))

class StackedClassifier:
    def __init__(self, base_models, meta_model, base_scalers=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_scalers = base_scalers or [None] * len(base_models)
    def predict_proba(self, X):
        base_preds = []
        for model, scaler in zip(self.base_models, self.base_scalers):
            X_in = scaler.transform(X) if scaler else X
            base_preds.append(model.predict(X_in))
        return self.meta_model.predict_proba(np.column_stack(base_preds))

print('Loading local model...')
bundle = joblib.load('ncaa_model_local.pkl')
print(f'  n_train: {bundle["n_train"]}, features: {len(bundle["feature_cols"])}')

print('Compressing...')
buf = io.BytesIO()
joblib.dump(bundle, buf, compress=3)
compressed = buf.getvalue()
print(f'  Compressed: {len(compressed)/1024:.0f} KB')

b64 = base64.b64encode(compressed).decode('ascii')
print(f'  Base64: {len(b64)/1024:.0f} KB')

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
                       'model_type': bundle.get('model_type',''),
                       'size_bytes': len(compressed)},
          'updated_at': datetime.utcnow().isoformat()},
    timeout=300,
)
if resp.ok:
    print(f'  ✅ Upload successful ({len(compressed)/1024:.0f} KB)')
else:
    print(f'  ❌ Failed: {resp.status_code} {resp.text[:300]}')
