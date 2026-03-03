import os

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://lxaaqtqvlwjvyuedyauo.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_ANON_KEY') or os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or ''
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
