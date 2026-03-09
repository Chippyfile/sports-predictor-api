import sys, pickle
sys.path.insert(0, '.')
from ml_utils import StackedRegressor, StackedClassifier

class DebugUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f'  PICKLE WANTS: module={module!r} name={name!r}')
        if module == '__main__' and name in ('StackedRegressor', 'StackedClassifier'):
            if name == 'StackedRegressor': return StackedRegressor
            return StackedClassifier
        if module == 'ncaa_train_local' and name in ('StackedRegressor', 'StackedClassifier'):
            if name == 'StackedRegressor': return StackedRegressor
            return StackedClassifier
        return super().find_class(module, name)

with open('ncaa_model_local.pkl', 'rb') as f:
    try:
        DebugUnpickler(f).load()
    except Exception as e:
        print(f'\n  FAILED AT: {e}')
