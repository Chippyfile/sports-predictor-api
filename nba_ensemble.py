"""
nba_ensemble.py — EnsembleRegressor class for NBA 5-model ensemble.

Must be importable on Railway for pickle/joblib to deserialize the model.
Defined at module level (not inside __main__) for portability.
"""
import numpy as np


class EnsembleRegressor:
    """Simple averaging ensemble. Pickle-compatible.
    
    Stores individual models + uses Lasso's coef_ for SHAP display.
    Compatible with existing predict pipeline's reg.predict(X) interface.
    """
    def __init__(self, models, model_names, shap_model=None):
        self.models = models
        self.model_names = model_names
        self._shap_model = shap_model
        # Expose Lasso's coef_ for backward-compatible SHAP
        if shap_model is not None and hasattr(shap_model, 'coef_'):
            self.coef_ = shap_model.coef_
        elif models and hasattr(models[0], 'coef_'):
            self.coef_ = models[0].coef_
    
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.mean(preds, axis=0)
    
    def get_individual_predictions(self, X):
        """Return dict of name -> predictions for diagnostics."""
        return {name: m.predict(X) for name, m in zip(self.model_names, self.models)}
