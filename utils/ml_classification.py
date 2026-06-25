"""
Paderborn Bearing Dataset - ML Classification
===============================================
Traditional ML pipeline (RF, GBT, XGBoost) with hand-crafted features,
plus cross-validation and evaluation helpers.
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. TRADITIONAL ML PIPELINE
# ============================================================

class TraditionalMLPipeline:
    """
    Traditional ML classification using hand-crafted features.
    Replicates and extends the approach in the Lessmeier et al. paper.
    """

    def __init__(self):
        self.pipelines = self._init_pipelines()
        self.results = {}

    def _init_pipelines(self) -> Dict:
        """Wrap each classifier in a Pipeline so the scaler is re-fit inside every CV fold."""
        classifiers = {
            'RF':  RandomForestClassifier(n_estimators=100, random_state=42),
            'GBT': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGB': XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=6, random_state=42,
                                  eval_metric='mlogloss', verbosity=0),
        }
        return {
            name: Pipeline([('scaler', StandardScaler()), ('model', clf)])
            for name, clf in classifiers.items()
        }

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> Dict:
        """
        Train all models and evaluate on test set.

        Args:
            X_train, y_train: Training data and labels
            X_test, y_test: Test data and labels
            feature_names: Optional list of feature names

        Returns:
            Dictionary of results per model
        """
        results = {}

        for name, pipe in self.pipelines.items():
            print(f"  Training {name}...", end=' ')
            # scaler is fit only on X_train inside the pipeline — no leakage
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                'accuracy': acc,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'report': classification_report(y_test, y_pred, output_dict=True),
            }

            print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

        self.results = results
        # fitted_pipelines now holds the full Pipeline objects (scaler + model)
        self.fitted_pipelines = dict(self.pipelines)
        return results

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       groups: Optional[np.ndarray] = None,
                       n_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation for all models.

        Args:
            X: Feature matrix.
            y: Encoded label array.
            groups: Bearing ID array (same length as X). When provided, uses
                StratifiedGroupKFold so no bearing spans train and val folds.
                When None, falls back to StratifiedKFold.
            n_folds: Number of CV folds.

        Returns:
            Dictionary with mean and std accuracy per model.
        """
        # Pass raw X — each Pipeline re-fits its own scaler inside each fold
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=n_folds)
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {}
        for name, pipe in self.pipelines.items():
            scores = cross_val_score(pipe, X, y, cv=cv, groups=groups,
                                     scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'all_scores': scores,
            }
            print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

        return results
