"""SVM Multiclass Model using One-Against-All (One-vs-Rest) strategy.
Paper Reference: Section "Rice disease classification", Eqs. 14-16 (Pages 13-14) & Table 11.
"""

import pickle
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


class RicePlantClassifier:
    """Classical One-Against-All Support Vector Machine classifier."""

    def __init__(
        self,
        kernel: str = "linear",
        c_param: float = 1.0,
        standardize: bool = True,
        seed: int = 42
    ):
        self.kernel = kernel.lower()
        self.c_param = c_param
        self.standardize = standardize
        self.seed = seed
        self.scaler: Optional[StandardScaler] = StandardScaler() if standardize else None
        self.model = self._build_model()
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self):
        if self.kernel == "linear":
            # LinearSVC natively implements One-vs-Rest and is optimized for high dimensions
            base = LinearSVC(
                C=self.c_param,
                loss="squared_hinge",
                dual=False,
                max_iter=3000,
                random_state=self.seed
            )
            # Wrap in OneVsRestClassifier for strict architectural fidelity to Eq. 14
            return OneVsRestClassifier(base)
        elif self.kernel == "rbf":
            base = SVC(
                kernel="rbf",
                C=self.c_param,
                gamma="scale",
                random_state=self.seed
            )
            return OneVsRestClassifier(base)
        elif self.kernel in ["poly", "polynomial"]:
            base = SVC(
                kernel="poly",
                degree=3,
                C=self.c_param,
                gamma="scale",
                random_state=self.seed
            )
            return OneVsRestClassifier(base)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}. Use 'linear', 'rbf', or 'poly'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RicePlantClassifier":
        """Fit scaler and One-vs-Rest SVM on training features."""
        X_mat = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_mat)
        else:
            X_scaled = X_mat

        self.model.fit(X_scaled, y)
        self.classes_ = np.array(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for feature array X."""
        X_mat = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_mat)
        else:
            X_scaled = X_mat

        return self.model.predict(X_scaled)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        X_mat = np.asarray(X, dtype=np.float64)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_mat)
        else:
            X_scaled = X_mat

        return self.model.decision_function(X_scaled)

    def save(self, filepath: str) -> None:
        """Serialize model, scaler, and configuration to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "kernel": self.kernel,
            "c_param": self.c_param,
            "standardize": self.standardize,
            "seed": self.seed,
            "scaler": self.scaler,
            "model": self.model,
            "classes": self.classes_
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: str) -> "RicePlantClassifier":
        """Deserialize model from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            kernel=payload["kernel"],
            c_param=payload["c_param"],
            standardize=payload["standardize"],
            seed=payload["seed"]
        )
        obj.scaler = payload["scaler"]
        obj.model = payload["model"]
        obj.classes_ = payload["classes"]
        return obj
