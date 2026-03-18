"""
Data loading, splitting, scaling, and DataLoader construction for tabular Bio-ML tasks.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


class DataManager:
    """Load tabular data, split, scale, and wrap in PyTorch DataLoaders.

    Parameters
    ----------
    config : dict
        Full experiment config containing at least ``data`` and ``training`` keys.
        See module docstring or README for the expected schema.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_cfg = config["data"]
        self.training_cfg = config.get("training", {})

        self.scaler = None
        self.label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self) -> Dict[str, Any]:
        """Run the full pipeline and return a data dict."""
        df = self._load_dataframe()
        X, y, class_names = self._encode_target(df)
        splits = self._split(X, y)
        splits = self._scale(splits)

        num_classes = len(class_names)
        num_features = splits["X_train"].shape[1]
        class_weights = self._compute_class_weights(splits["y_train"], num_classes)

        train_loader, val_loader = self._build_loaders(splits)

        result: Dict[str, Any] = {
            "X_train": splits["X_train"],
            "y_train": splits["y_train"],
            "X_val": splits["X_val"],
            "y_val": splits["y_val"],
            "X_test": splits["X_test"],
            "y_test": splits["y_test"],
            "train_loader": train_loader,
            "val_loader": val_loader,
            "num_features": num_features,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_weights": class_weights,
        }

        logger.info(
            "Data prepared: %d features, %d classes, train=%d val=%d test=%d",
            num_features,
            num_classes,
            len(splits["X_train"]),
            len(splits["X_val"]),
            len(splits["X_test"]),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataframe(self) -> pd.DataFrame:
        path: str = self.data_cfg["path"]
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        logger.info("Loaded %s  shape=%s", path, df.shape)
        return df

    def _encode_target(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        target_col: str = self.data_cfg["target_column"]
        y_raw = df[target_col].values
        X = df.drop(columns=[target_col]).values.astype(np.float32)

        y = self.label_encoder.fit_transform(y_raw).astype(np.int64)
        class_names: List[str] = list(self.label_encoder.classes_)
        return X, y, class_names

    def _split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        test_size: float = self.data_cfg.get("test_size", 0.2)
        val_size: float = self.data_cfg.get("val_size", 0.2)
        random_state: int = self.data_cfg.get("random_state", 42)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val (val_size is relative to original data)
        relative_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val,
            random_state=random_state,
            stratify=y_temp,
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def _scale(self, splits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scaler_name: str = self.data_cfg.get("scaler", "standard").lower()
        scaler_cls = _SCALERS.get(scaler_name)

        if scaler_cls is None:
            return splits

        self.scaler = scaler_cls()
        splits["X_train"] = self.scaler.fit_transform(splits["X_train"]).astype(
            np.float32
        )
        splits["X_val"] = self.scaler.transform(splits["X_val"]).astype(np.float32)
        splits["X_test"] = self.scaler.transform(splits["X_test"]).astype(np.float32)
        logger.info("Applied %s scaling.", scaler_name)
        return splits

    @staticmethod
    def _compute_class_weights(
        y_train: np.ndarray, num_classes: int
    ) -> torch.Tensor:
        """Inverse-frequency weights, normalized so they sum to ``num_classes``."""
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        # Avoid division by zero for classes absent in training set
        counts = np.maximum(counts, 1.0)
        inv_freq = 1.0 / counts
        weights = inv_freq / inv_freq.sum() * num_classes
        return torch.tensor(weights, dtype=torch.float32)

    def _build_loaders(
        self, splits: Dict[str, np.ndarray]
    ) -> tuple[DataLoader, DataLoader]:
        batch_size: int = self.training_cfg.get("batch_size", 64)
        oversampling: bool = self.training_cfg.get("oversampling", False)

        # Build TensorDatasets
        train_ds = TensorDataset(
            torch.tensor(splits["X_train"], dtype=torch.float32),
            torch.tensor(splits["y_train"], dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(splits["X_val"], dtype=torch.float32),
            torch.tensor(splits["y_val"], dtype=torch.long),
        )

        # Optional WeightedRandomSampler for class imbalance
        sampler: Optional[WeightedRandomSampler] = None
        shuffle_train = True
        if oversampling:
            class_counts = np.bincount(splits["y_train"])
            sample_weights = 1.0 / class_counts[splits["y_train"]]
            sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle_train = False  # sampler and shuffle are mutually exclusive

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, val_loader
