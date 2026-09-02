
"""
FeatureEngineering.py

This file handles feature engineering for the dataset.
It cleans the data, caps outliers, and generates new features.

Purpose:
- transform(): Prepare the training data with new features
- transform_new(): Apply the same feature engineering to new data,
  e.g., when generating synthetic patients for prediction.
"""

import numpy as np
import pandas as pd

class FeatureEngineering:
    def __init__(self, df, save_csv=False, csv_path=r"..\data\framingham_feature_eng.csv"):
        self.df = df.copy()
        self.save_csv = save_csv
        self.csv_path = csv_path
        self.clip_bounds = {}
        self.glucose_mean = None
        self.glucose_std = None
        self.is_fitted = False

    def fit(self):
        """Learn preprocessing statistics from the training data only."""
        for col in ["sysBP", "diaBP", "totChol", "glucose", "cigsPerDay", "BMI"]:
            if col in self.df.columns:
                self.clip_bounds[col] = (
                    self.df[col].quantile(0.005),
                    self.df[col].quantile(0.995),
                )

        if "glucose" in self.df.columns:
            self.glucose_mean = float(self.df["glucose"].mean())
            glucose_std = float(self.df["glucose"].std())
            self.glucose_std = glucose_std if glucose_std > 0 else 1.0

        self.is_fitted = True
        return self

    def cap_outliers(self, col):
        # Clip extreme values to reduce the effect of outliers
        if not self.is_fitted:
            self.fit()
        low, high = self.clip_bounds[col]
        self.df[col] = self.df[col].clip(low, high)

    def _transform_frame(self, frame):
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineering.fit() must run before transforming data")

        transformed = frame.copy()
        for col, (low, high) in self.clip_bounds.items():
            if col in transformed.columns:
                transformed[col] = transformed[col].clip(low, high)

        transformed["pulse_pressure"] = transformed["sysBP"] - transformed["diaBP"]
        transformed["difference_bp_ratio"] = (
            transformed["pulse_pressure"] / transformed["sysBP"].replace(0, np.nan)
        )
        transformed["obese_flag"] = (transformed["BMI"] >= 30).astype(int)
        transformed["glucose_norm"] = (
            transformed["glucose"] - self.glucose_mean
        ) / self.glucose_std
        return transformed

    def transform(self):
        """
        transform:
        Use this on the training data.
        Cleans the dataset and generates new features for model training.
        """
        if not self.is_fitted:
            self.fit()
        self.df = self._transform_frame(self.df)

        # Save to CSV if enabled
        if self.save_csv:
            self.df.to_csv(self.csv_path, index=False)

        return self.df

    def transform_new(self, df_new):
        """
        transform_new:
        Use this on new data (e.g., a synthetic patient or prediction data).
        Applies the same feature engineering as on the training data,
        so the model receives consistent features.
        """
        if not self.is_fitted:
            self.fit()
        df_new = self._transform_frame(df_new)

        # Save to CSV if enabled
        if self.save_csv:
            df_new.to_csv(self.csv_path, index=False)

        return df_new
