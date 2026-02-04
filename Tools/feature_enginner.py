
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

    def cap_outliers(self, col):
        # Clip extreme values to reduce the effect of outliers
        low = self.df[col].quantile(0.005)
        high = self.df[col].quantile(0.995)
        self.df[col] = self.df[col].clip(low, high)

    def transform(self):
        """
        transform:
        Use this on the training data.
        Cleans the dataset and generates new features for model training.
        """
        # Cap outliers for main numeric columns
        for col in ["sysBP", "diaBP", "totChol", "glucose", "cigsPerDay", "BMI"]:
            if col in self.df.columns:
                self.cap_outliers(col)

        # Pulse pressure
        self.df["pulse_pressure"] = self.df["sysBP"] - self.df["diaBP"]

        # NEW: ratio of pressure difference
        self.df["difference_bp_ratio"] = self.df["pulse_pressure"] / self.df["sysBP"]

        # Obesity flag
        self.df["obese_flag"] = (self.df["BMI"] >= 30).astype(int)

        # Normalize glucose
        self.df["glucose_norm"] = (self.df["glucose"] - self.df["glucose"].mean()) / self.df["glucose"].std()

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
        df_new = df_new.copy()

        # Pulse pressure and ratio
        if "sysBP" in df_new and "diaBP" in df_new:
            df_new["pulse_pressure"] = df_new["sysBP"] - df_new["diaBP"]
            df_new["difference_bp_ratio"] = df_new["pulse_pressure"] / df_new["sysBP"]

        # Obesity flag
        if "BMI" in df_new:
            df_new["obese_flag"] = (df_new["BMI"] >= 30).astype(int)

        # Normalize glucose
        if "glucose" in df_new:
            df_new["glucose_norm"] = (df_new["glucose"] - df_new["glucose"].mean()) / df_new["glucose"].std()

        # Save to CSV if enabled
        if self.save_csv:
            df_new.to_csv(self.csv_path, index=False)

        return df_new
