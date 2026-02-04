import pandas as pd
import joblib
from ctgan import CTGAN
import sys, os

# add parent directory so we can import the feature engineering class
# this  line code to find absoulte path dynamic  of file if we transport this code to other pc work without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# this try exept to import  file generateficitvepatient in other file ( predection.py)
try:
    # try import when running inside package
    from .feature_enginner import FeatureEngineering
except ImportError:
    # fallback import when running as normal script
    from feature_enginner import FeatureEngineering

# path where the ctgan model will be stored or loaded from
MODEL_PATH = r"./Model_predection/ctgan_model.pkl"


# class to trainig ctgan model to creat patient like real dataset to execute predection dynamcily
class SyntheticPatientGenerator:
    def __init__(self):
        # load the cleaned dataset used for training ctgan
        df = pd.read_csv("./data/framingham_feature_enginnered.csv")

        # fill missing numeric values with medians
        num_cols = df.select_dtypes(include=['int64','float64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        # fill missing categorical values with modes
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # keep the prepared data for training the generator
        self.df_raw = df

        # check if a ctgan model already exists
        if os.path.exists(MODEL_PATH):
            # load existing model to avoid retraining
            print("loading existing ctgan model...")
            self.ctgan = joblib.load(MODEL_PATH)

        else:
            # train ctgan if no model is found
            print("training ctgan model... (slow the first time)")
            self.ctgan = CTGAN(epochs=50)
            self.ctgan.fit(self.df_raw)

            # make sure folder exists and save the model
            os.makedirs("./Model_predection", exist_ok=True)
            joblib.dump(self.ctgan, MODEL_PATH)
            print("ctgan model saved")

    def generate(self, n_patients=1):
        # generate a number of synthetic patients
        print(f"generating {n_patients} synthetic patients...")

        # ctgan creates new rows similar to the original dataset
        synthetic_df = self.ctgan.sample(n_patients)

        # return generated samples
        return synthetic_df
