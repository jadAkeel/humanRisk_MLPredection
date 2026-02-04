import pandas as pd
import os

# get the folder path of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# input dataset path and where to save the balanced version
input_file = os.path.join(BASE_DIR, "..", "data", "framingham.csv")
output_file = os.path.join(BASE_DIR, "..", "data", "framingham-balanced.csv")

# show where the file is coming from
print("reading dataset from:", input_file)

# load the original dataset
df = pd.read_csv(input_file)

# fill missing values in all columns depending on type
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        # fill numeric columns with their median
        df[col] = df[col].fillna(df[col].median())
    else:
        # fill categorical columns with the most common value
        df[col] = df[col].fillna(df[col].mode()[0])

# name of the target column used for balancing
target_col = "TenYearCHD"

# split dataset into two groups based on the target value
df0 = df[df[target_col] == 0].reset_index(drop=True)
df1 = df[df[target_col] == 1].reset_index(drop=True)

# find the smallest group size so both sides match
min_count = min(len(df0), len(df1))

# randomly sample equal amounts to balance the classes
df0_bal = df0.sample(n=min_count, random_state=42).reset_index(drop=True)
df1_bal = df1.sample(n=min_count, random_state=42).reset_index(drop=True)

# merge both balanced parts together
balanced_df = pd.concat([df0_bal, df1_bal], ignore_index=True)

# shuffle rows to avoid any ordering patterns
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# save the balanced dataset
balanced_df.to_csv(output_file, index=False)
print("✔ balanced & shuffled dataset saved:")
print(output_file)

# preview the first few rows
print("\nfirst 10 rows after shuffle:")
print(balanced_df.head(10))
