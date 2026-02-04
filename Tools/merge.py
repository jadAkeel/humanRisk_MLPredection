import pandas as pd
import os 

# get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# build file paths in a safe way that works no matter where the script is run from
# base_dir points to the folder of this script, so the paths here are not affected by the current working directory
input_file = os.path.join(BASE_DIR, "..", "data", "framingham-balanced.csv")
input_file2 = os.path.join(BASE_DIR, "..", "data", "framingham-balancedV2.csv")
output_file = os.path.join(BASE_DIR, "..", "data", "framinghamMerged.csv")


# load both datasets
df1 = pd.read_csv(input_file)
df2 = pd.read_csv(input_file2)

# combine both datasets into one large dataframe
df_merged = pd.concat([df1, df2], ignore_index=True)

# save the merged result to a new csv file
df_merged.to_csv(output_file, index=False)
