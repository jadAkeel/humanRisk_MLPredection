import pandas as pd
import joblib
import sys, os

# add project root to python path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.generateficitvepatient import SyntheticPatientGenerator

# load the trained decision tree model
model = joblib.load(r"./Model_predection/model.pkl")

# load the synthetic patient generator (ctgan based)
gen = SyntheticPatientGenerator()

# numbers of healthy and sick synthetic patients we want to collect
TARGET_HEALTHY = 5
TARGET_SICK = 5

healthy_list = []
sick_list = []
count_h = 0
count_s = 0

print("\ngenerating patients by true generated chd...\n")

# stage 1 — keep generating synthetic patients until we collect the needed amount
while count_h < TARGET_HEALTHY or count_s < TARGET_SICK:

    # generate one synthetic patient
    patient_df = gen.generate(1)
    patient = patient_df.iloc[0:1].copy()

    # skip if ctgan didn't generate the chd column
    if "TenYearCHD" not in patient.columns:
        print("⚠ ctgan did not generate TenYearCHD! skipping...")
        continue

    chd_value = int(patient["TenYearCHD"].values[0])

    # keep track of the original generated chd label
    patient.loc[:, "TenYearCHD_generated"] = chd_value

    # add to the correct group based on the label
    if chd_value == 0 and count_h < TARGET_HEALTHY:
        healthy_list.append(patient)
        count_h += 1
        print(f" added healthy #{count_h} (chd=0)")

    elif chd_value == 1 and count_s < TARGET_SICK:
        sick_list.append(patient)
        count_s += 1
        print(f" added sick #{count_s} (chd=1)")

# stage 2 — combine all collected patients into one dataset
final_patients = pd.concat(healthy_list + sick_list).reset_index(drop=True)

# save true generated chd values before prediction
true_chd = final_patients["TenYearCHD_generated"].copy()

# remove chd columns before giving data to the model
for col in ["TenYearCHD", "TenYearCHD_generated"]:
    if col in final_patients.columns:
        final_patients = final_patients.drop(columns=[col])

# stage 3 — predict chd using the trained model
preds = model.predict(final_patients)
probs = model.predict_proba(final_patients)[:, 1]

# add prediction results back to the dataframe
final_patients["Pred"] = preds
final_patients["Prob"] = probs
final_patients["TenYearCHD_generated"] = true_chd

# display and save final output
print("\n       final patients (with comparison) \n")
print(final_patients)

final_patients.to_csv("patientPredection.csv", index=False)
print("\nsaved generated_patients.csv")
