# imports
# load libraries needed for data handling, model training, and evaluation
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from Tools.feature_enginner import FeatureEngineering

# load dataset and apply feature engineering
# this prepares all extra features needed for training
df = pd.read_csv("./data/framinghamMerged.csv")

fe = FeatureEngineering(df, save_csv=True, csv_path="./data/framingham_feature_enginnered.csv")
df_fe = fe.transform()

# separate the target from the features
target_col = "TenYearCHD"
X = df_fe.drop(target_col, axis=1)
y = df_fe[target_col]

# split into training and testing sets (70/30)
# stratify keeps class balance similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# grid search to find best hyperparameters for decision tree
param_grid = {
    "criterion": ["gini"],
    "class_weight": ["balanced"],
    "max_depth": [5, 4, 6, 8, None],
    "min_samples_split": [5, 8, 10, 15, 20],
    "min_samples_leaf": [5, 8, 10, 15, 20]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1
)

# train the grid search to find the best settings
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# use best settings to create a base model for pruning
clf = DecisionTreeClassifier(random_state=42, **best_params)
clf.fit(X_train, y_train)

# get pruning path values (alpha values)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

train_scores = []

# test different pruning values to find the best one
for ccp_alpha in ccp_alphas:
    ccp_alpha = max(float(ccp_alpha), 0.0)
    tmp_clf = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=ccp_alpha,
        **best_params
    )
    tmp_clf.fit(X_train, y_train)
    score = tmp_clf.score(X_train, y_train)
    train_scores.append(score)

# choose the alpha that gives the best training score
best_alpha = ccp_alphas[np.argmax(train_scores)]
print("best alpha:", best_alpha)

# train final model using best pruning value
final_clf = DecisionTreeClassifier(
    random_state=42, ccp_alpha=best_alpha, **best_params
)
final_clf.fit(X_train, y_train)

# save model for later predictions
joblib.dump(final_clf, "Model_predection/model.pkl")
print("\nmodel saved successfully as model.pkl")

# evaluate final model on the test set
y_pred = final_clf.predict(X_test)
print("\naccuracy:", accuracy_score(y_test, y_pred))
print("balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

# show the most important features used by the tree
feat_importances = pd.Series(final_clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
print("\n=== top 15 important features ===")
print(feat_importances.head(15))

# check how each column correlates with the target
corr = df.corr()[target_col].sort_values(ascending=False)
print("\n=== top features by correlation with target ===")
print(corr.head(16))
