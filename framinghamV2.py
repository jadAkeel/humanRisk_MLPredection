# imports
# load libraries needed for data handling, model training, and evaluation
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from Tools.feature_enginner import FeatureEngineering

# load the raw dataset
df = pd.read_csv("./data/framinghamMerged.csv")

# separate the target from the features
target_col = "TenYearCHD"
X = df.drop(target_col, axis=1)
y = df[target_col]

# split into training and testing sets (70/30)
# stratify keeps class balance similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Learn preprocessing statistics from the training partition only, then apply
# the same transformation to the held-out data to prevent evaluation leakage.
fe = FeatureEngineering(X_train)
fe.fit()
X_train = fe.transform()
X_test = fe.transform_new(X_test)

# Keep the existing engineered dataset available to the synthetic-data helper,
# but transform it with statistics learned from the training partition.
df_fe = fe.transform_new(X)
df_fe[target_col] = y.to_numpy()
df_fe.to_csv("./data/framingham_feature_enginnered.csv", index=False)

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

validation_scores = []
pruning_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# test different pruning values using only the training data
for ccp_alpha in ccp_alphas:
    ccp_alpha = max(float(ccp_alpha), 0.0)
    tmp_clf = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=ccp_alpha,
        **best_params
    )
    scores = cross_val_score(
        tmp_clf,
        X_train,
        y_train,
        cv=pruning_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    validation_scores.append(scores.mean())

# choose alpha by cross-validated balanced accuracy instead of training accuracy,
# which otherwise favors an unpruned, overfit tree.
best_alpha = ccp_alphas[np.argmax(validation_scores)]
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
feat_importances = pd.Series(final_clf.feature_importances_, index=X_train.columns)
feat_importances = feat_importances.sort_values(ascending=False)
print("\n=== top 15 important features ===")
print(feat_importances.head(15))

# check how each column correlates with the target
corr = df.corr()[target_col].sort_values(ascending=False)
print("\n=== top features by correlation with target ===")
print(corr.head(16))
