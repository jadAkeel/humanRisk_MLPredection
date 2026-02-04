# 1) import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# 2) load data
# read the balanced dataset
df = pd.read_csv(r"./data/framingham.csv")
print(f"number of rows: {df.shape[0]}")
print(f"columns: {df.columns.tolist()}")

# 3) define features and target
# separate the target column from the rest
target_col = "TenYearCHD"
X = df.drop(target_col, axis=1)
y = df[target_col]

# 4) split data
# split dataset into training and testing sets (70/30)
# stratify keeps the target distribution consistent in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\nclass distribution in training set: {np.bincount(y_train)}")
print(f"class distribution in test set: {np.bincount(y_test)}")

# 5) gridsearchcv for best hyperparameters
# try different combinations to find the best decision tree settings
param_grid = {
    "criterion": ["gini"],
    "class_weight": ["balanced"],
    "max_depth": [4,5,6,8,None],
    "min_samples_split": [5,8,10,15,20],
    "min_samples_leaf": [5,8,10,15,20]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1
)

# train the grid search to find the best model parameters
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("\nbest hyperparameters:")
print(best_params)

# 6) cost-complexity pruning
# train a temporary model to get pruning values
clf = DecisionTreeClassifier(random_state=42, **best_params)
clf.fit(X_train, y_train)

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

train_scores = []

# try pruning with different alpha values to see which works best
for alpha in ccp_alphas:
    alpha = max(float(alpha), 0.0)
    tmp_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha, **best_params)
    tmp_clf.fit(X_train, y_train)
    y_train_pred = tmp_clf.predict(X_train)
    train_scores.append(balanced_accuracy_score(y_train, y_train_pred))

# choose the alpha that gives the highest training score
if len(train_scores) == 0:
    best_alpha = 0.0
else:
    best_alpha = ccp_alphas[np.argmax(train_scores)]
    best_alpha = float(max(best_alpha, 0.0))

print(f"\nbest pruning alpha: {best_alpha}")

# 7) train final model
# build the final decision tree using the chosen pruning value
final_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha, **best_params)
final_clf.fit(X_train, y_train)

# 8) evaluation
# test the model and print performance metrics
y_pred = final_clf.predict(X_test)
print("\nfinal evaluation\n")
print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print("\nclassification report:")
print(classification_report(y_test, y_pred))


# 9 and 10 are optional analyses, uncomment to run them
# show the most important features used by the tree
# show correlation of features with the target



# 9) feature importance
# show the top features that influence the prediction
feat_importances = pd.Series(final_clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
print("\ntop 15 important features")
print(feat_importances.head(15))

# 10) correlation analysis
# check which features are most correlated with the target
corr = df.corr()[target_col].sort_values(ascending=False)
print("\n=== top features by correlation with target ===")
print(corr.head(16))
