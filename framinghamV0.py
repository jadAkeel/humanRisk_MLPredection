import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# load the balanced dataset used for training
df = pd.read_csv("./data/framingham.csv")

# separate features from the target column
target_col = "TenYearCHD"
X = df.drop(target_col, axis=1)
y = df[target_col]

# split data into training and testing sets (70% train, 30% test)
# stratify keeps the class balance the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# create a basic decision tree model with default settings
clf = DecisionTreeClassifier(random_state=42)

# train the model on the training data
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# print different evaluation metrics to check how the model performs
print("accuracy:", accuracy_score(y_test, y_pred))
print("balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
