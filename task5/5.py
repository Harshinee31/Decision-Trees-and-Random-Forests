# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree

# Step 2: Load the uploaded CSV file from Colab
from google.colab import files
uploaded = files.upload()  # Upload heart.csv when prompted

# Step 3: Read the CSV
df = pd.read_csv("heart.csv")

# Step 4: Quick overview of data
print("Shape of dataset:", df.shape)
print(df.head())

# Step 5: Define features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train a Decision Tree (default)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\nDefault Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Step 8: Train a pruned Decision Tree
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print("\nPruned Decision Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))

# Step 9: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Step 10: Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importances from Random Forest")
plt.show()

# Step 11: Cross-validation
dt_cv = cross_val_score(dt_pruned, X, y, cv=5)
rf_cv = cross_val_score(rf, X, y, cv=5)
print("\nCross-validation Accuracy (Decision Tree):", dt_cv.mean())
print("Cross-validation Accuracy (Random Forest):", rf_cv.mean())
