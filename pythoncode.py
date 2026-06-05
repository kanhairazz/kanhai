import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("heart.csv",sep="\t")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Decision Tree
# -------------------------
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

print(classification_report(y_test, dt_pred))

# Visualize Tree
plt.figure(figsize=(20,10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True
)
plt.show()

# -------------------------
# Random Forest
# -------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

# -------------------------
# Feature Importance
# -------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

# -------------------------
# Cross Validation
# -------------------------
dt_cv = cross_val_score(
    dt,
    X,
    y,
    cv=5
)

rf_cv = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("Decision Tree CV Accuracy:",
      dt_cv.mean())

print("Random Forest CV Accuracy:",
      rf_cv.mean())
