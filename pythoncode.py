import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Load Dataset
# ==========================

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# KNN Model
# ==========================

k = 3

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = knn.predict(X_test)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Experiment with K Values
# ==========================

k_values = range(1, 21)

accuracy_scores = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = accuracy_score(y_test, pred)

    accuracy_scores.append(score)

# ==========================
# Best K Value
# ==========================

best_k = k_values[np.argmax(accuracy_scores)]

print("\nBest K:", best_k)
print("Best Accuracy:", max(accuracy_scores))

# ==========================
# Plot K vs Accuracy
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    accuracy_scores,
    marker='o'
)

plt.title("K Value vs Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)

plt.show()
