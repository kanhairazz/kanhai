# ==========================================
# TASK 4: LOGISTIC REGRESSION CLASSIFICATION
# Breast Cancer Wisconsin Dataset
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("breast-cancer-wisconsin-data.csv",sep="\t")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

# ==========================================
# DATA CLEANING
# ==========================================

# Drop unnecessary columns

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# Convert diagnosis column
# M = 1 (Malignant)
# B = 0 (Benign)

df['diagnosis'] = df['diagnosis'].map({
    'M': 1,
    'B': 0
})

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")
print(accuracy)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# PRECISION
# ==========================================

precision = precision_score(y_test, y_pred)

print("\nPrecision Score")
print(precision)

# ==========================================
# RECALL
# ==========================================

recall = recall_score(y_test, y_pred)

print("\nRecall Score")
print(recall)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# ROC AUC SCORE
# ==========================================

roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score")
print(roc_auc)

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid()

plt.show()

# ==========================================
# THRESHOLD TUNING
# ==========================================

custom_threshold = 0.30

y_custom_pred = (
    y_prob >= custom_threshold
).astype(int)

print("\nThreshold =", custom_threshold)

print("\nCustom Confusion Matrix")
print(confusion_matrix(y_test, y_custom_pred))

print("\nCustom Classification Report")
print(classification_report(
    y_test,
    y_custom_pred
))

# ==========================================
# SIGMOID FUNCTION
# ==========================================

x = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-x))

plt.figure(figsize=(8, 5))

plt.plot(x, sigmoid)

plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid()

plt.show()

print("\nProject Completed Successfully!")
