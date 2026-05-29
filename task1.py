import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv", sep="\t")

# Basic Information
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Cabin"].fillna("Unknown", inplace=True)

# Encode Categorical Variables
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Feature Scaling
scaler = StandardScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Fare"] = scaler.fit_transform(df[["Fare"]])

# Detect Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Boxplot of Fare")
plt.show()

# Remove Outliers using IQR
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

print("Original Shape:", df.shape)
print("After Outlier Removal:", df_clean.shape)

# Save cleaned dataset
df_clean.to_csv("cleaned_titanic.csv", index=False)

print("Data Cleaning Completed Successfully!")
