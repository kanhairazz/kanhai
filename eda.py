import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("Coronary_Artery_Disease.csv")

# ==========================
# Basic Information
# ==========================
print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("\n" + "="*50)
print("COLUMN NAMES")
print(df.columns.tolist())

print("\n" + "="*50)
print("DATASET INFO")
print(df.info())

print("\n" + "="*50)
print("MISSING VALUES")
print(df.isnull().sum())

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print(df.describe())

# ==========================
# Numerical Columns
# ==========================
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nNumerical Columns:")
print(numeric_cols)

# ==========================
# Histograms
# ==========================
df[numeric_cols].hist(
    figsize=(20, 15),
    bins=20
)

plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# ==========================
# Boxplots
# ==========================
important_features = [
    'Age',
    'BMI',
    'BP',
    'TG',
    'LDL',
    'HDL'
]

plt.figure(figsize=(12,6))
sns.boxplot(data=df[important_features])
plt.title("Boxplots of Important Medical Features")
plt.xticks(rotation=45)
plt.show()

# ==========================
# Correlation Matrix
# ==========================
plt.figure(figsize=(16,12))

corr_matrix = df[numeric_cols].corr()

sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=False
)

plt.title("Correlation Matrix")
plt.show()

# ==========================
# Pair Plot
# ==========================
pair_features = [
    'Age',
    'BMI',
    'BP',
    'TG',
    'LDL',
    'HDL'
]

sns.pairplot(df[pair_features])
plt.show()

# ==========================
# Gender Distribution
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)

plt.title("Gender Distribution")
plt.show()

# ==========================
# Disease Distribution
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='Cath', data=df)

plt.title("Coronary Artery Disease Distribution")
plt.show()

# ==========================
# Age vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='Age',
    data=df
)

plt.title("Age vs Coronary Artery Disease")
plt.show()

# ==========================
# BMI vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='BMI',
    data=df
)

plt.title("BMI vs Coronary Artery Disease")
plt.show()

# ==========================
# BP vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='BP',
    data=df
)

plt.title("Blood Pressure vs Coronary Artery Disease")
plt.show()

# ==========================
# LDL vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='LDL',
    data=df
)

plt.title("LDL vs Coronary Artery Disease")
plt.show()

# ==========================
# HDL vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='HDL',
    data=df
)

plt.title("HDL vs Coronary Artery Disease")
plt.show()

# ==========================
# TG vs Disease
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(
    x='Cath',
    y='TG',
    data=df
)

plt.title("Triglycerides vs Coronary Artery Disease")
plt.show()

# ==========================
# Scatter Plot
# ==========================
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='LDL',
    y='HDL',
    hue='Cath',
    data=df
)

plt.title("LDL vs HDL")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
