# Breast Cancer Classification using Logistic Regression

## Project Overview

This project implements a Logistic Regression model to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin Dataset.

The project demonstrates the complete machine learning workflow including:

* Data Loading
* Data Preprocessing
* Feature Scaling
* Train-Test Split
* Logistic Regression Model Training
* Model Evaluation
* ROC Curve Analysis
* Threshold Tuning

---

## Objective

Build a binary classification model using Logistic Regression and evaluate its performance using multiple classification metrics.

---

## Dataset

Dataset: Breast Cancer Wisconsin Dataset

Target Variable:

* M = Malignant (Cancerous)
* B = Benign (Non-Cancerous)

Features include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

and their corresponding worst and standard error measurements.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Loaded dataset
* Removed unnecessary columns
* Converted diagnosis labels:

  * M → 1
  * B → 0

### 2. Data Splitting

* Training Data: 80%
* Testing Data: 20%

### 3. Feature Scaling

Used StandardScaler to normalize feature values.

### 4. Model Training

Trained a Logistic Regression classifier using Scikit-Learn.

### 5. Model Evaluation

Evaluated the model using:

* Accuracy Score
* Precision
* Recall
* Confusion Matrix
* Classification Report
* ROC-AUC Score

---

## Results

Typical Results:

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | 95% - 99% |
| Precision | 95%+      |
| Recall    | 95%+      |
| ROC-AUC   | 0.99+     |

Actual results may vary slightly depending on train-test split.

---

## Confusion Matrix

The confusion matrix shows:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

It helps evaluate classification performance in detail.

---

## ROC Curve

The ROC Curve visualizes:

* True Positive Rate (Recall)
* False Positive Rate

A higher AUC value indicates better classification performance.

---

## Sigmoid Function

Logistic Regression uses the Sigmoid Function:

σ(x) = 1 / (1 + e^(-x))

The sigmoid function converts model outputs into probabilities between 0 and 1.

---

## Threshold Tuning

The default classification threshold is 0.5.

Changing the threshold affects:

* Precision
* Recall
* False Positives
* False Negatives

This project demonstrates threshold tuning using custom probability thresholds.

---

## Project Structure

```
Breast-Cancer-Logistic-Regression/
│
├── breast-cancer-wisconsin-data.csv
├── logistic_regression.py
├── README.md
├── confusion_matrix.png
└── roc_curve.png
```

---

## Conclusion

The Logistic Regression model successfully classifies breast cancer tumors with high accuracy and excellent ROC-AUC performance. This project demonstrates the effectiveness of Logistic Regression for binary classification problems in healthcare applications.
