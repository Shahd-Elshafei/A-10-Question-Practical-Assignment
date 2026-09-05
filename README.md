# ML Fundamentals: A 10-Question Practical Assignment

A collection of 10 self-contained machine learning exercises covering regression, classification, clustering, dimensionality reduction, and end-to-end pipelines using NumPy and scikit-learn. Each question is implemented, evaluated, and interpreted with a written discussion of the results.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Questions Covered](#questions-covered)
- [Requirements](#requirements)
- [Usage](#usage)
- [Authors](#authors)

## Overview

This notebook is a hands-on tour of core supervised and unsupervised ML techniques. It moves from implementing linear regression by hand (the normal equation) through polynomial regression, logistic regression with threshold tuning, tree-based classifiers, KNN, K-Means clustering, PCA, and a full preprocessing + modeling pipeline on a synthetic housing dataset — with a written interpretation accompanying each result.

## Questions Covered

1. **Linear Regression from Scratch vs. Scikit-Learn** — Implements the normal equation (`θ = (XᵀX)⁻¹Xᵀy`) manually and compares the resulting intercept/slope and predictions against scikit-learn's `LinearRegression`.
2. **Polynomial Regression (Degree 3)** — Fits a degree-3 polynomial regression and compares its R² against a simple linear model, showing the polynomial better captures diminishing returns in the data.
3. **Logistic Regression Pipeline & Threshold Analysis** — Builds a `StandardScaler` + `LogisticRegression` pipeline on the breast cancer dataset and analyzes the precision/recall trade-off at different classification thresholds, recommending a threshold that maximizes recall for a medical-diagnosis use case.
4. **Decision Tree Classifier & Feature Importance** — Trains a depth-limited Decision Tree, extracts and visualizes the top feature importances, and discusses the train/test accuracy gap as a sign of mild overfitting.
5. **Random Forest Classifier, Manual Metrics & Stratified CV** — Trains a Random Forest, manually derives precision/recall/F1 from the confusion matrix, and validates performance with stratified cross-validation.
6. **KNN Classification with Optimal K Search** — Scales features and sweeps K from 1–30 to find the value of K that maximizes test accuracy.
7. **K-Means Clustering, Elbow Curve & Silhouette Analysis** — Generates synthetic blob data, computes inertia across a range of K values, and uses the elbow method and silhouette score to identify the optimal number of clusters.
8. **PCA Dimensionality Reduction Analysis** — Fits PCA on scaled features and examines explained variance ratio (individual and cumulative) to determine how many components are needed to retain most of the variance.
9. **Pipeline with PCA & Support Vector Classifier** — Chains `StandardScaler → PCA (95% variance) → SVC` in a single pipeline and evaluates it with a classification report and cross-validation.
10. **End-to-End Housing Price Pipeline** — Builds a synthetic housing dataset (with missing values and a categorical `LOCATION` feature), constructs a `ColumnTransformer`-based preprocessing pipeline (imputation, scaling, one-hot encoding) feeding into a `RandomForestRegressor`, and discusses why R² can turn out negative on weakly correlated/synthetic targets.

## Requirements

- Python 3.8+
- Packages:
  ```
  numpy
  pandas
  matplotlib
  scikit-learn
  ```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Task3_Shahd_Walid_Said_Elshafei.ipynb
   ```
3. Run the cells sequentially from top to bottom — each question is self-contained and does not depend on the others, so individual cells can also be run in isolation for testing specific sections.
