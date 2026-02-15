# Breast Cancer Classification - ML Assignment 2

## Problem Statement
Predict whether a tumor is malignant or benign using machine learning models.

## Dataset Description
The Breast Cancer dataset contains 4025 instances and 16 numerical features.
Target variable:
M → Malignant
B → Benign

## Models Used

| ML Model           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------          |----------|-----   |---------- |--------|----    |-----   |
| Logistic Regression|0.893168  |0.85272 |0.753425   |0.447154|0.561224|0.527186|
| Decision Tree      |0.837267  |0.694049|0.468750   |0.487805|0.478088|0.381845|
| KNN                |0.855901  |0.729776|0.568627   |0.235772|0.333333|0.300586|
| Naive Bayes        |0.795031  |0.749946|0.365385   |0.463415|0.408602|0.289690|
| Random Forest      |0.893168  |0.830979|0.768116   |0.430894|0.552083|0.523647|
| XGBoost            |0.891925  |0.838638|0.704545   |0.504065|0.587678|0.537249|

## Observations
Logistic Regression performed well due to linear separability.
Decision Tree slightly overfitted.
KNN sensitive to scaling.
Naive Bayes assumed feature independence.
Random Forest improved stability.
XGBoost showed highest overall performance.
