import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

# Load sample data (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

#train test split, test_size signifies 30% data for testing
# random_state ensures reproducibility (data shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# following are only for multiclass classification, there are different methods and options for regression and binary classification

# evalmetric 'mlogloss' is suitable for multi-class classification
""" other options for multiclass classification are:
    mlogloss, merror, auc, aucpr (area under the curve)
"""

""" other options for objective are:
    multi:softprob - similar to softmax but o/ps probabiity distribution

"""
# objective softmax is activation function used for multi-class classification
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost model accuracy: {accuracy*100:.2f}%")


#confusion matrix : comparison of predicted classes against actual classes
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#precision : how many of the predicted positives are actually positive
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")

#recall : how many of the actual positives were correctly identified
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.4f}")

#f1 score : harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

#roc auc score : area under the curve for multi-class classification
# multiclass ovr (one vs rest) approach is used to calculate roc auc score
# similarly multiclass ova (one vs all) approach can also be used
y_prob = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"ROC AUC Score: {roc_auc:.4f}")