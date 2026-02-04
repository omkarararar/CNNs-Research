import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.datasets import load_iris

# Load sample data (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

#train test split, test_size signifies 30% data for testing
# random_state ensures reproducibility (data shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# evalmetric 'mlogloss' is suitable for multi-class classification
# objective softmax is activation function used for multi-class classification
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost model accuracy: {accuracy*100:.2f}%")
