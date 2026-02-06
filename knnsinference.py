import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# euclidean distance : calculates straight-line distance between two points
def euDist(x1, x2):
    return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

# knn prediction : finds k nearest neighbors and returns majority class
def knn_pred(training_data, training_label, test_point, k):
    distances = []
    for i in range(len(training_data)):
        distances.append((euDist(training_data[i], test_point), training_label[i]))
    # sort by distance (ascending)
    distances.sort()
    # get k closest neighbors
    neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in neighbors]
    # return most common label among neighbors
    return Counter(labels).most_common(1)[0][0]

# knn batch prediction : predicts labels for multiple test points
def knn_batch_pred(training_data, training_label, test_data, k):
    return [knn_pred(training_data, training_label, point, k) for point in test_data]

# sample data
training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
training_labels = ['A', 'A', 'A', 'B', 'B', 'B']
test_data = [[4, 5], [5, 6], [1, 1], [7, 7]]
test_labels = ['A', 'B', 'A', 'B']
k = 3

# make predictions
predictions = knn_batch_pred(training_data, training_labels, test_data, k)
print(f"Predictions: {predictions}")

# accuracy : how many predictions were correct
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# confusion matrix : comparison of predicted classes against actual classes
print("Confusion Matrix:")
print(confusion_matrix(test_labels, predictions))

# precision : how many of the predicted positives are actually positive
precision = precision_score(test_labels, predictions, average='weighted')
print(f"Precision: {precision:.4f}")

# recall : how many of the actual positives were correctly identified
recall = recall_score(test_labels, predictions, average='weighted')
print(f"Recall: {recall:.4f}")

# f1 score : harmonic mean of precision and recall
f1 = f1_score(test_labels, predictions, average='weighted')
print(f"F1 Score: {f1:.4f}")