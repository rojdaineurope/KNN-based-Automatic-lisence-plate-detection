import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn_features import extract_features
import joblib

#this code for training knn
X = []  # feature vectors
y = []  # labels

# PLATE = 1
for file in os.listdir("DATASET/plate"):
    img = cv2.imread(f"DATASET/plate/{file}")
    if img is None:
        continue
    features = extract_features(img)
    X.append(features)
    y.append(1)

# NON-PLATE = 0
for file in os.listdir("DATASET/non_plate"):
    img = cv2.imread(f"DATASET/non_plate/{file}")
    if img is None:
        continue
    features = extract_features(img)
    X.append(features)
    y.append(0)


# KNN model
knn = KNeighborsClassifier(
    n_neighbors=2,
)
knn.fit(X, y)
joblib.dump(knn, "ml/knn_model.pkl")

print(" KNN model trained and saved.")


