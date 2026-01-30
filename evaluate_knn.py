import cv2
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from knn_features import extract_features

#this is for datasets
X = []
y = []

# PLATE=1
for file in os.listdir("DATASET/plate"):
    img = cv2.imread(f"DATASET/plate/{file}")
    if img is None:
        continue
    X.append(extract_features(img))
    y.append(1)

# NON-PLATE = 0
for file in os.listdir("DATASET/non_plate"):
    img = cv2.imread(f"DATASET/non_plate/{file}")
    if img is None:
        continue
    X.append(extract_features(img))
    y.append(0)

X = np.array(X)
y = np.array(y)

# TRAIN / TEST SPLIT 20/80

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # %20 test %80 train
    random_state=42,
    stratify=y
)


#upload the model and train it
knn = joblib.load("ml/knn_model.pkl")
knn.fit(X_train, y_train)

#prediction on test
y_pred = knn.predict(X_test)

#metrics for measure performance
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy : {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall   : {rec:.2f}")


#CONFUSION MATRIX
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Plate", "Plate"]
)

disp.plot(cmap="Blues")
plt.title("KNN Plate / Non-Plate Confusion Matrix (Test Set)")
plt.show()
