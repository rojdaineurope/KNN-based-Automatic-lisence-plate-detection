import cv2
import numpy as np


#this function turns image to the feture vector for giving KNN model
def extract_features(img):
    # grayscale turn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    gray = cv2.resize(gray, (128, 64))

    # HOG
    hog = cv2.HOGDescriptor(
        _winSize=(128,64),
        _blockSize=(16,16),
        _blockStride=(8,8),
        _cellSize=(8,8),
        _nbins=9
    )
    hog_features = hog.compute(gray).flatten()
    #HOG extracts edge orientations and shape information,this intelligence technique using with KNN for distinguish plate and non plate areas

    # intensity features
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    #i extracted contrast and light information from the image
    return np.hstack([hog_features, mean_intensity, contrast])

