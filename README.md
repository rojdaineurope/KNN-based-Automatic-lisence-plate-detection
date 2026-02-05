# KNN-based-Automatic-lisence-plate-recognition-system (ALPR)
integration with haar cascade model and KNN together to recognize the vehicles plates


This project is a **real-time Automatic License Plate Recognition (ALPR) system** developed using **classical digital image processing techniques**, **Optical Character Recognition (OCR)**, and a **lightweight machine learning approach (KNN)**.  
The system detects vehicle license plates from images or live webcam video, recognizes the plate characters, and performs **access control** based on a predefined authorization list.

## Project Overview

The goal of this project is to accurately recognize license plates **under different and challenging conditions** such as:
- Low or excessive lighting
- Shadows and reflections
- Dirty or low-contrast plates
- Camera angle variations

Instead of relying on deep learning models, the system focuses on **image enhancement techniques taught in Digital Image Processing courses**, combined with a simple but effective ML classifier.

## Features

-  Real-time license plate detection using a webcam  
-  Support for both **static images** and **live video frames**
-  Image enhancement pipeline:
  - Grayscale conversion
  - Noise reduction (Bilateral Filter)
  - Contrast stretching
  - Adaptive Histogram Equalization (CLAHE)
  - Adaptive thresholding
- License plate detection using **Haar Cascade**
- Plate / Non-plate verification using **K-Nearest Neighbors (KNN)**
- Character recognition using **Tesseract OCR**
- Garage access control system:
  - Authorized access
  - Unauthorized access warning
  - Already-inside detection
- Real-time logging of entries and exits into a CSV file

## System Architecture

The system consists of the following stages:

1. **Image Acquisition**
   - Webcam (real-time)
   - Static image dataset

2. **Image Preprocessing**
   - Grayscale conversion
   - Noise reduction with Bilateral Filter
   - Contrast enhancement
   - CLAHE for adaptive contrast handling
   - Adaptive thresholding
   - Image resizing for OCR improvement

3. **License Plate Detection**
   - Haar Cascade classifier for candidate detection
   - KNN classifier to eliminate false positives

4. **Character Recognition (OCR)**
   - Tesseract OCR
   - Post-processing with regex for Turkish license plate format

5. **Decision & Logging**
   - Authorized / Unauthorized decision
   - CSV-based real-time logging

## Machine Learning Component

A **KNN classifier** is used to distinguish between **plate** and **non-plate** regions detected by Haar Cascade.

### Extracted Features
- Grayscale intensity histogram
- Mean intensity
- Contrast (standard deviation)
- Histogram of Oriented Gradients (HOG)

### Performance Comparison

| Method            | Accuracy | Precision | Recall |
|-------------------|----------|-----------|--------|
| Baseline          | 82%      | 63%       | 63%    |
| KNN + HOG         | 91%      | 78%       | 90%    |
| HOG + Tuned k=2   | 93%      | 85%       | 85%    |

## Dataset

- Turkish license plate images
- Initial dataset from **Kaggle**
- Manually expanded with images captured under:
  - Daylight
  - Evening
  - Low-light conditions
  - Different angles and noise levels
- Total dataset size: **2500+ images**

## Technologies Used

- Python
- OpenCV
- NumPy
- scikit-learn
- Tesseract OCR
- Matplotlib
- Haar Cascade Classifier

## Limitations

- OCR performance may degrade due to:
  - Reflections and glare
  - Extreme lighting conditions
  - Phone screen brightness when plates are shown digitally
- No physical gate or hardware integration (software-only system)

## Future Improvements

- Camera exposure adjustment
- Motion blur reduction
- Larger and more diverse dataset
- Angle-aware plate detection
- Integration with physical access control systems

## References

- Anagnostopoulos et al., *License Plate Recognition from Still Images and Video Sequences*, IEEE
- Gonzalez & Woods, *Digital Image Processing*
- Turkish License Plate Dataset – Kaggle
- OpenCV Haar Cascade Documentation
- Smith, R., *Tesseract OCR Engine*
