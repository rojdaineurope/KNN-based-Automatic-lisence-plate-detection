import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


#Görüntüleri oku
original = cv2.imread("Images/original/plate1.jpg", cv2.IMREAD_GRAYSCALE)
dark = cv2.imread("Images/dark/plate1_dark.jpg", cv2.IMREAD_GRAYSCALE)
bright = cv2.imread("Images/bright/plate1_bright.jpg", cv2.IMREAD_GRAYSCALE)

def show_histogram(img, title):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=[0,256],color="black")
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

#with this histograms we can see how pixel intensity is distributed
show_histogram(original, "Original Image Histogram")
show_histogram(dark, "Dark Image Histogram")
show_histogram(bright, "Bright Image Histogram")

#we enhanced the overbright plate 
img = cv2.imread("Images/bright/plate1_bright.jpg", cv2.IMREAD_GRAYSCALE)

#cv2.imshow("dark Plate", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#We apply a contrast stretching method for low contrast images
def contrast_stretching(img, low_percent=2, high_percent=98):
    
    p_low, p_high = np.percentile(img, (low_percent, high_percent))

    
    stretched = np.clip((img - p_low) * 255.0 / (p_high - p_low), 0, 255)

    return stretched.astype("uint8")

stretched =contrast_stretching(img)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("low contrast plate")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(stretched, cmap="gray")
plt.title("After contrast stretching")

plt.axis("off")
plt.show()


#For equalize the histograms we use first classic histogram equalization
# Histogram Equalization
equalized = cv2.equalizeHist(stretched)


# Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(stretched)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("bright image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE (Adaptive HE)")
plt.axis("off")

plt.show()
show_histogram(clahe_img, "After CLAHE Histogram")

#I made an enhance operation before that some darkest and dirty plates which i added on report then i save the enhanced versions to the enhanced folder!
#cv2.imwrite("Images/enhanced/enhancedd.jpeg", clahe_img)
#cv2.imwrite("Images/enhanced/camurenhanced6.jpeg", clahe_img)

