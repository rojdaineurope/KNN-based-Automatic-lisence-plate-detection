import cv2
import pytesseract
import re
import tkinter as tk
from gui import GarageDoorGUI
from config import AUTHORIZED_PLATES
import joblib
from ml.knn_features import extract_features

#tesseract way
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Authorized plates
AUTHORIZED_PLATES = {
    "34CSV725",
    "35ADA725",
    "16BJK1903"
}

#apllying our knn model
knn_model = joblib.load("ml/knn_model.pkl")

def is_plate_ml(img_roi):
    try:
        features = extract_features(img_roi)
        prediction = knn_model.predict([features])[0]
        return prediction == 1  # 1= plate, 0= non-plate
    except:
        return False

# for Webcam its adjusting 0 for normal camera its 1
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#Haar Cascade is a rule-based object detection method commonly used for license plate localization
harcascade = "haarcascade_plate_number.xml"
min_area = 500

#we determine the general Turkish plate regular expression
def is_valid_plate(plate):
    return bool(re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$', plate))


def process_plate(img_roi):
    
    #_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    
    
    # noise reduction especially for noisy and dirty images
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
     # CONTRAST STRETCHING
    min_val = gray.min()
    max_val = gray.max()

    if max_val - min_val > 0:
        gray = ((gray - min_val) * (255 / (max_val - min_val))).astype("uint8")
    
    
    # histogram equalization not necessary anymore
    #gray = cv2.equalizeHist(gray)

    #Adaptive Histogram Equalization (CLAHE) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
   


    # Adaptive threshold value for each pixel is calculated separately by looking at the neighboring pixels surrounding that pixel
    thresh = cv2.adaptiveThreshold(
      gray, 255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY,
      11, 2
)
    #Adaptive threshold is more stable compare than otsu
    
     #  OCR enhancing,start image acqusition
    h, w = thresh.shape
    thresh = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


    text = pytesseract.image_to_string(
        thresh,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)       #strict teserract settings 

    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

    if is_valid_plate(cleaned):
        return cleaned
    return None



def process_camera(app):
    global last_plate, last_time
    success, img = cap.read()
    if not success:
        return

    plate_cascade = cv2.CascadeClassifier(harcascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in plates:
        if w * h > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #the region of windowing
            roi = img[y:y+h, x:x+w]
            
            if not is_plate_ml(roi):
                continue  # if it is not a plate then dont give it to OCR

            plate_text = process_plate(roi)

            if not plate_text:
                continue


            if plate_text:

            # unauthorized plate
               if plate_text not in AUTHORIZED_PLATES:
                  app.add_entry(plate_text, "Unauthorized Access")
                  continue

             #authorized but already inside
               if plate_text in app.garage_entries:
                  app.add_entry(plate_text, "Already Inside")
                  continue

    #authorize but its first entry
               app.garage_entries.add(plate_text)
               app.add_entry(plate_text, "Authorized Access")


    cv2.imshow("Plate Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return

    app.master.after(100, process_camera, app)


def main():
    root = tk.Tk()
    garage_entries = set()
    app = GarageDoorGUI(root, None, garage_entries)
    root.after(100, process_camera, app)
    root.mainloop()


if __name__ == "__main__":
    main()
