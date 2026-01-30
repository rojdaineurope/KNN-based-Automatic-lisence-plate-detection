import cv2
import os
import random

#In this code i split dataset images ,from the ROI's ,if model windowing wrong area then sended the non plate dataset, if it is plate then send to the plate dataset
CARS_DIR = "dataset/cars"
PLATE_DIR = "dataset/plate"
NON_PLATE_DIR = "dataset/non_plate"
CASCADE_PATH = "haarcascade_plate_number.xml"

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(NON_PLATE_DIR, exist_ok=True)

cascade = cv2.CascadeClassifier(CASCADE_PATH)

plate_count = 0
non_plate_count = 0


def resize_for_display(img, max_width=1200, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)

    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


for img_name in os.listdir(CARS_DIR):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(CARS_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 20)
    )

    
    for (x, y, w, h) in plates:

        
        ratio = w / h
        if ratio < 2 or ratio > 6:
            continue

        crop = gray[y:y + h, x:x + w]

       
        debug = img.copy()
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        debug_small = resize_for_display(debug)

        cv2.imshow("Plate Candidate | y=plate  n=non-plate", debug_small)
        key = cv2.waitKey(0)

        if key == ord("y"):
            cv2.imwrite(
                os.path.join(PLATE_DIR, f"plate_{plate_count}.jpg"),
                crop
            )
            plate_count += 1
        else:
            cv2.imwrite(
                os.path.join(NON_PLATE_DIR, f"non_plate_{non_plate_count}.jpg"),
                crop
            )
            non_plate_count += 1

    
    h_img, w_img = gray.shape

    for _ in range(2):
        x = random.randint(0, w_img - 120)
        y = random.randint(0, h_img - 40)

        crop = gray[y:y + 40, x:x + 120]

        cv2.imwrite(
            os.path.join(NON_PLATE_DIR, f"random_{non_plate_count}.jpg"),
            crop
        )
        non_plate_count += 1

cv2.destroyAllWindows()

print("Plate number:", plate_count)
print("Non-plate number:", non_plate_count)
