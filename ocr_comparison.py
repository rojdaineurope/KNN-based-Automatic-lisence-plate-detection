import cv2
import pytesseract

def ocr_read(path):
    img = cv2.imread(path, 0)

    # adjust the image sizes
    height,width= img.shape
    img = cv2.resize(img, (width*2, height*2))

    # threshold
    _, img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text= pytesseract.image_to_string(
        img,
        config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    return text.strip()
img=cv2.imread("Images/camurlu/camurlu1.jpeg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("OCR", img); cv2.waitKey(0); cv2.destroyAllWindows()

#we gave dirty plate image to OCR first then enhanced image 
print("Original OCR:", ocr_read("Images/camurlu/camurlu1.jpeg"))
print("Enhanced OCR:", ocr_read("Images/enhanced/camurenhanced6.jpeg"))








