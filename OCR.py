import pytesseract
from PIL import Image
import cv2

# Path to the Tesseract executable (you might not need this on Ubuntu)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # You may need to set the correct path to Tesseract on your system

# Open an image file
image_path = 'image.jpeg'
img = Image.open(image_path)

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(img)

# Print the extracted text
print(text)

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

cv2.imshow("", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
