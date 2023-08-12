import cv2
from matplotlib import pyplot as pltd
import sys

img = cv2.imread('ball.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imag_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

try:
    xml_data = cv2.CascadeClassifier('ball.xml')
except Exception as e:
    print(e)


detecting = xml_data.detectMultiScale(img_gray,
                                   minSize = (30, 30))

count = len(detecting)

if count!=0:
    for (a, b, width, height) in detecting:
        cv2.rectangle(imag_rgb, (a, b), # Highlighting detected object with rectangle
                      (a + height, b + width),
                      (0, 275, 0), 9)


pltd.subplot(1, 1, 1)
pltd.imshow(imag_rgb)
pltd.show()
