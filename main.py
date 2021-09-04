import numpy as np
import cv2 as cv
from funcs import *

image = cv.imread("input.jpg")
cv.namedWindow("main", cv.WINDOW_NORMAL)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# apply gaussian blur with 9x9 kernel
blurred = gauss_blur(gray, 9)

# inverse binary thresholding
thresh = ib_threshold(blurred, 50)

# morphological opening
kernel = np.ones((25, 25))
padded = padding(thresh, kernel)
eroded = morph(padded, kernel)
opened = morph(eroded, kernel, key=1)

# edge detection
edges, points = find_edges(opened)

# draw contour and display text
final = draw(image, points)

# display output
cv.imshow("main", final)
cv.waitKey(0)

