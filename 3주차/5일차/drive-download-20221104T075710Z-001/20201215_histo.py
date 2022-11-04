### 이미지 처리 - histogram ###
import cv2 as cv
import numpy as np
# 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\road.jpg')

# Histogram view
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
histo_image = np.zeros((original_image.shape[0], 256), dtype=np.uint8)
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])
cv.normalize(histogram, histogram, 0, histo_image.shape[0], cv.NORM_MINMAX)

for x, y in enumerate(histogram) :
    print("x =", x, ", y =", y)
    cv.line(histo_image, (x, histo_image.shape[0]), (x, histo_image.shape[0] - int(y)), 255)

result_image = np.hstack([gray_image, histo_image])

cv.imshow("Original", original_image)
cv.imshow("Histogram", result_image)
cv.waitKey(0)
cv.destroyAllWindows()