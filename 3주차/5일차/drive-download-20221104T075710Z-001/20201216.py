### 이미지 처리 - histogram ###
import cv2 as cv
import numpy as np
# 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\road.jpg')

# Histogram view
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
eq = cv.equalizeHist(gray_image)


### Bilateral
blurred_image = np.hstack([original_image,
                           cv.bilateralFilter(original_image, 5, 21, 21),
                           cv.bilateralFilter(original_image, 7, 31, 31),
                           cv.bilateralFilter(original_image, 9, 41, 41)])

# cv.imshow("Equalization", np.hstack([gray_image, eq]))
cv.imshow("Bi-lateral", blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()