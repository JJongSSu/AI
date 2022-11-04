### 이미지 처리 - histogram ###
import cv2 as cv
import numpy as np
# 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\road.jpg')

#masking
mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
(cX, cY) = (original_image.shape[1] // 2, original_image.shape[0] //2)
cv.rectangle(mask, (cX-75, cY-75), (cX+75, cY+75), 255, -1)
masked_image = original_image.copy()
masked_image = cv.bitwise_and(original_image, masked_image, mask=mask)

cv.imshow("Original", original_image)
cv.imshow("Masked", masked_image)
cv.waitKey(0)
cv.destroyAllWindows()