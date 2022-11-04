### 이미지 처리 - histogram ###
import cv2 as cv
import numpy as np
# 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\coin_1.jpg')

# Histogram view
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray_image, (5,5), 0)
(T, thresh) = cv.threshold(blurred, 155,255, cv.THRESH_BINARY)
# object -> white
(T, threshinv) = cv.threshold(blurred, 155,255, cv.THRESH_BINARY_INV)

# Adaptive Thresholding
threshinv1 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 4)
threshinv2 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3)


#변환처리 : 가장자리(엣지) 검출
# canny = cv.Canny(original_image, 100, 255)
laplacian = cv.Laplacian(gray_image, cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

sobelX = cv.Sobel(gray_image, cv.CV_64F, 1, 0)
sobelY = cv.Sobel(gray_image, cv.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv.bitwise_or(sobelX, sobelY)

coin = cv.bitwise_and(original_image, original_image, mask=threshinv)

#counting coin...

original_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
# blurred0 = cv.GaussianBlur(gray_image, (3,3),0)
blurred = cv.GaussianBlur(gray_image, (11,11),0)
edged = cv.Canny(blurred, 30, 150)
(cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
print('-----')
#print(cnts)
print('-----')
coins = original_image.copy()
cv.drawContours(coins, cnts, -1, (0,255, 0), 2)

cv.imshow("Original", original_image)
# cv.imshow("Thresholding", thresh)
# cv.imshow("Threshold-Inversing", threshinv)
# cv.imshow("Adaptive Threshold-Inv11", threshinv1)
# cv.imshow("Adaptive Threshold-Inv15", threshinv2)
# cv.imshow("Laplacian", laplacian)
# cv.imshow("Sobel", sobelCombined)
cv.imshow("Edges", edged)
cv.imshow("Coins", coin)
cv.waitKey(0)
cv.destroyAllWindows()