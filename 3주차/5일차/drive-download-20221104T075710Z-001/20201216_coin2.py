### 이미지 처리 - histogram ###
import cv2 as cv
import numpy as np
# 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\coin.jpg')

# Histogram view
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray_image, (5,5), 0)
(T, thresh) = cv.threshold(blurred, 155,255, cv.THRESH_BINARY)
# object -> white
(T, threshinv) = cv.threshold(blurred, 155,255, cv.THRESH_BINARY_INV)

coin = cv.bitwise_and(original_image, original_image, mask=threshinv)

#counting coin...
blurred = cv.GaussianBlur(gray_image, (17,17),0)
edged = cv.Canny(blurred, 80, 180)
(cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
print('-----')
coins = original_image.copy()
cv.drawContours(coins, cnts, -1, (0,255, 0), 2)
i = 0
#bounding box
for (i, c) in enumerate(cnts):
    (x,y,w,h) = cv.boundingRect(c)
    print("Coin #{}".format(i+1))
    coin = original_image[y:y+h, x:x+w]
    cv.imshow("Coin"+str(++i), coin)
    mask = np.zeros(gray_image.shape[:2], dtype='uint8')
    ((centerX, centerY), radius) = cv.minEnclosingCircle(c)
    cv.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    cv.bitwise_and(original_image, original_image, mask=mask)
    mask = original_image[y:y+h, x:x+w]
    cv.imshow("Masked"+str(++i), mask)

cv.imshow("Original", original_image)
cv.imshow("Edges", edged)
cv.imshow("Coins", coin)
cv.waitKey(0)
cv.destroyAllWindows()