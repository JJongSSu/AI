### 이미지 처리 ###
import cv2 as cv
import numpy as np
# 도로 이미지
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\road.jpg', cv.IMREAD_GRAYSCALE)
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\ball.jpg', cv.IMREAD_GRAYSCALE)
detect_image = original_image.copy()
# 변환처리 : Hough Transform
canny = cv.Canny(original_image, 400, 300, apertureSize=5, L2gradient = True)
# canny = cv.Canny(original_image, 0, 255)

#### Line Detection
# lines = cv.HoughLines(canny, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)
#
# for i in lines:
#     rho, theta = i[0][0], i[0][1]
#     a, b = np.cos(theta), np.sin(theta)
#     x0, y0 = a*rho, b*rho
#
#     scale = original_image.shape[0] + original_image.shape[1]
#
#     x1 = int(x0 + scale * -b)
#     y1 = int(y0 + scale * a)
#     x2 = int(x0 - scale * -b)
#     y2 = int(y0 - scale * a)
#
#     cv.line(detect_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv.circle(detect_image, (x0, y0), 3, (255, 0, 0), 5, cv.FILLED)

### Circle Detection
circles = cv.HoughCircles(original_image, cv.HOUGH_GRADIENT, 1, 100, param1 = 250, param2 = 10,
                           minRadius = 10, maxRadius = 120)
print(circles.shape)
for i in circles[0]:
    print("(", i[0], ",", i[1], "), ", i[2])
    # cv.circle(detect_image, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 3)
    cv.circle(detect_image, (i[0], i[1]), int(i[2]), (0, 0, 255), 3)


cv.imshow("Canny", canny)
cv.imshow("Hough Transform", detect_image)
cv.waitKey(0)
cv.destroyAllWindows()