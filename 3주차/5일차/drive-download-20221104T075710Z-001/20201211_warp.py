### 이미지 처리 ###
import cv2 as cv
import numpy as np

# Read image with option
original_image = cv.imread('C:\\Users\\honesty\\Desktop\\test2.jpg', cv.IMREAD_COLOR)

#변환처리 : Warp1
height, width, channel = original_image.shape

srcPoint=np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint=np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
matrix = cv.getPerspectiveTransform(srcPoint, dstPoint)

warp_image = cv.warpPerspective(original_image, matrix, (width, height))

#Warp2
rows, cols, ch = original_image.shape

pts1 = np.float32([[200,100],[400,100],[200,200]])
pts2 = np.float32([[200,300],[400,200],[200,400]])

# pts1의 좌표에 표시. Affine 변환 후 이동 점 확인.
cv.circle(original_image, (200,100), 10, (255,0,0),-1)
cv.circle(original_image, (400,100), 10, (0,255,0),-1)
cv.circle(original_image, (200,200), 10, (0,0,255),-1)

M = cv.getAffineTransform(pts1, pts2)

warp_image2 = cv.warpAffine(original_image, M, (cols,rows))

cv.imshow('Original', original_image)
cv.imshow("Warp Perspective", warp_image)
cv.imshow("Warp2 Affine", warp_image2)

cv.waitKey(0)
cv.destroyAllWindows()