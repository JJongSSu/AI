### 이미지 처리 ###
import cv2 as cv

# Read image with option

# original_image = cv.imread('C:\\Users\\honesty\\Desktop\\test2.jpg', cv.IMREAD_COLOR)
# original_image1 = cv.imread('C:\\Users\\honesty\\Desktop\\test1.jpg', cv.IMREAD_GRAYSCALE)
original_image1 = cv.imread('C:\\Users\\honesty\\Desktop\\car.jpg', cv.IMREAD_GRAYSCALE)
original_image2 = cv.imread('C:\\Users\\honesty\\Desktop\\test3.jpg', cv.IMREAD_GRAYSCALE)

# #변환처리 : 이미지 덧셈/곱셈
# alpha = 0.6;  beta = 0.7
# add_image1 = original_image1 + original_image2
# add_image2 = original_image1 * 0.5 + original_image2 * 0.5
# add_image3 = original_image1 * alpha + original_image2 * (1-alpha)
# add_image4 = cv.addWeighted(original_image1, alpha, original_image2, beta, 0.0)

# #변환처리 : 이미지 add()
# alpha = 0.6;  beta = 0.7
# add_image1 = cv.add(original_image1, original_image2)
# add_image2 = cv.add(original_image1 * 0.5, original_image2 * 0.5)
# add_image3 = original_image1 * alpha + original_image2 * (1-alpha)
# add_image4 = cv.addWeighted(original_image1, alpha, original_image2, beta, 0.0)
#
# #filter
# mask = original_image1.copy()
# mask10 = mask.copy()
# mask10[:,:0] = 10
# add_image1_mask = cv.add(original_image1, original_image2, mask=mask10)

#Morphology
#blur
blur_image = cv.blur(original_image1, (5, 5), anchor=(-1, -1), borderType=cv.BORDER_DEFAULT)

#sobel
sobel = cv.Sobel(blur_image, cv.CV_8U, 1, 0, 3)

#kernel 5 * 25
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 25))

#binary
ret, bin_image = cv.threshold(sobel, 120, 255, cv.THRESH_BINARY)

#닫힘연산
morph_image = cv.morphologyEx(bin_image, cv.MORPH_CLOSE, kernel, iterations=5)

cv.imshow('Original1', original_image1)
# cv.imshow('Original2', original_image2)
# cv.imshow('Add', add_image1)
# cv.imshow('Add 50%', add_image2)
# cv.imshow('Add 60:40', add_image3)
# cv.imshow('Add Weighted 60:70', add_image4)
# cv.imshow('Add Mask10', add_image1_mask)
cv.imshow('Morph', morph_image)

cv.waitKey(0)
cv.destroyAllWindows()