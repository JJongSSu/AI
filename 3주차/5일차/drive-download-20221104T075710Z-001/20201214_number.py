### 이미지 처리 ###
import cv2 as cv

# Read image with option

# original_image = cv.imread('C:\\Users\\honesty\\Desktop\\test2.jpg', cv.IMREAD_COLOR)
# original_image1 = cv.imread('C:\\Users\\honesty\\Desktop\\test1.jpg', cv.IMREAD_GRAYSCALE)
original_image1 = cv.imread('D:\\PersonalData\\Desktop\\car.jpg', cv.IMREAD_GRAYSCALE)
original_image2 = cv.imread('D:\\PersonalData\\Desktop\\test3.jpg', cv.IMREAD_GRAYSCALE)

#Morphology
#blur
blur_image = cv.blur(original_image1, (5, 5), anchor=(-1, -1), borderType=cv.BORDER_DEFAULT)

#sobel
sobel = cv.Sobel(blur_image, cv.CV_8UC1, 1, 0, borderType=cv.BORDER_DEFAULT)

#kernel 5 * 25
kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 120))

#binary
ret, bin_image = cv.threshold(sobel, 150, 255, cv.THRESH_BINARY)

#닫힘연산
morph_image = cv.morphologyEx(bin_image, cv.MORPH_CLOSE, kernel, iterations=1)

cv.imshow('Original1', original_image1)
cv.imshow('Blur', blur_image)
cv.imshow('Edge', sobel)
cv.imshow('Binary', bin_image)
cv.imshow('Morph', morph_image)

cv.waitKey(0)
cv.destroyAllWindows()