import cv2
import numpy as np

# dataset = input("MNIST or CIFAR10 :  ")
dataset = 'CIFAR10'
image = cv2.imread('image.jpg')
cv2.imshow("original",image)
cv2.waitKey()

def bilateral(image):
    
    # Apply bilateral filtering
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=80, sigmaSpace=80)

    return filtered_image.reshape(image.shape)

def gaussian_blur(image):
    
    filtered_image = cv2.GaussianBlur(image, (3,3), 5)
    
    return filtered_image.reshape(image.shape)


def histogram(image):
    
    if dataset == 'MNIST':
        filtered_image = cv2.equalizeHist(image)
    
    elif dataset == 'CIFAR10':
        
        h,s,v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        v = cv2.equalizeHist(v)
        
        filtered_image = cv2.merge((h,s,v))
        filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_HSV2BGR)
    
    return filtered_image.reshape(image.shape)


def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit = 3.0 , tileGridSize=(9, 9))
    
    if dataset == 'MNIST':
        filtered_image = clahe.apply(image)
    elif dataset == 'CIFAR10':
        h,s,v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))      
        v = clahe.apply(v)
        
        filtered_image = cv2.merge((h,s,v))
        filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_HSV2BGR)
    
    return filtered_image.reshape(image.shape)


def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image.reshape(image.shape)

def median_blur(image):
    
    filtered_image = cv2.medianBlur(image, 3)
    return filtered_image.reshape(image.shape)


cv2.imshow("bi",bilateral(image))
cv2.waitKey()

cv2.imshow("gb",gaussian_blur(image))
cv2.waitKey()

cv2.imshow("hist",histogram(image))
cv2.waitKey()

cv2.imshow("ahe",adaptive_histogram_equalization(image))
cv2.waitKey()

cv2.imshow("mb",median_blur(image))
cv2.waitKey()

cv2.imshow("sharpen",sharpen(image))
cv2.waitKey()
