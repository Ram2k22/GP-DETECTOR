import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import cv2
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name.')
# flags.DEFINE_string('enhance', 'bi', 'Training dataset name.')


def bilateral(image):
    
    # Apply bilateral filtering
    filtered_image = cv2.bilateralFilter(image, d=5, sigmaColor=80, sigmaSpace=80)

    return filtered_image.reshape(image.shape)

def gaussian_blur(image):
    
    filtered_image = cv2.GaussianBlur(image, (3,3), 5)
    
    return filtered_image.reshape(image.shape)


def histogram(image):
    
    if FLAGS.dataset == 'MNIST':
        
        filtered_image = cv2.equalizeHist(image)
    
    elif FLAGS.dataset == 'CIFAR10' or FLAGS.dataset == 'BT':
        
        h,s,v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        
        v = cv2.equalizeHist(v)
        
        filtered_image = cv2.merge((h,s,v))
        filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_HSV2BGR)
    
    return filtered_image.reshape(image.shape)

def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit = 2.0 , tileGridSize=(8, 8))
    
    if FLAGS.dataset == 'MNIST':
        filtered_image = clahe.apply(image)
    elif FLAGS.dataset == 'CIFAR10' or FLAGS.dataset == 'BT':
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


source = os.getcwd()+f'/adv_image/images/{FLAGS.dataset}'

# enhances = ['Original','Bilateral','Gaussian_Blur','Histogram','Adaptive_Histogram_Equalization','Median_Blur','Sharpen'] 
enhances = ['Original','bi','gb','hist','ahe','mb','sharpen'] 

for folder in enhances:
    if not(os.path.isdir(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{folder}')):
        os.makedirs(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{folder}')   
    
for file in tqdm(os.listdir(source)):
            
    img = cv2.imread(f'{source}/{file}')
    
        
    if FLAGS.dataset == 'MNIST' :
        img = np.mean(img, axis=2)
        img = np.expand_dims(img, axis=2)
        img = cv2.resize(img,(28,28))
        
    elif FLAGS.dataset == 'CIFAR10' :
        img = cv2.resize(img,(32,32))
    
    elif FLAGS.dataset == 'BT' :
        img = cv2.resize(img,(30,30))
        
    
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[0]}/{file}',img)
    
    img = img.astype(np.uint8)
        
    bi_image = bilateral(img)
    gb_image = gaussian_blur(img)       
    hist_image = histogram(img)
    ahe_image = adaptive_histogram_equalization(img)
    mb_image = median_blur(img)
    s_image = sharpen(img)
    
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[1]}/{file[:-4]}_bi.png', bi_image)
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[2]}/{file[:-4]}_gb.png', gb_image)        
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[3]}/{file[:-4]}_hist.png', hist_image)
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[4]}/{file[:-4]}_ahe.png', ahe_image)
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[5]}/{file[:-4]}_mb.png', mb_image)
    cv2.imwrite(f'{os.getcwd()}/adv_image/filter/{FLAGS.dataset}/{enhances[6]}/{file[:-4]}_sharpen.png', s_image)
    
    
    # cv2.imshow('',cv2.resize(bi_image,(500,500)))
    # cv2.waitKey(0)
    # cv2.imshow('',cv2.resize(gb_image,(500,500)))
    # cv2.waitKey(0)
    # cv2.imshow('',cv2.resize(hist_image,(500,500)))
    # cv2.waitKey(0)
            
                
print('--------------------- Done ----------------------------')

rows = len(enhances)
cols = len(os.listdir(source))

fig, axes = plt.subplots(rows, cols, figsize=(10,10))
for i,folder in enumerate(enhances):
    
    if folder == 'Adaptive_Histogram_Equalization':
        folder = 'Adaptive\nHistogram\nEqualization'
    axes[i, 0].text(-1.3, 0.5, folder, transform=axes[i, 0].transAxes,
                    va='center', ha='center', fontsize=12)
    if folder == 'Adaptive\nHistogram\nEqualization':
        folder = 'Adaptive_Histogram_Equalization'
    loc = os.getcwd()+f"/adv_image/filter/{FLAGS.dataset}/{folder}"
    for j,file in enumerate(os.listdir(loc)):
        
        image = cv2.imread(loc+ f"/{file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[i,j].imshow(image)
        axes[i,j].axis('off')
        


plt.suptitle('Enhancing Techniques on '+FLAGS.dataset + ' Dataset', fontweight='bold')
plt.show()



    



