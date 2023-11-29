import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import cv2
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name.')



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
    
    elif FLAGS.dataset == 'CIFAR10':
        
        h,s,v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        
        v = cv2.equalizeHist(v)
        
        filtered_image = cv2.merge((h,s,v))
        filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_HSV2BGR)
    
    return filtered_image.reshape(image.shape)


source = os.getcwd()+f'/adv_image/{FLAGS.dataset}/'

enhances = ['Bilateral','Gaussian_Blur','Histogram']
attacks =  os.listdir(source)

for folder in enhances:
    for attack in attacks:
        if not(os.path.isdir(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/{folder}/{attack}')):
            os.makedirs(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/{folder}/{attack}')   
    
for folder in tqdm(attacks):
    
    for file in os.listdir(source+folder):    
        
        img = cv2.imread(f'{source}{folder}/{file}')
        
            
        if FLAGS.dataset == 'MNIST' :
            img = np.mean(img, axis=2)
            img = np.expand_dims(img, axis=2)
            img = cv2.resize(img,(28,28))
            
        if FLAGS.dataset == 'CIFAR10' :
            img = cv2.resize(img,(32,32))
            
        
        cv2.imwrite(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/Original/{folder}/{file}',img)
        
        img = img.astype(np.uint8)
            
        bi_image = bilateral(img)
        gb_image = gaussian_blur(img)       
        hist_image = histogram(img)
        
        cv2.imwrite(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/Bilateral/{folder}/{file[:-4]}_bi.png', bi_image)
        cv2.imwrite(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/Gaussian_Blur/{folder}/{file[:-4]}_gb.png', gb_image)        
        cv2.imwrite(f'{os.getcwd()}/adv_filter/{FLAGS.dataset}/Histogram/{folder}/{file[:-4]}_hist.png', hist_image)
                          
print('--------------------- Done ----------------------------')
