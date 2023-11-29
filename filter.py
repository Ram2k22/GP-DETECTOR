from tqdm import tqdm

import tensorflow as tf
import numpy as np
import cv2
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name.')
flags.DEFINE_string('enhance', 'bi', 'Training dataset name.')




def bilateral(image):
    
    # Apply bilateral filtering
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=80, sigmaSpace=80)

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



# ------------------------------ Code Begins -------------------------------------------------------

source = os.getcwd()+f'/adv_data/{FLAGS.dataset}'

attacks = ['BIM','DeepFool','FGSM','JSMA']
    
    
for i in tqdm(range(len(attacks))):
    print(f'\n{FLAGS.enhance} on {attacks[i]} started')
    
    folders = os.listdir(f'{source}/{attacks[i]}')
    
    for sub in tqdm(folders): 
             
        if not(os.path.isdir(f'{source}/{attacks[i]}_{FLAGS.enhance}/{sub}')):
            os.makedirs(f'{source}/{attacks[i]}_{FLAGS.enhance}/{sub}')
            
        for file in os.listdir(f'{source}/{attacks[i]}/{sub}'):
            if file.endswith('adv.npy'):
                
                img = np.load(f'{source}/{attacks[i]}/{sub}/{file}')
                
                img = img.astype(np.uint8)
            
                
                if FLAGS.enhance == 'bi':                    
                    filter_image = bilateral(img)
                      
                elif FLAGS.enhance == 'gb':                    
                    filter_image = gaussian_blur(img)
                        
                elif FLAGS.enhance == 'hist':                    
                    filter_image = histogram(img)
                    
                elif FLAGS.enhance == 'sharpen':
                    filter_image = sharpen(img)
                
                elif FLAGS.enhance == 'mb':
                    filter_image = median_blur(img)
                    
                elif FLAGS.enhance == 'ahe':
                    filter_image = adaptive_histogram_equalization(img)
                           
                np.save(f'{source}/{attacks[i]}_{FLAGS.enhance}/{sub}/{file[:-4]}_{FLAGS.enhance}.npy',filter_image)
            
            elif file.endswith('origin.npy'):
                
                img = np.load(f'{source}/{attacks[i]}/{sub}/{file}')
                
                np.save(f'{source}/{attacks[i]}_{FLAGS.enhance}/{sub}/{file}',img)
                
                    
print('--------------------- Done ----------------------------')



    



