import os
import cv2
import GPy
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'CIFAR10', 'Training dataset name.')
flags.DEFINE_string('attack', 'DeepFool', 'Training dataset name.')
flags.DEFINE_string('enhance', '', 'Training dataset name.')
flags.DEFINE_string('images', 'adv', 'Training dataset name.')


if FLAGS.dataset == 'MNIST':
    kmodel = load_model('saved_models/MNIST_model.h5')
elif FLAGS.dataset == 'CIFAR10':
    kmodel = load_model('saved_models/cifar10_ResNet32v1_model.h5')
elif FLAGS.dataset == 'BT':
    kmodel = load_model('saved_models/braintumor_model.h5')
    

# --------------------- Starting ------------------------

# for adversial images
if FLAGS.images == 'adv':
    path = os.getcwd() + f"/adv_image/{FLAGS.dataset}/"
    files = ['Original', 'FGSM', 'BIM', 'JSMA', 'DeepFool']

    m = GPy.models.GPClassification.load_model(f"saved_models/GP/{FLAGS.dataset}_{FLAGS.attack}.gp.zip")

# for Enhancing images
elif FLAGS.images == 'en':
    path = os.getcwd() + f"/adv_image/filter/{FLAGS.dataset}/"
    files = ['Original','bi','gb','hist','ahe','mb','sharpen']

    m = GPy.models.GPClassification.load_model(f"saved_models/GP/{FLAGS.dataset}_{FLAGS.attack}_{FLAGS.enhance}.gp.zip")


rows = len(files)
cols = len(os.listdir(path+files[0]))

fig, axes = plt.subplots(rows, cols,figsize=(50,50))

for i,folder in tqdm(enumerate(files)):
    
    axes[i, 0].text(-0.5, 0.5, files[i], transform=axes[i, 0].transAxes,
                    va='center', ha='center', fontsize=12)
        
    for j,file in enumerate(os.listdir(path+folder)):

        image = cv2.imread(path + folder +'/'+ file)

        if FLAGS.dataset == 'MNIST':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape(28,28,1)
            # image = np.mean(image, axis=2)
            # image = np.expand_dims(image, axis=2)
            
            
        image_buffer = image[np.newaxis, :, :, :]
        img = kmodel.predict(image_buffer/255)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = m.predict(img)
        
        # image = cv2.resize(image,(300,300))
        axes[i,j].imshow(image)
        axes[i,j].axis('off')
        
        if label[0][0][0] >= 0.5 :           
            axes[i,j].set_title(f"Fake:{np.argmax(img,axis = 1)}")
        else:
            axes[i,j].set_title(f"Real:{np.argmax(img,axis = 1)}")
            
plt.subplots_adjust(hspace=0.3) 
plt.suptitle('Adversarial Images Detection on '+FLAGS.dataset+' Dataset',fontweight = 'bold')
plt.show()