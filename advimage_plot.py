import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import cv2
import os


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'MNIST', 'Training dataset name.')

dataset = FLAGS.dataset
    
path = os.getcwd() + f"/adv_image/{dataset}/"

# files = os.listdir(path)
files = ['Original','FGSM','BIM','JSMA','DeepFool']

rows = len(files)
cols = len(os.listdir(path+files[0]))

fig, axes = plt.subplots(rows, cols, figsize=(10,100))


for i,folder in tqdm(enumerate(files)):
    path2 = path + folder
    
    axes[i, 0].text(-0.5, 0.5, files[i], transform=axes[i, 0].transAxes,
                    va='center', ha='center', fontsize=12)
    
    
    for j,file in enumerate(os.listdir(path2)):
        image = cv2.imread(path2 + f"/{file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[i,j].imshow(image)
        axes[i,j].axis('off')

plt.suptitle('Adversarial Attacks on '+FLAGS.dataset + ' Dataset', fontweight='bold')

plt.show()