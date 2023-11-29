import os
import warnings

import cv2
import keras
import numpy as np

from tqdm import tqdm
import tensorflow as tf


from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model

warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
keras.backend.set_learning_phase(0)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'CIFAR10', 'Training dataset name.')


def evalute_save(adv_model):
    for i in tqdm(range(len(files))):
        
        test_input = x_test[i][np.newaxis,:,:,:]
        y_input = [origin_pred[i]]
        
        if attack == 'BIM':
            y_input = keras.utils.to_categorical(y_input, 10)
            
        adv_x_eval = adv_model.eval(session=sess, feed_dict={ x: test_input /255, y: y_input})

        file = adv_x_eval * 255

        label = np.argmax(main_model.predict(adv_x_eval/255),axis = 1)

        if not(os.path.isdir(path+f"{attack}")):
            os.makedirs(path+f"{attack}")
            
        adv_img = file[0]
        adv_img = adv_img.astype(np.uint8)
        # adv_img = cv2.cvtColor(adv_img,cv2.COLOR_BGR2RGB)
        # adv_img = cv2.resize(adv_img,(500,500))
        
        cv2.imwrite((path+f"{attack}/{files[i][:-4]}_{attack}_({origin_pred[0]})({label[0]}).png"),adv_img)
    
    
    
    
    
if __name__ == "__main__":   

    if FLAGS.dataset == 'MNIST':
        model_file_name = 'MNIST_model.h5'
        img_rows, img_cols, n_channels = 28, 28, 1
        n_classes = 1
        
    elif FLAGS.dataset == 'CIFAR10':
        model_file_name = 'cifar10_ResNet32v1_model.h5'
        img_rows, img_cols, n_channels = 32, 32, 3
        n_classes = 1

    elif FLAGS.dataset == 'BT':
        model_file_name = 'braintumor_model.h5'
        img_rows, img_cols, n_channels = 30, 30, 3
        n_classes = 1
        

    print("\n load model...")
    main_model = load_model(f'saved_models/{model_file_name}')
    wrapped_model = KerasModelWrapper(main_model)
    print("model loaded.")

    source_path = os.getcwd()+f"/adv_image/images/{FLAGS.dataset}/"
    path = os.getcwd()+f"/adv_image/{FLAGS.dataset}/"
    
    files = os.listdir(source_path)
    
    x_test = np.empty(( len(files), img_rows, img_cols, n_channels), dtype= np.uint8)
        
    for i in tqdm(range(len(files))):
        
        img = cv2.imread(source_path+files[i])
        
        img = img.astype(np.uint8)
        
        file_array = cv2.resize(img,(img_rows, img_cols))
        
        if FLAGS.dataset == 'MNIST' :
            file_array = np.mean(file_array, axis=2)
            file_array = np.expand_dims(file_array, axis=2)
        
        if not(os.path.isdir(path+f"Original/")):
            os.makedirs(path+f"Original/")
            
        
        cv2.imwrite((path+f"Original/{files[i][:-4]}_original.png"),file_array)
        
        file_array = np.expand_dims(file_array,axis=0)
        x_test[i] = file_array 



    origin_pred = np.argmax(main_model.predict(x_test/255), axis=1)
    origin_pred = origin_pred.reshape(origin_pred.shape[0], 1)
    
        
    
        
# -----------------------   FGSM ATTACK   ------------------------------
    x = tf.placeholder(tf.float32, shape=(None,  img_rows,  img_cols,  n_channels))
    y = tf.placeholder(tf.float32, shape=(None,  n_classes))
    
    attack = 'FGSM'
    
    FGSM_attack = FastGradientMethod(wrapped_model,sess)

    if FLAGS.dataset == 'MNIST':
        params = {'eps': 0.1, 'clip_min': 0., 'clip_max': 1.}
        
    if FLAGS.dataset == 'CIFAR10':
        params = {'eps': 1/255, 'clip_min': 0., 'clip_max': 1.}
        
    adv_x = FGSM_attack.generate( x, **params)
    
    evalute_save(adv_x)        
                    
    
    # -----------------------   JSMA ATTACK   ------------------------------
    
    x = tf.placeholder(tf.float32, shape=(None,  img_rows,  img_cols,  n_channels))
    y = tf.placeholder(tf.float32, shape=(None,  n_classes))
    
    attack = 'JSMA'
    
    jsma_attack = SaliencyMapMethod(wrapped_model, sess=sess)
    params = {'clip_min': 0., 'clip_max': 1.}
    adv_x = jsma_attack.generate(x, **params)
    
    evalute_save(adv_x)
    
    # -----------------------   DeepFool ATTACK   ------------------------------
    
    x = tf.placeholder(tf.float32, shape=(None,  img_rows,  img_cols,  n_channels))
    y = tf.placeholder(tf.float32, shape=(None,  n_classes))
    
    attack = 'DeepFool'
    deepfool_attack = DeepFool(wrapped_model, sess=sess)
    params = {'nb_candidate': 10, 'max_iter': 100, 'clip_min': 0., 'clip_max': 1., 'verbose':False}
    adv_x = deepfool_attack.generate(x, **params)
    
    evalute_save(adv_x)
    
    
    # -----------------------   BIM ATTACK   ------------------------------
    
    x = tf.placeholder(tf.float32, shape=(None,  img_rows,  img_cols,  n_channels))
    y = tf.placeholder(tf.float32, shape=(None,  n_classes))
       
    attack = 'BIM'
    
    y = tf.placeholder(tf.float32, shape=(None,  10))
    
    bim_attack = BasicIterativeMethod(wrapped_model, sess=sess)
    
    if FLAGS.dataset == 'MNIST':
        params = {'eps': 0.1, 'eps_iter': 0.1/10, 'nb_iter': 10,'y': y,
                  'clip_min': 0., 'clip_max': 1.}
    if FLAGS.dataset == 'CIFAR10':
        params = {'eps':   1/255, 'eps_iter': 1/255/10, 'nb_iter': 10, 'y': y,
                  'clip_min': 0., 'clip_max': 1.}
            
    adv_x = bim_attack.generate(x, **params)
    adv_x = tf.stop_gradient(adv_x)
    
    evalute_save(adv_x)
    
    # ----------------  end ------------------------
    

