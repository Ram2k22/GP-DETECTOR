import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

for i in range(5):
    orig = np.load(str(i)+"_origin.npy").reshape(28,28,1)
    adv = np.load(str(i)+"_adv.npy").reshape(28,28,1)
    kmodel = load_model(r'C:\Users\tutik\Desktop\ML(08-07)\GP-based-Adversarial-Detection\saved_models\MNIST_model.h5')
    origin_label=np.argmax(kmodel.predict(orig[np.newaxis,:,:,:]/255), axis=1)
    adv_label = np.argmax(kmodel.predict(adv[np.newaxis,:,:,:]/255), axis = 1)

    print(origin_label)
    print(adv_label)

    # Saving the Image
    orig_image = np.squeeze(orig)
    adv_image = np.squeeze(adv)
    plt.imshow(orig_image)
    plt.imshow(adv_image)
    plt.axis('off')
    plt.savefig('original_'+str(i)+'.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('adversarial'+str(i)+'.png', bbox_inches='tight', pad_inches=0)




