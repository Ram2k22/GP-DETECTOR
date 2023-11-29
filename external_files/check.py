from PIL import Image
import numpy as np
from keras.models import load_model


 
 
# load the image and convert into
# numpy array
img = Image.open('10.jpg')

# numpydata = np.load(r"C:\Users\HP\Downloads\ML\adv\adv_data\MNIST\DeepFool\1\0_origin.npy")
 
# asarray() class is used to convert
# PIL images into NumPy arrays
numpydata = np.asarray(img)
# print(numpydata)
numpydata=numpydata.reshape([28,28,1])
print(numpydata.shape)
kmodel = load_model(r"saved_models\MNIST_model.h5")

label = np.argmax(kmodel.predict(numpydata[np.newaxis,:,:,:]/255),axis = 1)




print("label=",label)