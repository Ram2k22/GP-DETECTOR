"""CIFAR10 small images classification dataset.
"""

from tqdm import tqdm
import numpy as np
import os
import cv2

class braintumor:
    def load_images(self,type):
        dirname = f'{os.getcwd()}/Dataset/{type}'
        x = []
        y = []
        for i in tqdm(os.listdir(dirname),desc=f"Loading {type[:-3]} data"):
            for j in os.listdir(f"{dirname}/{i}"):
                data = cv2.imread(f"{dirname}/{i}/{j}")
                data = cv2.resize(data,(30,30))
                x.append(data)
                y.append(int(i))
        
        return np.array(x),np.array(y)

        
    def load_data(self):
        """Loads CIFAR10 dataset.

        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        x_train , y_train = self.load_images('Training')
        x_test , y_test = self.load_images('Testing')
                
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        
        
        print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)
                
        return (x_train, y_train), (x_test, y_test)

# braintumor().load_data()