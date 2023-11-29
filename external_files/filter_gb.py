from tqdm import tqdm
import numpy as np
import cv2
import os

kernel_size = 3
sigma = 1.0
# Directory containing the .npy files

# for CIFAR10
# main = os.getcwd()+'/adv_data/CIFAR10/'
# li = ['BIM_e1','FGSM_e1','DeepFool','JSMA']

# for MNIST
main = os.getcwd()+'/adv_data/MNIST/'
li = ['BIM_e0.1','FGSM_e0.1','DeepFool','JSMA','BIM_e0.30','FGSM_e0.30']




for i in li:
  directory = main+i

  # Iterate over the .npy files in the directory
  for sub in tqdm(os.listdir(directory)):
    for filename in os.listdir(f'{directory}/{sub}'):
      if filename.endswith('adv.npy'):
          file_path = os.path.join(f'{directory}/{sub}', filename)

          # Load the .npy file
          adversarial_images = np.load(file_path)

          # Apply the filter
          
          filtered_images = cv2.GaussianBlur(adversarial_images, (kernel_size, kernel_size), sigma)      

          if not(os.path.isdir(f'{directory}_gb/{sub}')):
            os.makedirs(f'{directory}_gb/{sub}')
          
          # Save the filtered images with the same name
                
          filtered_file_path =  f'{directory}_gb/{sub}/{filename}_gb.npy'
          np.save(filtered_file_path, filtered_images.reshape(adversarial_images.shape))
          
      if filename.endswith('origin.npy'):
          file_path = os.path.join(f'{directory}/{sub}', filename)

          # Load the .npy file
          adversarial_images = np.load(file_path)      

          if not(os.path.isdir(f'{directory}_gb/{sub}')):
            os.makedirs(f'{directory}_gb/{sub}')
          
          # Save the filtered images with the same name
                
          filtered_file_path =  f'{directory}_gb/{sub}/{filename}'
          np.save(filtered_file_path, adversarial_images)




