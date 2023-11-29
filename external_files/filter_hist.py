from tqdm import tqdm
import numpy as np
import os

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
          
          # Calculate the histogram of the image
          hist, bins = np.histogram(adversarial_images.flatten(), bins=256, range=[0, 256])

          # Calculate the cumulative distribution function (CDF)
          cdf = hist.cumsum()

          # Normalize the CDF
          cdf_normalized = cdf * hist.max() / cdf.max()

          # Perform histogram equalization
          filtered_images = np.interp(adversarial_images.flatten(), bins[:-1], cdf_normalized).reshape(adversarial_images.shape)        

          if not(os.path.isdir(f'{directory}_hist/{sub}')):
            os.makedirs(f'{directory}_hist/{sub}')
          
          # Save the filtered images with the same name
                
          filtered_file_path =  f'{directory}_hist/{sub}/{filename}_hist.npy'
          np.save(filtered_file_path, filtered_images)
          
      if filename.endswith('origin.npy'):
          file_path = os.path.join(f'{directory}/{sub}', filename)

          # Load the .npy file
          adversarial_images = np.load(file_path)        

          if not(os.path.isdir(f'{directory}_hist/{sub}')):
            os.makedirs(f'{directory}_hist/{sub}')
          
          # Save the filtered images with the same name
                
          filtered_file_path =  f'{directory}_hist/{sub}/{filename}'
          np.save(filtered_file_path, adversarial_images)




