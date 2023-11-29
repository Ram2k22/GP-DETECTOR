from tqdm import tqdm
import numpy as np
import cv2
import os





# Apply bilateral filtering as an enhancement technique
def bilateral_filtering(image):
    # Convert image to grayscale if necessary
    print(image.shape)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    
    # Apply bilateral filtering
    enhanced_image = cv2.bilateralFilter(image, 9, 75, 75)  # Adjust the parameters as needed
    
    return enhanced_image
  
  
  

# Directory containing the .npy files

# for CIFAR10
# main = os.getcwd()+'/adv_data/CIFAR10/'
# li = ['BIM_e1','FGSM_e1','DeepFool','JSMA']

# for MNIST
main = os.getcwd()+'/adv_data/MNIST/'
li = ['BIM','FGSM_e0.1','DeepFool','JSMA','BIM_e0.30','FGSM_e0.30']



for i in li:
  directory = main+i

  # Iterate over the .npy files in the directory
  for sub in tqdm(os.listdir(directory)):
    for filename in os.listdir(f'{directory}/{sub}'):
      if filename.endswith('adv.npy'):
          file_path = os.path.join(f'{directory}/{sub}', filename)

          # Load the .npy file
          adversarial_images = np.load(file_path)
          print(adversarial_images.shape)

          # Apply the filter
          
          # Apply enhancement technique
          filtered_images = np.zeros_like(adversarial_images)
          for i in range(adversarial_images.shape[0]):
            filtered_images[i] = bilateral_filtering(adversarial_images[i])        

          if not(os.path.isdir(f'{directory}_bi/{sub}')):
            os.makedirs(f'{directory}_bi/{sub}')
                   
          # Save the filtered images with the same name     
          filtered_file_path =  f'{directory}_bi/{sub}/{filename}_bi.npy'
          np.save(filtered_file_path, filtered_images)
          
      if filename.endswith('origin.npy'):
          file_path = os.path.join(f'{directory}/{sub}', filename)

          # Load the .npy file
          adversarial_images = np.load(file_path)        

          if not(os.path.isdir(f'{directory}_bi/{sub}')):
            os.makedirs(f'{directory}_bi/{sub}')
          
          # Save the filtered images with the same name
                
          filtered_file_path =  f'{directory}_bi/{sub}/{filename}'
          np.save(filtered_file_path, adversarial_images)
          
          