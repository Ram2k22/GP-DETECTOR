import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

dataset = "BT"


source_path = os.getcwd()+"/adv_data/"+dataset
dest_path = os.getcwd()+"/adv_data/Images/"+dataset


attacks = os.listdir(source_path)
for attack in tqdm(attacks):
    for sub in os.listdir(source_path+"/"+attack):
        for file in os.listdir(source_path+"/"+attack+"/"+sub):
            
            
            # fig = plt.figure()
            # A = np.load(source_path+"/"+attack+"/"+sub+"/"+file)

            # plt.imshow(A, cmap='gray')
                        
            # if not(os.path.isdir(f"{dest_path}/{attack}/{sub}")):
            #     os.makedirs(f"{dest_path}/{attack}/{sub}")
            # fig.savefig(f"{dest_path}/{attack}/{sub}/{file}.png")



            # using cv2 module
            A = np.load(source_path+"/"+attack+"/"+sub+"/"+file)
            A = A.astype(np.uint8)
            
            if not(os.path.isdir(f"{dest_path}/{attack}/{sub}")):
                os.makedirs(f"{dest_path}/{attack}/{sub}")
                
            A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{dest_path}/{attack}/{sub}/{file[:-4]}.png",A)
            
            
            # Using PIL 
            
            # A = np.load(source_path+"/"+attack+"/"+sub+"/"+file)
            # if not(os.path.isdir(f"{dest_path}/{attack}/{sub}")):
            #     os.makedirs(f"{dest_path}/{attack}/{sub}")
            
            # A = A.astype(np.uint8)
            # A = Image.fromarray(A)
            # A.save(f"{dest_path}/{attack}/{sub}/{file[:-4]}.png")
