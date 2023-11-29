import os
import shutil
from tqdm import tqdm


main_path = os.getcwd()+"/adv_data/MNIST"
attacks = os.listdir(main_path)
for attack in tqdm(attacks):
    for sub in os.listdir(main_path+"/"+attack):
        for file in os.listdir(main_path+"/"+attack+"/"+sub):
        
            if file=='original':
                for i in os.listdir(main_path+"/"+attack+"/"+sub+"/original"):
                    source = main_path+"/"+attack+"/"+sub+"/original/"+i
                    destination = main_path+"/"+attack+"/"+sub+"/"+i
                    os.rename(source, destination)
                    
                
                shutil.rmtree(f"{main_path}/{attack}/{sub}/original")

  