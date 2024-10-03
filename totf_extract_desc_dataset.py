
import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from pose_utils import img_utils

from totf_StableDino import stabledino


import matplotlib.pyplot as plt
import copy

import sys

savepath = "/home/bernd/Workspace/StableDinoShot/desc_dataset/lmo/512/"

# Loading the config file:
with open(os.path.join("/home/bernd/Workspace/StableDinoShot/dino_pose_configs/cfg_lmo_inference_bop.json"), "r") as f:
    config = json.load(f)

# Loading a ground truth file to access segmentation masks to test zs6d:
with open(os.path.join(config['gt_path']), 'r') as f:
    data_gt = json.load(f)   

# For the loop through the objects
with open(os.path.join(config['templates_gt_path']), 'r') as f:
    temp_data = json.load(f)
    
template_keys = list(temp_data.keys())
print(template_keys)

_, _, last_part = config['templates_gt_path'].rpartition("_")
template_count,_,_ = last_part.rpartition(".")
template_data = np.load('/home/bernd/Workspace/DATA/lmo/templates/'+template_count+'/obj_poses.npy')

temp_angles =[]
for data_temp in template_data:         
    rot_template = data_temp[:3, :3]    
    temp_angles.append(rot_template)

external_size = int(sys.argv[1]) #sd size
external_timestep = int(sys.argv[2]) #sd size

external_raw = False if(int(sys.argv[3]) == 0) else True #raw

pose_estimator = stabledino(use_sd=1,use_dino=1,sd_size=external_size,sd_timestep=external_timestep,sd_raw=external_raw)

#pose_estimator = stabledino(use_sd=1,use_dino=1,sd_size=external_size,sd_timestep=external_timestep,sd_raw=external_raw)
extractor = pose_estimator.extractor_dino


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#img_id = '3'
# img_id = '8'
data =[]
Errorcount_mask = 0
Errorcount_Pose = 0
temp_count_dino = 0
temp_count_sd2 = 0
temp_count_sd = 0
for key in template_keys:
    #if key != '8' and key != '1':continue
    pose_estimator.set_paths_and_load_data(config['templates_gt_path'], config['norm_factor_path'],key)
    for img_id in data_gt:
        #if(img_id != '3'):continue
        for i in range(len(data_gt[img_id])):
            obj_number = i
            obj_id = data_gt[img_id][obj_number]['obj_id']
            if(obj_id != int(key)):continue
            print(f"Image:\t{img_id}\tObject:\t{obj_id}\tProcessing:\t{i}/{len(data_gt[img_id])-1}")
            cam_K = np.array(data_gt[img_id][obj_number]['cam_K']).reshape((3,3))
            #bbox = data_gt[img_id][obj_number]['bbox_visib']

            img_path = os.path.join(config['dataset_path'], data_gt[img_id][obj_number]['img_name'].split("./")[-1])
            img = Image.open(img_path)
            img_sd = copy.deepcopy(img)

            try:
                mask = data_gt[img_id][obj_number]['mask_sam']
                mask = img_utils.rle_to_mask(mask)
                mask = mask.astype(np.uint8)
            except:
                print("Error - no Mask avaiable")
                Errorcount_mask +=1
                continue

            mask_sd = copy.deepcopy(mask)

            bbox = img_utils.get_bounding_box_from_mask(mask)
            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = extractor.preprocess(img_crop, load_size=224)

            bbox = img_utils.get_bounding_box_from_mask(mask_sd)
            img_crop_sd, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img_sd), bbox)

            mask_crop_sd, _, _ = img_utils.make_quadratic_crop(mask_sd, bbox)
            img_crop_sd = cv2.bitwise_and(img_crop_sd, img_crop_sd, mask=mask_crop_sd)
            img_crop_sd = Image.fromarray(img_crop_sd)
            _, img_sd, _ = extractor.preprocess(img_crop_sd, load_size=pose_estimator.sd_size)


            img_crop_sd = copy.deepcopy(img_crop)



            with torch.no_grad():
                desc_dino = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)
                desc_dino = desc_dino.squeeze(0).squeeze(0).detach().cpu()

                desc_sd,sd_image = pose_estimator.get_SD_features(img_crop_sd)
                
                save = savepath+str(img_id)+"_"+str(key)

                if(pose_estimator.raw): #s2,s3,s4,s5
                        
                    desc_sd_s2 = desc_sd['s2'].detach().cpu().numpy()
                    desc_sd_s3 = desc_sd['s3'].detach().cpu().numpy()
                    desc_sd_s4 = desc_sd['s4'].detach().cpu().numpy()
                    desc_sd_s5 = desc_sd['s5'].detach().cpu().numpy()            

                   
                    # Saving all template crops and descriptors:

                    np.save((save+"_dino.npy"), desc_dino)
                    np.save((save+"_sd_s2.npy"), desc_sd_s2)
                    np.save((save+"_sd_s3.npy"), desc_sd_s3)
                    np.save((save+"_sd_s4.npy"), desc_sd_s4)
                    np.save((save+"_sd_s5.npy"), desc_sd_s5)

                    
                
                

                sd_image.save((save+".png"))



           
            
            


            

            

