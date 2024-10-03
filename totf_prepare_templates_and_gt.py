import argparse
import os
import json
import numpy as np
import torch
from src.pose_extractor import PoseViTExtractor
from tools.ply_file_to_3d_coord_model import convert_unique
from rendering.renderer_xyz import Renderer
from rendering.model import Model3D
from tqdm import tqdm
import cv2
from PIL import Image
from pose_utils import vis_utils
from pose_utils import img_utils
from rendering.utils import get_rendering, get_sympose

from totf_StableDino import stabledino


import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA

import sys

import datetime


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    
    parser.add_argument('arg1', type=int)
    parser.add_argument('arg2', type=int)
    parser.add_argument('arg3', type=int)
    parser.add_argument('--config_file', default="./dino_pose_configs/template_gt_preparation_configs/cfg_template_gt_generation.json")

    args = parser.parse_args()

    with open(os.path.join(args.config_file),'r') as f:
        config = json.load(f)

    with open(os.path.join(config['path_models_info_json']), 'r') as f:
        models_info = json.load(f)

    obj_poses = np.load(config['path_template_poses'])

    # Creating the output folder for the cropped templates and descriptors
    if not os.path.exists(config['path_output_templates_and_descs_folder']):
        os.makedirs(config['path_output_templates_and_descs_folder'])

    # Creating the models_xyz folder
    if not os.path.exists(config['path_output_models_xyz']):
        os.makedirs(config['path_output_models_xyz'])

    # Preparing the object models in xyz format:
    print("Loading and preparing the object meshes:")
    norm_factors = {}
    for obj_model_name in tqdm(os.listdir(config['path_object_models_folder'])):
        if obj_model_name.endswith(".ply"):
            obj_id = int(obj_model_name.split("_")[-1].split(".ply")[0])
            input_model_path = os.path.join(config['path_object_models_folder'], obj_model_name)
            output_model_path = os.path.join(config['path_output_models_xyz'], obj_model_name)
            # if not os.path.exists(output_model_path):
            x_abs,y_abs,z_abs,x_ct,y_ct,z_ct = convert_unique(input_model_path, output_model_path)

            norm_factors[obj_id] = {'x_scale':float(x_abs),
                                    'y_scale':float(y_abs),
                                    'z_scale':float(z_abs),
                                    'x_ct':float(x_ct),
                                    'y_ct':float(y_ct),
                                    'z_ct':float(z_ct)}

    with open(os.path.join(config['path_output_models_xyz'],"norm_factor.json"),"w") as f:
        json.dump(norm_factors,f)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    external_size = args.arg1 #sd size 
    external_timestep = args.arg2 #sd timestep 
    external_raw = False if(args.arg3 == 0) else True #sd size 

    start_model = datetime.datetime.now()

    pose_estimator = stabledino(use_sd=1,use_dino=1,sd_size=external_size,sd_timestep=external_timestep,sd_raw=external_raw)
    
    end_model = datetime.datetime.now()
    print(f"Model time: {end_model - start_model}") 
    extractor = pose_estimator.extractor_dino

    cam_K = np.array(config['cam_K']).reshape((3,3))

    ren = Renderer((config['template_resolution'][0], config['template_resolution'][1]), cam_K)

    template_labels_gt = dict()
    
    # Starte den hochauflösenden Timer
    start_all = datetime.datetime.now()


    with torch.no_grad():

        for template_name in tqdm(os.listdir(config['path_templates_folder'])):

            path_template_folder = os.path.join(config['path_templates_folder'], template_name)

            if os.path.isdir(path_template_folder) and template_name != "models" and template_name != "models_proc":

                path_to_template_desc = os.path.join(config['path_output_templates_and_descs_folder'],
                                                     template_name)

                if not os.path.exists(path_to_template_desc):
                    os.makedirs(path_to_template_desc)

                obj_id = template_name.split("_")[-1]

                model_info = models_info[str(obj_id)]

                obj_model = Model3D()
                model_path = os.path.join(config['path_output_models_xyz'], f"obj_{int(obj_id):06d}.ply")

                # Some objects are scaled inconsistently within the dataset, these exceptions are handled here:
                obj_scale = config['obj_models_scale']
                obj_model.load(model_path, scale=obj_scale)
                print(model_path)

                files = os.listdir(path_template_folder)
                filtered_files = list(filter(lambda x: not x.startswith('mask_'), files))
                #filtered_files.sort(key=lambda x: os.path.getmtime(os.path.join(path_template_folder,x)))
                filtered_files.sort(key=lambda x: x)
                print(filtered_files)
                tmp_list = []

                for i, file in enumerate(filtered_files):

                    
                    # Beende den Timer
                    start_single = datetime.datetime.now()

                    print("File: ",i)
                    # Preparing mask and bounding box [x,y,w,h]
                    mask_path = os.path.join(path_template_folder, f"mask_{file}")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    crop_size = max(w,h)

                    # Preparing cropped image and desc
                    img = cv2.imread(os.path.join(path_template_folder, file))
                    print(path_template_folder, file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_crop, crop_x, crop_y = img_utils.make_quadratic_crop(img, [x, y, w, h])

                    img_crop_sd = copy.deepcopy(img_crop)
                    
                    img_prep, img_crop, _ = extractor.preprocess(Image.fromarray(img_crop), load_size=224)
                    desc = extractor.extract_descriptors(img_prep.to(device), layer=11, facet='key', bin=False, include_cls=True)                    
                    desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()

                    desc_sd,sd_image = pose_estimator.get_SD_features(Image.fromarray(img_crop_sd))

                    

                    R = obj_poses[i][:3,:3]
                    t = obj_poses[i].T[-1,:3]
                    sym_continues = [0,0,0,0,0,0]
                    keys = model_info.keys()

                    if('symmetries_continuous' in keys):
                        sym_continues[:3] = model_info['symmetries_continuous'][0]['axis']
                        sym_continues[3:] = model_info['symmetries_continuous'][0]['offset']
                    
                    rot_pose, rotation_lock = get_sympose(R, sym_continues)                 
                    
                    img_uv, depth_rend, bbox_template = get_rendering(obj_model, rot_pose, t/1000., ren)

                    img_uv = img_uv.astype(np.uint8)

                    img_uv,_,_ = img_utils.make_quadratic_crop(img_uv, [crop_y, crop_x, crop_size, crop_size])
                    
                    # ACHTUNG: START - NEU
                    img_uv2, depth_rend2, bbox_template2 = get_rendering(obj_model, rot_pose, t/1000., ren)
                    img_uv2 = img_uv2.astype(np.uint8)     

                    img_uv2_mask = cv2.cvtColor(img_uv2, cv2.COLOR_BGR2GRAY)
                    contours_uv2, _ = cv2.findContours(img_uv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x_uv2, y_uv2, w_uv2, h_uv2 = cv2.boundingRect(contours_uv2[0])    
                    crop_size_uv2 = max(w_uv2,h_uv2)   

                    if(crop_size_uv2 > w_uv2):x_uv2=x_uv2-(crop_size_uv2-w_uv2)/2  
                    if(crop_size_uv2 > h_uv2):y_uv2=y_uv2-(crop_size_uv2-h_uv2)/2    

                    img_crop_uv2,_,_ = img_utils.make_quadratic_crop(img_uv2, [x_uv2, y_uv2, crop_size_uv2, crop_size_uv2])
                    img_uv = cv2.resize(img_crop_uv2, (pose_estimator.real_size, pose_estimator.real_size))
                    
                    
                    
                    if(pose_estimator.raw): #s2,s3,s4,s5
                        
                        desc_sd_s2 = desc_sd['s2'].detach().cpu().numpy()
                        desc_sd_s3 = desc_sd['s3'].detach().cpu().numpy()
                        desc_sd_s4 = desc_sd['s4'].detach().cpu().numpy()
                        desc_sd_s5 = desc_sd['s5'].detach().cpu().numpy()
                        
                        
                        # Storing template information:
                        tmp_dict = {"img_id": str(i),
                                    "img_name":os.path.join(os.path.join(path_template_folder,file)),
                                    "mask_name":os.path.join(os.path.join(path_template_folder,f"mask_{file}")),
                                    "obj_id": str(obj_id),
                                    "bbox_obj": [x,y,w,h],
                                    "cam_R_m2c": R.tolist(),
                                    "cam_t_m2c": t.tolist(),
                                    "model_path": os.path.join(config['path_object_models_folder'], f"obj_{int(obj_id):06d}.ply"),
                                    "model_info": models_info[str(obj_id)],
                                    "cam_K": cam_K.tolist(),
                                    "img_crop": os.path.join(path_to_template_desc, file),
                                    "img_desc": os.path.join(path_to_template_desc, f"{file.split('.')[0]}.npy"),
                                    "img_desc_sd_s2": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_sd_s2.npy"),
                                    "img_desc_sd_s3": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_sd_s3.npy"),
                                    "img_desc_sd_s4": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_sd_s4.npy"),
                                    "img_desc_sd_s5": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_sd_s5.npy"),
                                    #"img_desc_sd2": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_sd2.npy"),
                                    "uv_crop": os.path.join(path_to_template_desc, f"{file.split('.')[0]}_uv.npy")

                                    }
                        
                        # Beende den Timer
                        end_single = datetime.datetime.now()
                        print(f"Signle time: {end_single - start_single}")   

                        tmp_list.append(tmp_dict)

                        # Saving all template crops and descriptors:
                        np.save(tmp_dict['uv_crop'], img_uv)
                        np.save(tmp_dict['img_desc'], desc)
                        np.save(tmp_dict['img_desc_sd_s2'], desc_sd_s2)
                        np.save(tmp_dict['img_desc_sd_s3'], desc_sd_s3)
                        np.save(tmp_dict['img_desc_sd_s4'], desc_sd_s4)
                        np.save(tmp_dict['img_desc_sd_s5'], desc_sd_s5)
                        sd_image.save(tmp_dict['img_crop'])

                    # Angenommen, x ist Ihr Numpy-Array der Größe (1, 384, 32, 32)
                    
                    
                template_labels_gt[str(obj_id)] = tmp_list
      
    # Beende den Timer
    end_all = datetime.datetime.now()

    # Berechne die verstrichene Zeit und zeige sie an
    print(f"All time: {end_all - start_all}")          

    with open(config['output_template_gt_file'], 'w') as f:
        json.dump(template_labels_gt, f)












