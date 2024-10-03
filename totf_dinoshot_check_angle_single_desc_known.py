
from totf_StableDino import stabledino
import os
import json

import cv2
from PIL import Image
import pose_utils.img_utils as img_utils
import pose_utils.vis_utils as vis_utils
import numpy as np
import time
import matplotlib.pyplot as plt

from pose_utils import eval_utils


import torch
#Data
import pandas as pd
from datetime import date
from datetime import datetime
import copy

import sys

from sklearn.decomposition import PCA

from sd_utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace

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

# Instantiating the pose estimator:
# This involves handing over the path to the templates_gt file and the corresponding object norm_factors.
external_size = int(sys.argv[1]) #sd size
external_timestep = int(sys.argv[2]) #sd size

external_raw = False if(int(sys.argv[3]) == 0) else True #sd size 

external_level = str(sys.argv[4]) #sd size 
external_clusters = int(str(sys.argv[5])) #sd size 
external_ppc = int(str(sys.argv[6])) #sd size 
external_dim = int(str(sys.argv[7])) #sd size 

pose_estimator = stabledino(use_sd=1,use_dino=1,sd_size=external_size,sd_timestep=external_timestep,sd_raw=external_raw,sd_level=external_level,pc_dim=external_dim)

data_out =[]
BOP_data=[]
ac_count1 = 0
ac_count2 = 0
Errorcount_mask = 0
Errorcount_Pose = 0
temp_count_dino = 0
temp_count_sd = 0
temp_count_sd_uncut  = 0
temp_count_sd_adim  = 0
temp_count_sd_uncut_adim = 0
for key in template_keys:
    #if key != '1' and key != '8':continue
    #if key != '1' :continue
    pose_estimator.set_paths_and_load_data(config['templates_gt_path'], config['norm_factor_path'],key)
    for img_id in data_gt:
        #if(img_id != '58'):continue
        for i in range(len(data_gt[img_id])):
            obj_number = i
            obj_id = data_gt[img_id][obj_number]['obj_id']
            if(obj_id != int(key)):continue
            cam_K = np.array(data_gt[img_id][obj_number]['cam_K']).reshape((3,3))

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

            try:
                start_time = time.time()
                
                # To estimate the objects Rotation R and translation t the input image, the object_id, a segmentation mask and camera matrix are necessary
                R_gt = np.array(data_gt[img_id][obj_number]['cam_R_m2c']).reshape(3, 3)
                t_gt = np.array(data_gt[img_id][obj_number]['cam_t_m2c'])
            
                save = savepath+str(img_id)+"_"+str(key)
                try:
                    desc_dino = np.load((save+"_dino.npy"))
                    if(pose_estimator.raw):
                        desc_sd = np.load((save+"_sd_"+ pose_estimator.level +".npy"))
                        desc_sd_2 = np.load((save+"_sd_s2.npy"))
                        desc_sd_3 = np.load((save+"_sd_s3.npy"))
                        desc_sd_4 = np.load((save+"_sd_s4.npy"))
                        desc_sd_5 = np.load((save+"_sd_s5.npy"))
                    else:
                        desc_sd = np.load((save+"_sd.npy"))
                except:
                    print("Unknown Desc")
                    continue

                #Turn me On or Off
                # ON
                # matchedtemplate_dino = pose_estimator.get_template_knowndesc(img, str(obj_id), mask, cam_K, desc_dino,bbox=None)
                # R_est, t_est = pose_estimator.get_pose_from_dino_template(img, str(obj_id), mask, cam_K,matchedtemplate_dino, bbox=None)  
                # err,acc = eval_utils.calculate_score(R_est,R_gt,obj_id)

                # Off
                matchedtemplate_dino = [[0,0]]
                R_est, t_est = None, None
                err,acc = None, None 

                matchedtemplate_sd = pose_estimator.get_template_sd_knowndesc(img_sd, str(obj_id), mask_sd, cam_K,desc_sd,bbox=None,index = -1)   
                # R_sd, t_sd =  pose_estimator.get_pose_from_dino_template(img, str(obj_id), mask, cam_K,matchedtemplate_sd, bbox=None)
                # err_sd,acc_sd = eval_utils.calculate_score(R_sd,R_gt,obj_id)   
                # Off
                # matchedtemplate_sd = [[0,0]]
                R_sd, t_sd = None, None
                err_sd,acc_sd = None, None  
                #new way - 10 best dim + matching with all - uncutted desc
                #print(2)
                matchedtemplate_sd_uncut = matchedtemplate_sd

                path_sd_uncut = "/home/bernd/Workspace/DATA/lmo/desc/301/obj_"+str(obj_id)+"/"+str(matchedtemplate_sd_uncut[0][1]).zfill(6)

                bbox_sd_uncut = img_utils.get_bounding_box_from_mask(mask_sd)
                img_crop_sd_uncut, _,_ = img_utils.make_quadratic_crop(np.array(img), bbox_sd_uncut)
                mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox_sd_uncut)
                img_crop_sd_uncut = cv2.bitwise_and(img_crop_sd_uncut, img_crop_sd_uncut, mask=mask_crop)
                img1_sd_uncut = Image.fromarray(img_crop_sd_uncut)
                img2_sd_uncut = Image.open(path_sd_uncut+".png")
                img1_sd_uncut = resize(img1_sd_uncut, pose_estimator.real_size, resize=True, to_pil=True, edge=pose_estimator.EDGE_PAD)
                img2_sd_uncut = resize(img2_sd_uncut, pose_estimator.real_size, resize=True, to_pil=True, edge=pose_estimator.EDGE_PAD)

                # Angenommen, dies sind Ihre Tensoren
                tensor1_2 = torch.from_numpy(desc_sd_2)
                tensor1_3 = torch.from_numpy(desc_sd_3)
                tensor1_4 = torch.from_numpy(desc_sd_4)
                tensor1_5 = torch.from_numpy(desc_sd_5)

                tensor2_2 = torch.from_numpy(np.load(path_sd_uncut+'_sd_s2.npy'))
                tensor2_3 = torch.from_numpy(np.load(path_sd_uncut+'_sd_s3.npy'))
                tensor2_4 = torch.from_numpy(np.load(path_sd_uncut+'_sd_s4.npy'))
                tensor2_5 = torch.from_numpy(np.load(path_sd_uncut+'_sd_s5.npy'))

                # Erstellen Sie ein Wörterbuch und fügen Sie die Tensoren hinzu
                features1_sd_uncut = {
                    's2': tensor1_2,
                    's3': tensor1_3,
                    's4': tensor1_4,
                    's5': tensor1_5
                }

                features2_sd_uncut = {
                    's2': tensor2_2,
                    's3': tensor2_3,
                    's4': tensor2_4,
                    's5': tensor2_5
                }
                categories_sd_uncut = [['dog'], ['dog']]
                if type(categories_sd_uncut) == str:
                    categories_sd_uncut = [categories_sd_uncut]
                result_sd_uncut = pose_estimator.compute_pair_feature(pose_estimator.model, pose_estimator.aug, img1_sd_uncut, img2_sd_uncut,features1_sd_uncut,features2_sd_uncut, categories_sd_uncut, dist='cos')
                R_sd_uncut, t_sd_uncut = pose_estimator.get_pose_from_sd_cluster(result_sd_uncut,img, img1_sd_uncut, img2_sd_uncut, obj_id, mask, cam_K,matchedtemplate_sd_uncut,path=path_sd_uncut, bbox=None,ppc=external_ppc,clusters = external_clusters)
                err_sd_uncut,acc_sd_uncut = eval_utils.calculate_score(R_sd_uncut,R_gt,obj_id)                
                   
                # # Off
                # matchedtemplate_sd_uncut = [[0,0]]
                # R_sd_uncut, t_sd_uncut = None, None
                # err_sd_uncut,acc_sd_uncut = None, None 

                #new way - all dim + matching with all - cutted desc
                #print(3)
                # matchedtemplate_sd_adim = pose_estimator.get_template_sd_knowndesc_cutted(img_sd, str(obj_id), mask_sd, cam_K,desc_sd,bbox=None,index = -1)
                #R_sd_adim, t_sd_adim =  pose_estimator.get_pose_from_sd_template_new(img_sd, str(obj_id), mask_sd, cam_K,desc_sd,matchedtemplate_sd, bbox=None,index =-1)     
                # err_sd_adim,acc_sd_adim = eval_utils.calculate_score(R_sd_adim,R_gt,obj_id) 
                # Off
                matchedtemplate_sd_adim  = [[0,0]]
                R_sd_adim , t_sd_adim  = None, None
                err_sd_adim ,acc_sd_adim  = None, None                

                #new way - all dim + matching with all - uncutted desc
                #print(4)
                # matchedtemplate_sd_uncut_adim = pose_estimator.get_template_sd_knowndesc_uncutted(img_sd_uncut, str(obj_id), mask_sd_uncut, cam_K,desc_sd_uncut,bbox=None,index = -1)
                # R_sd_uncut_adim, t_sd_uncut_adim =  pose_estimator.get_pose_from_dino_template(img, str(obj_id), mask, cam_K,matchedtemplate_sd_uncut_adim, bbox=None)
                # # R_sd_uncut_adim, t_sd_uncut_adim =  pose_estimator.get_pose_from_sd_template_new_uncut(img_sd_uncut, str(obj_id), mask_sd_uncut, cam_K,desc_sd_uncut,matchedtemplate_sd_uncut, bbox=None,index =-1)     
                # err_sd_uncut_adim,acc_sd_uncut_adim = eval_utils.calculate_score(R_sd_uncut_adim,R_gt,obj_id)
                # Off
                matchedtemplate_sd_uncut_adim = [[0,0]]
                R_sd_uncut_adim, t_sd_uncut_adim = None, None
                err_sd_uncut_adim,acc_sd_uncut_adim = None, None 
                
                #new way - 10 best dim + matching with all - cutted desc - dino estimator
                #print(5)
                # matchedtemplate_sd_oma = matchedtemplate_sd_adim#pose_estimator.get_template_sd_knowndesc_cutted(img_sd, str(obj_id), mask_sd, cam_K,desc_sd,bbox=None,index = -1)
                # R_sd_oma, t_sd_oma =  pose_estimator.get_pose_from_dino_template(img, str(obj_id), mask, cam_K,matchedtemplate_sd_oma, bbox=None)    
                # err_sd_oma,acc_sd_oma = eval_utils.calculate_score(R_sd_oma,R_gt,obj_id) 
                # Off
                matchedtemplate_sd_oma = [[0,0]]
                R_sd_oma, t_sd_oma = None, None
                err_sd_oma,acc_sd_oma = None, None 

            

                end_time = time.time()    

            except:
                print("Error - no Pose avaiable")
                Errorcount_Pose +=1
                continue

     
            # difference_matrix_dino = temp_angles[matchedtemplate_dino[0][1]] - R_gt
            # frobenius_norm_dino = np.linalg.norm(difference_matrix_dino)
            # difference_matrix_sd = temp_angles[matchedtemplate_sd[0][1]] - R_gt
            # frobenius_norm_sd = np.linalg.norm(difference_matrix_sd)
            
            best_template = 0
            best_norm = 180.
            count=0
            for data_temp in temp_angles:             
                difference_matrix = data_temp - R_gt
                frobenius_norm = np.linalg.norm(difference_matrix)
                if frobenius_norm < best_norm:
                    best_norm = frobenius_norm
                    best_template = count
                count+=1


        

            if(matchedtemplate_dino[0][1] == best_template):temp_count_dino +=1
            if(matchedtemplate_sd[0][1] == best_template):temp_count_sd +=1
            if(matchedtemplate_sd_uncut[0][1] == best_template):temp_count_sd_uncut +=1
            if(matchedtemplate_sd_adim[0][1] == best_template):temp_count_sd_adim +=1
            if(matchedtemplate_sd_uncut_adim[0][1] == best_template):temp_count_sd_uncut_adim +=1
            if(acc_sd == 1):ac_count1 += 1
            if(acc_sd_uncut == 1):ac_count2 += 1

            print(f"Image: {img_id} - Object: {obj_id} - Template: {matchedtemplate_dino[0][1]} - {matchedtemplate_sd[0][1]} - {matchedtemplate_sd_adim[0][1]}  - {best_template} - Acc: {acc_sd} - {acc_sd_uncut} - {acc} - temp: {temp_count_dino} - {temp_count_sd} - {temp_count_sd_uncut} - {temp_count_sd_adim} - {temp_count_sd_uncut_adim} - hier: {ac_count1} - {ac_count2} - Err: {err_sd} - {err_sd_uncut}")

           
            data_out.append((img_id,obj_id,best_template,
                         matchedtemplate_dino[0][1]         ,matchedtemplate_dino[0][0]         ,err,acc,
                         matchedtemplate_sd[0][1]           ,matchedtemplate_sd[0][0]           ,err_sd,acc_sd,
                         matchedtemplate_sd_uncut[0][1]     ,matchedtemplate_sd_uncut[0][0]     ,err_sd_uncut,acc_sd_uncut,
                         matchedtemplate_sd_adim[0][1]      ,matchedtemplate_sd_adim[0][0]      ,err_sd_adim,acc_sd_adim,
                         matchedtemplate_sd_uncut_adim[0][1],matchedtemplate_sd_uncut_adim[0][0],err_sd_uncut_adim,acc_sd_uncut_adim,
                         matchedtemplate_sd_oma[0][1]       ,matchedtemplate_sd_oma[0][0]       ,err_sd_oma,acc_sd_oma))
            

            # Konvertieren Sie die Numpy-Matrizen in Listen und dann in Zeichenketten
            # print(R_sd_uncut,"\n",R_est,"\n\n",t_sd_uncut,"\n",t_est)
            R_sd_str = ' '.join(map(str, R_sd_uncut.flatten().tolist()))
            t_sd_str = ' '.join(map(str, t_sd_uncut.flatten().tolist()))
            # R_sd_str = ' '.join(map(str, R_est.flatten().tolist()))
            # t_sd_str = ' '.join(map(str, t_est.flatten().tolist()))

            # Fügen Sie die formatierten Daten in Ihre Liste ein
            BOP_data.append(("000002", img_id, obj_id, 1, R_sd_str, t_sd_str, -1))  # scene_id, im_id, obj_id, score, R, t, time
            
            #data_out.append((range_index,temp_count_sd))
            #print(f"R_est:\n {R_est}")
            #print(f"t_est:\n {t_est}")

            #out_img = vis_utils.draw_3D_bbox_on_image(np.array(img), R_est, t_est, cam_K, data_gt[img_id][obj_number]['model_info'], factor=1.0)
            #plt.imshow(out_img)
            #plt.show()


# DataFrame erstellen und transponieren
date_now = date.today()
now = datetime.now()

# Datum und Uhrzeit in einem bestimmten Format ausgeben
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
# Excel-Datei speichern
excel_dateiname = "/home/bernd/Workspace/StableDinoShot/Results/TemplateMatching/"+str(dt_string)+"---"+str(external_size)+"-"+str(external_timestep)+"-"+str(external_clusters)+"-"+str(external_ppc)+"-"+str(external_dim)+".xlsx"
BOP_excel = "/home/bernd/Workspace/StableDinoShot/Results/BOP/"+str(dt_string)+"---"+str(external_size)+"-"+str(external_timestep)+"-"+str(external_clusters)+"-"+str(external_ppc)+"-"+str(external_dim)+"_lmo-test.csv" #METHOD_DATASET-test.csv

# Erstellen Sie einen DataFrame aus Ihrer Datenliste
#df = pd.DataFrame(data_out, columns=['img_id', 'obj_id', 'best_template', 'matchedtemplate_dino', 'matchedtemplate_sd', 'best_norm', 'frobenius_norm_dino', 'frobenius_norm_sd', 'err-dino', 'acc-dino', 'err-sd', 'acc-sd','cosine_dino','cosine_sd','matchedtemplate_sd_cross','mt_manhattan','mt_euclid','mt_jaccard'])
df = pd.DataFrame(data_out, columns=['img_id','obj_id','best_template',
                         'matchedtemplate_dino[0][1]'         ,'matchedtemplate_dino[0][0]'         ,'err','acc',
                         'matchedtemplate_sd[0][1]'           ,'matchedtemplate_sd[0][0]'           ,'err_sd','acc_sd',
                         'matchedtemplate_sd_uncut[0][1]'     ,'matchedtemplate_sd_uncut[0][0]'     ,'err_sd_uncut','acc_sd_uncut',
                         'matchedtemplate_sd_adim[0][1]'      ,'matchedtemplate_sd_adim[0][0]'      ,'err_sd_adim','acc_sd_adim',
                         'matchedtemplate_sd_uncut_adim[0][1]','matchedtemplate_sd_uncut_adim[0][0]','err_sd_uncut_adim','acc_sd_uncut_adim',
                         'matchedtemplate_sd_oma[0][1]'           ,'matchedtemplate_sd_oma[0][0]'           ,'err_sd_oma','acc_sd_oma'])

BOP_df = pd.DataFrame(BOP_data)
# Speichern Sie den DataFrame in eine Excel-Datei
df.to_excel(excel_dateiname, index=False)
BOP_df.to_csv(BOP_excel, index=False, header=False)
"""sum = 0
for img_id,obj_id,acc in data_out:
sum+=acc
acc_tot = sum/len(data_out)
print(f"Gesamte Acc:\t{acc_tot}")"""


