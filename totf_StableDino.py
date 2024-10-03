import os
import json
import torch
from tqdm import tqdm
import numpy as np
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import logging
from src.pose_extractor import PoseViTExtractor
from sklearn.cluster import KMeans
import math
from extractor_sd import load_model, process_features_and_mask, get_mask
from sd_utils.utils_correspondence import resize 
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA as sklearnPCA


from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import copy


import torch.nn.functional as F
import torchvision.transforms as transforms


class stabledino:

    def __init__(self, use_sd=1, use_dino=1,  model_type='dino_vits8', stride=4, subset_templates=1, max_crop_size=80,sd_size=512,sd_ver="v1-3",sd_timestep=50,sd_raw=True,sd_level='s5',pc_dim = 128):
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        #DINO Stuff      
        self.extractor_dino = None
        self.model_type = model_type
        self.stride = stride
        self.subset_templates = subset_templates
        self.max_crop_size = max_crop_size


        self.max_crop_size = max_crop_size
        self.templates_gt = None
        self.norm_factors = None  
        self.USE_SD = use_sd
        self.USE_DINO = use_dino                         
        self.templates_desc_dino = {}                         
        self.templates_desc_sd = {}
        self.templates_mask = {}
        self.templates_gt = {}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # SD Stuff
        self.sd_size = sd_size
        self.SEED = 42
        self.TIMESTEP = sd_timestep
        self.VER = sd_ver   
        self.model = None
        self.aug = None
        self.MASK = True     
        self.raw=sd_raw
        self.level = sd_level
        self.templateCrop = None    
        self.CO_PCA = True
        self.PCA_DIMS = [pc_dim,pc_dim,pc_dim]
        #self.PCA_DIMS = [256, 256, 256]
        self.SIZE =512
        self.real_size=sd_size#244
        self.EDGE_PAD = False

        self.dist = 'cos'

        
        if self.USE_SD:
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed(self.SEED)
            torch.backends.cudnn.benchmark = True
            self.model, self.aug = load_model(diffusion_ver=self.VER, image_size=self.SIZE, num_timesteps=self.TIMESTEP)
        
        if self.USE_DINO:
            self.extractor_dino = PoseViTExtractor(model_type = self.model_type, stride = self.stride, device = self.device)
            self.patch_size = self.extractor_dino.model.patch_embed.patch_size  

        print("Modell initialized.")
    
        
    def set_paths_and_load_data(self,templates_gt_path, norm_factors_path, target_obj_id='ALL'):

        try:
            with open(os.path.join(templates_gt_path),'r') as f:
                self.templates_gt = json.load(f)

            with open(os.path.join(norm_factors_path), 'r') as f:
                self.norm_factors = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load templates or norm_factors: {e}")
            raise    

                                
        self.templates_desc_dino = {}
        self.templates_desc_sd = {}
        self.templates_desc_sd_uncut = {}
        self.templates_mask = {}
        self.templateCrop = {}
        templates_gt_subset = {}
        print("Load Templates and Mask - Start")
        
        try:
            ending = None
            if(self.raw): ending = "_"+str(self.level)
            else: ending = ""
            for obj_id, template_labels in tqdm(self.templates_gt.items()):
                if(str(obj_id)!=target_obj_id and not target_obj_id=='ALL'):continue
                if self.USE_DINO:
                    self.templates_desc_dino[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc'])).unsqueeze(0) # Why unsqueeze? - damit torch.cat durchgeführt werden kann
                                                            for i, template_label in enumerate(template_labels) 
                                                            if i%self.subset_templates==0], dim=0)   
                if self.USE_SD or True:   # Here delete True
                    self.templates_desc_sd[obj_id] = torch.cat([torch.from_numpy(np.load(template_label['img_desc_sd'+ending])).unsqueeze(0) # Why unsqueeze? - damit torch.cat durchgeführt werden kann
                                                            for i, template_label in enumerate(template_labels) 
                                                            if i%self.subset_templates==0], dim=0)    
                    
                
                templates_gt_subset[obj_id] = [template_label for i, template_label in 
                                            enumerate(template_labels) if i%self.subset_templates==0]
                
                self.templateCrop[obj_id]=[cv2.resize(cv2.imread(template_label['img_crop'], 0),(self.sd_size,self.sd_size)) for i,template_label in enumerate(template_labels) if i%self.subset_templates==0]


                
                print("Loaded Desc of Obj_Id: ",obj_id)
            
            print("Load Templates and Mask - End")
            self.templates_gt = templates_gt_subset

        except Exception as e:
            self.logger.error(f"Error processing template descriptors: {e}")
            raise
        
        print("Preparing templates and loading of extractor is done!")

    # From Tale of two Features with slight changes
    def get_SD_features(self,img1):           
        img1_input = resize(img1, self.real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        with torch.no_grad():
            features1 = process_features_and_mask(self.model, self.aug, img1_input,  mask=False, raw=self.raw)
        return(features1,img1_input)

    # From Tale of two Features with slight changes
    def compute_pair_feature(self,model, aug, img1, img2,features1,features2, category, dist='cos'):
        result = []

        img1 = resize(img1, self.real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        img2 = resize(img2, self.real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)

        with torch.no_grad():       
            processed_features1, processed_features2 = self.co_pca(features1, features2, self.PCA_DIMS)          
            num_patches = processed_features1.shape[3]
            img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
            img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                                
            mask1 = get_mask(model, aug, img1, category[0])
            mask2 = get_mask(model, aug, img2, category[-1])
            result.append([img1_desc.cpu(), img2_desc.cpu(), mask1.cpu(), mask2.cpu()])

        return result
    
    # From Tale of two Features with slight changes
    def vis_pca_mask(result,save_path):
        # PCA visualization mask version
        for (feature1,feature2,mask1,mask2) in result:
            num_patches = int(math.sqrt(feature1.shape[-2]))
            feature1 = feature1.squeeze() # shape (3600,768*2)
            feature2 = feature2.squeeze() # shape (3600,768*2)
            chennel_dim = feature1.shape[-1]
            # resize back
            src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
            tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
            resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
            resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
            src_feature_upsampled = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
            tgt_feature_upsampled = tgt_feature_reshaped * resized_tgt_mask.repeat(src_feature_reshaped.shape[0],1,1)
            feature1=src_feature_upsampled.reshape(chennel_dim,-1).permute(1,0)
            feature2=tgt_feature_upsampled.reshape(chennel_dim,-1).permute(1,0)

            n_components=4 # the first component is to seperate the object from the background
            pca = sklearnPCA(n_components=n_components)
            feature1_n_feature2 = torch.cat((feature1,feature2),dim=0) # shape (7200,768*2)
            feature1_n_feature2 = pca.fit_transform(feature1_n_feature2.cpu().numpy()) # shape (7200,3)
            feature1 = feature1_n_feature2[:feature1.shape[0],:] # shape (3600,3)
            feature2 = feature1_n_feature2[feature1.shape[0]:,:] # shape (3600,3)
            
            
            fig, axes = plt.subplots(4, 2, figsize=(10, 14))
            for show_channel in range(n_components):
                if show_channel==0:
                    continue
                # min max normalize the feature map
                feature1[:, show_channel] = (feature1[:, show_channel] - feature1[:, show_channel].min()) / (feature1[:, show_channel].max() - feature1[:, show_channel].min())
                feature2[:, show_channel] = (feature2[:, show_channel] - feature2[:, show_channel].min()) / (feature2[:, show_channel].max() - feature2[:, show_channel].min())
                feature1_first_channel = feature1[:, show_channel].reshape(num_patches,num_patches)
                feature2_first_channel = feature2[:, show_channel].reshape(num_patches,num_patches)

                axes[show_channel-1, 0].imshow(feature1_first_channel)
                axes[show_channel-1, 0].axis('off')
                axes[show_channel-1, 1].imshow(feature2_first_channel)
                axes[show_channel-1, 1].axis('off')
                axes[show_channel-1, 0].set_title('Feature 1 - Channel {}'.format(show_channel ), fontsize=14)
                axes[show_channel-1, 1].set_title('Feature 2 - Channel {}'.format(show_channel ), fontsize=14)


            feature1_resized = feature1[:, 1:4].reshape(num_patches,num_patches, 3)
            feature2_resized = feature2[:, 1:4].reshape(num_patches,num_patches, 3)

            axes[3, 0].imshow(feature1_resized)
            axes[3, 0].axis('off')
            axes[3, 1].imshow(feature2_resized)
            axes[3, 1].axis('off')
            axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
            axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)

            plt.tight_layout()
            plt.show()


    # From Tale of two Features with slight changes -> 4 layers instead of 3 + interpolated to the biggest
    def co_pca(self,features1, features2, dim=[128,128,128,128]):
         
        dim=[128,128,128,128]
        processed_features1 = {}
        processed_features2 = {}
        s5_size = features1['s5'].shape[-1]
        s4_size = features1['s4'].shape[-1]
        s3_size = features1['s3'].shape[-1]
        s2_size = features1['s2'].shape[-1]
        # Get the feature tensors
        s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
        s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
        s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)
        s2_1 = features1['s2'].reshape(features1['s2'].shape[0], features1['s2'].shape[1], -1)

        s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
        s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
        s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
        s2_2 = features2['s2'].reshape(features2['s2'].shape[0], features2['s2'].shape[1], -1)
        # Define the target dimensions
        target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2], 's2': dim[2] }

        # Compute the PCA
        for name, tensors in zip(['s5', 's4', 's3', 's2'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2], [s2_1, s2_2]]):
            target_dim = target_dims[name]

            # Concatenate the features
            features = torch.cat(tensors, dim=-1) # along the spatial dimension
            features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)
            
            # equivalent to the above, pytorch implementation
            mean = torch.mean(features[0], dim=0, keepdim=True)
            centered_features = features[0] - mean
            U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
            reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
            features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
            

            # Split the features
            processed_features1[name] = features[:, :, :features.shape[-1] // 2] # Bx(d)x(t_x)
            processed_features2[name] = features[:, :, features.shape[-1] // 2:] # Bx(d)x(t_y)

        # reshape the features
        processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
        processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
        processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)
        processed_features1['s2']=processed_features1['s2'].reshape(processed_features1['s2'].shape[0], -1, s2_size, s2_size)

        processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
        processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
        processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)
        processed_features2['s2']=processed_features2['s2'].reshape(processed_features2['s2'].shape[0], -1, s2_size, s2_size)

        # Upsample s5, s4, and s3 spatially by a factor of 2 to match s2 size
        for layer in ['s5', 's4', 's3']:
            processed_features1[layer] = F.interpolate(processed_features1[layer], size=(processed_features1['s2'].shape[-2:]), mode='bilinear', align_corners=False)
            processed_features2[layer] = F.interpolate(processed_features2[layer], size=(processed_features2['s2'].shape[-2:]), mode='bilinear', align_corners=False)

        # Concatenate upsampled_s5, s4, s3 and s2 to create new features
        features1_gether = torch.cat([processed_features1['s2'], processed_features1['s3'], processed_features1['s4'], processed_features1['s5']], dim=1)
        features2_gether = torch.cat([processed_features2['s2'], processed_features2['s3'], processed_features2['s4'], processed_features2['s5']], dim=1)

        # # TRY 1 - only layer 11
        # features1_gether = processed_features1['s2']
        # features2_gether = processed_features2['s2']

        return features1_gether, features2_gether


    def perform_clustering(self,features, n_clusters=10):
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        # Convert the features to float32
        features = features.cpu().detach().numpy().astype('float32')
        # Initialize a k-means clustering index with the desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # Train the k-means index with the features
        kmeans.fit(features)
        # Assign the features to their nearest cluster
        labels = kmeans.predict(features)

        return labels
    
    # From Tale of two Features with big changes
    def cluster_and_match(self,result,img1,img2,uv_img = None,norm_factor = None, n_clusters=120,ppc = 20):
        for (feature1,feature2,mask1,mask2) in result:
            # feature1 shape (1,1,3600,768*2)
            num_patches = int(math.sqrt(feature1.shape[-2]))
            # pca the concatenated feature to 3 dimensions
            feature1 = feature1.squeeze() # shape (3600,768*2)
            feature2 = feature2.squeeze() # shape (3600,768*2)
            #size = self.real_size
            size = int(math.sqrt(feature1.shape[0]))
            img1_resized = img1.resize((size, size))
            img1_bw = np.array(img1_resized.convert('L'))  # '1' für binäre Schwarz-Weiß-Maske
            img1_bw[img1_bw > 0] = 1
            img1_bw = torch.from_numpy(img1_bw).cuda()
            img2_resized = img2.resize((size, size))
            img2_bw = np.array(img2_resized.convert('L')) # '1' für binäre Schwarz-Weiß-Maske
            img2_bw[img2_bw > 0] = 1
            img2_bw = torch.from_numpy(img2_bw).cuda()           


            src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
            tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()

            
            resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(size, size), mode='nearest').squeeze().cuda()
            resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(size, size), mode='nearest').squeeze().cuda()
            
            resized_src_mask = resized_src_mask * img1_bw
            resized_tgt_mask = resized_tgt_mask * img2_bw

            src_feature_upsampled = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
            tgt_feature_upsampled = tgt_feature_reshaped * resized_tgt_mask.repeat(tgt_feature_reshaped.shape[0],1,1)

            feature1=src_feature_upsampled.unsqueeze(0)
            feature2=tgt_feature_upsampled.unsqueeze(0)
            
            w1, h1 = feature1.shape[2], feature1.shape[3]
            w2, h2 = feature2.shape[2], feature2.shape[3]

            features1_2d = feature1.reshape(feature1.shape[1], -1).permute(1, 0)
            features2_2d = feature2.reshape(feature2.shape[1], -1).permute(1, 0)        

            labels_img1 = self.perform_clustering(features1_2d, n_clusters)
            labels_img2 = self.perform_clustering(features2_2d, n_clusters)

            cluster_means_img1 = [features1_2d.cpu().detach().numpy()[labels_img1 == i].mean(axis=0) for i in range(n_clusters)]
            cluster_means_img2 = [features2_2d.cpu().detach().numpy()[labels_img2 == i].mean(axis=0) for i in range(n_clusters)]

            distances = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img2, axis=0), axis=-1)

            # Use Hungarian algorithm to find the optimal bijective mapping
            row_ind, col_ind = linear_sum_assignment(distances)

            relabeled_img2 = np.zeros_like(labels_img2)
            for i, match in zip(row_ind, col_ind):
                relabeled_img2[labels_img2 == match] = i

            labels_img1 = labels_img1.reshape(w1, h1)
            relabeled_img2 = relabeled_img2.reshape(w2, h2)

            labels_img1 = labels_img1
            labels_img2 = relabeled_img2

            # Finden Sie die einzigartigen Werte und ihre Häufigkeiten in labels_img1
            unique1, counts1 = np.unique(labels_img1, return_counts=True)
            unique2, counts2 = np.unique(labels_img2, return_counts=True)
            # Finden Sie den Index des am häufigsten auftretenden Werts
            index1 = np.argmax(counts1)
            # Finden Sie den am häufigsten auftretenden Wert - hintergrund
            background = unique1[index1]

            scale = self.real_size/src_feature_reshaped.shape[2]
            
            result = []
            src = src_feature_reshaped.cpu().numpy()
            tgt = tgt_feature_reshaped.cpu().numpy()

            # Vorbereitung der Indizes
            indices1 = np.transpose(np.nonzero(labels_img1[..., np.newaxis] == np.arange(n_clusters)))
            indices2 = np.transpose(np.nonzero(labels_img2[..., np.newaxis] == np.arange(n_clusters)))

            """# Mean Filter
            sum_mean,counter_mean = 0,0
            for i in range(n_clusters):
                if i != background:# and i not in edges:
                    # Extrahieren Sie die Indizes für das aktuelle Cluster
                    cluster_indices1 = indices1[indices1[:, -1] == i][:, :-1]
                    cluster_indices2 = indices2[indices2[:, -1] == i][:, :-1]

                    # Extrahieren Sie die Merkmale für das aktuelle Cluster
                    features1 = src[:, cluster_indices1[:, 0], cluster_indices1[:, 1]]
                    features2 = tgt[:, cluster_indices2[:, 0], cluster_indices2[:, 1]]

                    # Berechnen Sie die Kosinus-Ähnlichkeiten
                    similarities = cosine_similarity(features1.T, features2.T)    
                    counter_mean += 1   
                    sum_mean += similarities.mean()          
            
            sum_mean = sum_mean/counter_mean
            low_mean_list = []
            for i in range(n_clusters):
                if i != background:# and i not in edges:
                    # Extrahieren Sie die Indizes für das aktuelle Cluster
                    cluster_indices1 = indices1[indices1[:, -1] == i][:, :-1]
                    cluster_indices2 = indices2[indices2[:, -1] == i][:, :-1]

                    # Extrahieren Sie die Merkmale für das aktuelle Cluster
                    features1 = src[:, cluster_indices1[:, 0], cluster_indices1[:, 1]]
                    features2 = tgt[:, cluster_indices2[:, 0], cluster_indices2[:, 1]]

                    # Berechnen Sie die Kosinus-Ähnlichkeiten
                    similarities = cosine_similarity(features1.T, features2.T)    
                    if(similarities.mean() <= sum_mean):low_mean_list.append((i))   """

            result = []

            # Durchlaufen Sie alle Cluster
            for i in range(n_clusters):
                if i != background: # Hier and i not in low_mean_list:
                    # Extrahieren Sie die Indizes für das aktuelle Cluster
                    cluster_indices1 = indices1[indices1[:, -1] == i][:, :-1]
                    cluster_indices2 = indices2[indices2[:, -1] == i][:, :-1]

                    # Extrahieren Sie die Merkmale für das aktuelle Cluster
                    features1 = src[:, cluster_indices1[:, 0], cluster_indices1[:, 1]]
                    features2 = tgt[:, cluster_indices2[:, 0], cluster_indices2[:, 1]]

                    # Berechnen Sie die Kosinus-Ähnlichkeiten
                    similarities = cosine_similarity(features1.T, features2.T)

                    # Finden Sie die Indizes der maximalen Ähnlichkeiten
                    max_similarities_indices = np.argmax(similarities, axis=1)

                    # Fügen Sie die Ergebnisse zur Ergebnisliste hinzu
                    for j, (x1, y1) in enumerate(cluster_indices1):
                        similarity = similarities[j, max_similarities_indices[j]]
                        x2, y2 = cluster_indices2[max_similarities_indices[j]]
                        result.append((similarity, x1, y1, x2, y2, i))#,similarities.mean()))

            

            result = np.array(result)
            result_dict = {}
            #Filtern der Punkte pro Cluster
            for item in result:
                similarity, x1, y1, x2, y2, i = item
                key = (i)
                if key not in result_dict:
                    result_dict[key] = [(similarity, x1, y1, x2, y2)]
                else:
                    result_dict[key].append((similarity, x1, y1, x2, y2))
                    # Sortieren Sie die Liste in absteigender Reihenfolge der Ähnlichkeit
                    result_dict[key].sort(key=lambda x: x[0], reverse=True)
                    # Behalten Sie nur die drei höchsten Werte
                    result_dict[key] = result_dict[key][:ppc]#1

            result_filtered = []
            for i, values in result_dict.items():
                for value in values:
                    result_filtered.append((*value, i))
            result = result_filtered

            result = np.array(result)

            
            data = result
            similarities = data[:, 0]
            x1 = data[:, 1:3]
            x2 = data[:, 3:5]
            i_values = data[:, 5]
            # Verwenden von den Acht-Punkte-Algorithmus mit RANSAC, um die Fundamentalmatrix zu schätzen
            Fer, mask = cv2.findFundamentalMat(x1, x2, cv2.FM_RANSAC, 0.75, 0.99)
            # Verwenden der Maske, um die Inliers zu filtern
            inliers1 = x1[mask.ravel()==1]
            inliers2 = x2[mask.ravel()==1]
            inlier_similarities = similarities[mask.ravel()==1]
            inlier_i_values = i_values[mask.ravel()==1]

            filtered_result = list(zip(inlier_similarities, inliers1[:, 0], inliers1[:, 1], inliers2[:, 0], inliers2[:, 1], inlier_i_values)) 
            filtered_result = sorted(filtered_result, key=lambda x: x[0], reverse=True)


            best_results = filtered_result[:]#200 # round(len(filtered_result)/2)

            

            
            def cosine_similarity_single(A, B):
                A = A.flatten()  # reshape Array A to 1D
                B = B.flatten()  # reshape Array B to 1D
                dot_product = np.dot(A, B)
                norm_A = np.linalg.norm(A)
                norm_B = np.linalg.norm(B)
                return dot_product / (norm_A * norm_B)


            scale = self.real_size/src_feature_reshaped.shape[2] #HIER        
            kp1 = []
            kp2 = []
            # Subpixel genauigkeit
            for r in best_results:
                similarity, x1, y1, x2, y2, i = r
                
                # 2.nd Picture
                max_x,max_y = 3,3# src.shape[2]
                min_x,min_y= 0,0

                direction_array_2 = np.zeros((3, 3))
                if int(x2) == feature1.shape[2]-1:
                    max_x = max_x-1
                if int(y2) == feature2.shape[2]-1:
                    max_y = max_y-1
                if int(x2) == 0:
                    min_x = min_x+1
                if int(y2) == 0:
                    min_y = min_y+1

                features1 = src[:, int(x1):int(x1)+1, int(y1):int(y1)+1]
                features2 = None
                
                for x_index in range(min_x,max_x,1):
                    for y_index in range(min_y,max_y,1):
                        features2 = tgt[:, int(x2)+x_index-1:int(x2)+x_index, int(y2)+y_index-1:int(y2)+y_index]
                        similarity = cosine_similarity_single(features1, features2)
                        direction_array_2[x_index][y_index]=similarity
                direction_array_2 = np.array(direction_array_2)
                #Normalisieren
                direction_array_2 = (direction_array_2 - np.min(direction_array_2)) / (np.max(direction_array_2) - np.min(direction_array_2))

                left = direction_array_2.T[0].mean()
                right = direction_array_2.T[2].mean()
                top = direction_array_2[0].mean()
                bottom = direction_array_2[2].mean()

                x_factor_2 = bottom - top
                y_factor_2 = right - left

                # 1. Pic
                max_x,max_y = 3,3# src.shape[2]
                min_x,min_y= 0,0

                direction_array_1 = np.zeros((3, 3))
                if int(x1) == feature1.shape[2]-1:
                    max_x = max_x-1
                if int(y1) == feature2.shape[2]-1:
                    max_y = max_y-1
                if int(x1) == 0:
                    min_x = min_x+1
                if int(y1) == 0:
                    min_y = min_y+1

                features1 = None
                features2 = tgt[:, int(x1):int(x1)+1, int(y1):int(y1)+1]
                
                for x_index in range(min_x,max_x,1):
                    for y_index in range(min_y,max_y,1):
                        features1 = src[:, int(x1)+x_index-1:int(x1)+x_index, int(y1)+y_index-1:int(y1)+y_index]
                        similarity = cosine_similarity_single(features1, features2)
                        direction_array_1[x_index][y_index]=similarity
                direction_array_1 = np.array(direction_array_1)
                #Normalisieren
                direction_array_1 = (direction_array_1 - np.min(direction_array_1)) / (np.max(direction_array_1) - np.min(direction_array_1))

                left = direction_array_1.T[0].mean()
                right = direction_array_1.T[2].mean()
                top = direction_array_1[0].mean()
                bottom = direction_array_1[2].mean()

                x_factor_1 = bottom - top
                y_factor_1 = right - left
                
                x1_scaled = round(x1*scale+scale/2+x_factor_1*scale/2)
                y1_scaled = round(y1*scale+scale/2+y_factor_1*scale/2)

                x2_scaled = round(x2*scale+scale/2+x_factor_2*scale/2)
                y2_scaled = round(y2*scale+scale/2+y_factor_2*scale/2)

                if(x1_scaled > self.real_size -1): x1_scaled= self.real_size-1
                if(y1_scaled > self.real_size -1): y1_scaled= self.real_size-1
                if(x2_scaled > self.real_size -1): x2_scaled= self.real_size-1
                if(y2_scaled > self.real_size -1): y2_scaled= self.real_size-1

                # Fügen Sie die Koordinaten zu den entsprechenden Listen hinzu
                kp1.append((x1_scaled, y1_scaled))
                kp2.append((x2_scaled, y2_scaled))



            return kp1,kp2


     
    def get_pose_from_sd_cluster(self, result,img, img1, img2, obj_id, mask, cam_K,matched,path, bbox=None,ppc=20,clusters=120):
        try:            
            matched_templates = matched

            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            img_crop = Image.fromarray(img_crop)

            with torch.no_grad():   
                img_uv = np.load(path+"_uv.npy")
                img_uv = img_uv.astype(np.uint8)
                img_uv = cv2.resize(img_uv, (self.real_size, self.real_size))      

                points1, points2 = self.cluster_and_match(result,img1,img2,img_uv,self.norm_factors[str(obj_id)],ppc =ppc,n_clusters = clusters)

                resize_factor = self.real_size / float(img_crop.size[0])


                R_est, t_est = utils.get_pose_from_correspondences(points1, points2, 
                                                                   y_offset, x_offset, 
                                                                   img_uv, cam_K, 
                                                                   self.norm_factors[str(obj_id)], 
                                                                   scale_factor=1.0, 
                                                                   resize_factor=resize_factor)
                
                return R_est, t_est
        except Exception as e:
            self.logger.error(f"Error in get_pose (SD): {e}")
            raise

    def get_template_knowndesc(self, img, obj_id, mask, cam_K, desc_dino, bbox=None):
        try:
            matched_templates = utils.find_template_cpu(desc_dino, self.templates_desc_dino[obj_id], num_results=1)

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")

            #template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            return matched_templates
        except Exception as e:
            self.logger.error(f"Error in get_template (DINO): {e}")
            raise


    def get_pose_from_dino_template(self, img, obj_id, mask, cam_K,matched, bbox=None):
        try:
            if bbox is None:
                bbox = img_utils.get_bounding_box_from_mask(mask)

            img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
            mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
            img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            
            
            img_crop = Image.fromarray(img_crop)
            img_prep, _, _ = self.extractor_dino.preprocess(img_crop, load_size=224)

            with torch.no_grad():
                desc_dino = self.extractor_dino.extract_descriptors(img_prep.to(self.device), layer=11, facet='key', bin=False, include_cls=True)
                desc_dino = desc_dino.squeeze(0).squeeze(0).detach().cpu()

            #matched_templates = utils.find_template_cpu(desc_dino, self.templates_desc_dino[obj_id], num_results=1)
            matched_templates = matched

            if not matched_templates:
                raise ValueError("No matched templates found for the object. (DINO)")

            template = Image.open(self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'])

            with torch.no_grad():
                if img_crop.size[0] < self.max_crop_size:
                    crop_size = img_crop.size[0]
                else:
                    crop_size = self.max_crop_size

                resize_factor = float(crop_size) / img_crop.size[0]

                points1, points2, crop_pil, template_pil = self.extractor_dino.find_correspondences_fastkmeans(img_crop, template, num_pairs=20, load_size=crop_size)

                if not points1 or not points2:
                    raise ValueError("Insufficient correspondences found.")

                img_uv = np.load(f"{self.templates_gt[obj_id][matched_templates[0][1]]['img_crop'].split('.png')[0]}_uv.npy")
                img_uv = img_uv.astype(np.uint8)
                img_uv = cv2.resize(img_uv, (crop_size, crop_size))
                

                R_est, t_est = utils.get_pose_from_correspondences(points1, points2, 
                                                                   y_offset, x_offset, 
                                                                   img_uv, cam_K, 
                                                                   self.norm_factors[str(obj_id)], 
                                                                   scale_factor=1.0, 
                                                                   resize_factor=resize_factor)
                
                return R_est, t_est
        except Exception as e:
            self.logger.error(f"Error in get_pose (DINO): {e}")
            raise

    def get_template_sd_knowndesc(self, img, obj_id, mask, cam_K,desc_sd, bbox=None,index =None):
        try:
            desc_sd =  torch.from_numpy(desc_sd)
            matched_templates = self.find_template_cpu_sd_cross(desc_sd, self.templates_desc_sd[obj_id].numpy(),index=index, num_results=1)
            

            if not matched_templates:
                raise ValueError("No matched templates found for the object.")


            return matched_templates
        except Exception as e:
            self.logger.error(f"Error in get_template_sd_knowndesc: {e}")
            raise

    
    def find_template_cpu_sd_cross(self, desc_input, desc_templates, index, num_results):


        #print("find_template_cpu_sd_cross: ",desc_input.shape)
        #print(index)
        if index == -1:
            helper = []
            for dim in range(0,desc_input.shape[1],1):
                helper.append(dim)
            index = helper
        else:
            helper = []
            for dim in index:
                helper.append(int(dim))
            index = helper


        def build_pyramid(tensor, max_level):
            sizelimit = int(tensor.shape[2]/2)
            pyramid = [tensor]
            tensor_origin = copy.deepcopy(tensor)
            for i in range(0,max_level,1):
                scale = 1 - 0.2 * i
                tensor = copy.deepcopy(tensor_origin)
                tensor = F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
                #print(tensor.shape)
                pyramid.append(tensor)
                if(tensor.shape[2]<sizelimit):break
            return pyramid

        def cross_correlation(x, y):
            x = x - x.mean(dim=(-1, -2), keepdim=True)
            y = y - y.mean(dim=(-1, -2), keepdim=True)
            x = F.normalize(x, p=2, dim=(-1, -2))
            y = F.normalize(y, p=2, dim=(-1, -2))
            return (x * y).sum(dim=(-1, -2))

        def match_template(tensor, template):
            result = F.conv2d(tensor, template)
            return result


        def scale_image(img, scale):
            # Skalieren
            width, height = img.size
            scale = 1 - 0.2 * scale
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height))
            return img


        def process_tensors(tensor1, tensor2_list,index):
            
            #print(index)
            index_tensor = torch.tensor(index, dtype=torch.long).to(self.device)
            # Initialisieren Sie eine Liste, um die Ergebnisse für jeden Tensor zu speichern
            tensor1 = tensor1.to(self.device)  # Verschieben Sie tensor1 auf die GPU
            tensor1 = torch.index_select(tensor1, 1, index_tensor)  # Wählen Sie die Kanäle aus tensor1 aus
            results = []
            count = 0
            # Iterieren Sie über jeden Tensor in der Liste
            for tensor2 in tensor2_list:
                tensor2 = torch.from_numpy(tensor2).to(self.device)  # Verschieben Sie tensor1 auf die GPU                
                tensor2 = torch.index_select(tensor2, 1, index_tensor)  # Wählen Sie die Kanäle aus tensor2 aus
                # Initialisieren Sie die beste Korrelation als einen sehr niedrigen Wert
                best_corr = 0
                best_scale = None
                best_match = None
                best_position = (None, None)
                best_angle = None
                best_scale_factor = None

                # Erstellen Sie eine Pyramide für den ersten Tensor
                pyramid1 = build_pyramid(tensor1, max_level=2)

                # Durchlaufen Sie jede Ebene der Pyramide
                for i, p1 in enumerate(pyramid1):
                    # Fügen Sie eine Schleife hinzu, um das Bild zu drehen
                    for angle in range(-15, 30, 15):  # von -40° bis +40° in 10° Schritten
                        # Drehen des Bildes
                        p1_rotated = transforms.functional.rotate(p1, angle)

                        # Berechnen des Template Matching und die Kreuzkorrelation
                        result = match_template(tensor2, p1_rotated)
                        max_val, indices = torch.max(result.view(result.size(0), -1), dim=1)
                        y, x = np.unravel_index(indices.cpu().numpy(), result.size()[2:])

                        # Schneiden Sie das beste Match aus tensor2 aus
                        h, w = p1_rotated.size(-2), p1_rotated.size(-1)    
                        y, x = int(y), int(x)  # Konvertieren Sie y und x in Python-Integers
                        tensor2_cut = tensor2[..., y:y+h, x:x+w]

                        # Berechnen der Kreuzkorrelation
                        cc = cross_correlation(p1_rotated, tensor2_cut)

                        # Überprüfen ob dies die beste Korrelation ist
                        max_cc = cc.max().item()
                        mean_value = cc.mean().item()
                        if mean_value > best_corr:
                            best_corr = mean_value
                            best_scale = i
                            best_position = (x, y)
                            best_angle = angle  # Speichern Sie den besten Winkel
                            best_scale_factor = tensor1.shape[-1] / tensor2_cut.shape[-1]  # Berechnen Sie den Skalierungsfaktor

                # Fügen Sie die Ergebnisse für diesen Tensor zur Ergebnisliste hinzu
                results.append((best_corr, count, best_scale, best_position, best_angle,best_scale_factor))
                count += 1

            # Geben Sie die Ergebnisliste zurück
            return results

        # Rufen Sie die Funktion mit tensor1 und desc_img2 auf
        results = process_tensors(desc_input, desc_templates,index)

        # Jetzt können Sie den besten Tensor außerhalb der Funktion auswählen
        # Zum Beispiel können Sie den Tensor mit der höchsten best_corr auswählen
        best_result = max(results, key=lambda x: x[0])
        #print(f"Bestes Ergebnis: {best_result}")
        return [best_result]