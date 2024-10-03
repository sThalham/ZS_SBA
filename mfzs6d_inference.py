import copy
import os
import argparse
import configparser
import sys
import json
import torch
import time
import cv2
from PIL import Image
#from skimage import io, transform
import numpy as np
import math
#import matplotlib.pyplot as plt
#from torchvision import transforms, utils
from skimage import io, transform
from matplotlib import pyplot as plt
from scipy import spatial, linalg, optimize
from scipy.sparse import lil_matrix
import scipy

from mfzs6d_data_loader import HopeBOPDataset
from totf_StableDino import stabledino
from zs6d_dino2_extractor import PoseViTExtractor

from pose_utils import img_utils, utils, procrustes
from pose_utils import vis_utils, mfzs6d_utils, featuremetric_bundle_adjustment
import open3d
from scipy.optimize import least_squares

import json
import argparse



def load_pcd(data_path):
    # load meshes
    ply_path = os.path.join(data_path)
    pcd_model = open3d.io.read_point_cloud(ply_path)

    factor = 1.0
    if np.nanmax(pcd_model.points) < 10.0:
        factor = 1000.0
    model_vsd = {}
    model_vsd['pts'] = np.asarray(pcd_model.points)
    model_vsd['pts'] = model_vsd['pts'] * factor

    return model_vsd


def rectangular_crop_by_factor(img, mask, extractor, desc_img_size):
    if isinstance(mask, list):
        bbox = mask
    else:
        bbox = img_utils.get_bounding_box_from_mask(mask)
    img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)

    if isinstance(mask, list):
        mask_crop = None
    else:
        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)

    img_crop = Image.fromarray(img_crop)
    #img_prep, _, _ = extractor.preprocess(img_crop, load_size=224)
    img_vit, img_vanilla, _ = extractor.preprocess(img_crop, load_size=desc_img_size)

    return img_vit, img_vanilla, mask_crop, (y_offset, x_offset)

def filter_dict_key(data, key_name, value):
    return list(filter(lambda x: x.get(key_name) == value, data))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parse args and allow override of config args

    with open(sys.argv[1]) as handle:
        config = json.loads(handle.read())
    try:
        with open(sys.argv[1]) as handle:
            config = json.loads(handle.read())
    except ValueError as err:
        print("Expected syntax: \"python mf_zs6d_inference.py path/to/config/file.json\"")
        sys.exit()

    # All hail to parameter space
    external_size = int(512)  # sd size
    external_timestep = int(50)  # sd size
    template_size = config['template_size']
    number_keypoints = config['number_keypoints']
    external_raw = False

    # ... a parameter
    if not os.path.exists(config['results_path']):
        os.makedirs(config['results_path'])

    #############################
    # param space over
    #######################

    print('#################')
    print('Loading dataset with path: %s; stage: %s' %(config["dataset_path"], config["type"]))
    print('...')
    dataloader = HopeBOPDataset(config)
    print('Loading data done.')
    print('-Dataset info- Objects: ', config["obj_ids"])
    print('-Dataset info- Length %i' %dataloader.__len__())

    print('#################')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config["extractor"] not in ['sd', 'dino', 'totf']:
        raise ValueError('Please specify Stage. [\'onboarding_static\', \'onboarding_dynamic\', \'test\']')

    if config["extractor"] == "totf" or config["extractor"] == "sd":
        print('Instantiating TOTF with: image size: %i; time frame: %i' %(external_size, external_timestep))
        pose_estimator = stabledino(use_sd=1, use_dino=1, sd_size=external_size, sd_timestep=external_timestep,
                                    sd_raw=external_raw)
        extractor = pose_estimator.extractor_dino
        desc_img_size = pose_estimator.sd_size
        print('Instantiation of TOTF done')
    elif config["extractor"] == "dino":
        print('For now instantiating DinoV2...') #because, you know, memory issues on my high-end 2018 Dell Laptop
        extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)
        desc_img_size = 224
        print('Instantiating DinoV2 done.')

    print('##################')
    print('Loading templates ....')
    # template dictionary
    #tmp_dict = {"img_id": int(template['idx']),
    #            "obj_id": int(template['obj_id']),
    #            "cam_R_m2c": template['R'],
    #            "cam_t_m2c": template['t'],
    #            "cam_K": template['cam_K'],
    #            "img_crop": img_vit,
    #            "img_desc": desc
     #           }

    template_base_path = config['template_savepath']
    # Loading templates into gpu
    templates_desc = []
    templates_crops = []
    templates_R = []
    templates_t = []
    templates_K = []
    template_scales = []
    template_offsets = []
    template_bboxes = []
    for template in os.listdir(template_base_path):
        temp_path = os.path.join(template_base_path, template)
        el_templato = np.load(temp_path, allow_pickle=True).item()
        #print(el_templato.item())
        templates_desc.append(torch.from_numpy(el_templato['img_desc']))#.unsqueeze(0))
        templates_crops.append(el_templato['img_crop'])#.unsqueeze(0))
        templates_R.append(el_templato["cam_R_m2c"])
        templates_t.append(el_templato["cam_t_m2c"])
        templates_K.append(el_templato["cam_K"])
        template_scales.append(el_templato["template_scaling"])
        template_offsets.append(el_templato["template_offset"])
        template_bboxes.append(el_templato["bbox"])

    print("Preparing templates finished!")
    print('##################')
    print('Loading object models for visualization ....')
    object_models = [None] * 30
    for mesh_name in os.listdir(config['object_model_path']):
        if mesh_name.endswith('.ply'):
            input_file = os.path.join(config['object_model_path'], mesh_name)
            #pcd = o3d.io.read_point_cloud(input_file)
            mv = load_pcd(input_file)
            object_models[int(mesh_name[-10:-4])] = mv
            print('object mesh: ', int(mesh_name[-10:-4]))

    print('##################')
    print('Loading CNOS segmentations ....')
    with open(config["segmentation_path"]) as handle:
        cnos_seg = json.loads(handle.read())

    #print('cnos len: ', cnos_seg[0])

    print('##################')
    print('Estimating poses ....')
    current_image = 1
    time_global = time.time()
    for sample in dataloader:
        print('Processing test image %i of %i' %(current_image, dataloader.__len__()))
        current_image += 1
        time_template = time.time()

        # sample = {'idx': int(entry[:-4]), 'image': rgb, 'depth': depth, 'scene': int(scene), 'cam_K': scene_cam[str(int(entry[:-4]))]}
        img_id = sample['idx']
        scene_id = sample['scene']
        image = sample['image']
        depth = sample['depth']
        query_K = sample['cam_K']['cam_K']

        # CNOS default segmentations and detections
        # 'scene_id': int, 'image_id': int, 'category_id': int, 'bbox': list, 'score': f, f, 'segmentation': {'counts': list, 'size': list}
        cnos_scene = filter_dict_key(cnos_seg, 'scene_id', scene_id)
        cnos_img = filter_dict_key(cnos_scene, 'image_id', img_id)
        #print('cnos_img: ', len(cnos_img))
        for lp_idx, loc_prior in enumerate(cnos_img):
            prior_cls = loc_prior['category_id']
            prior_bbox = loc_prior['bbox']
            prior_segmentation = loc_prior['segmentation']  # keys: 'counts' and 'size'

            if prior_cls != 1:
                continue

            # something fucky here
            #prior_mask = np.zeros(prior_segmentation['size'][0] * prior_segmentation['size'][1], dtype=np.uint8)
            #print(prior_mask.shape)
            #prior_mask[prior_segmentation['counts']] = 255
            #prior_mask = np.reshape(prior_mask, prior_segmentation['size'])
            #plt.imshow(prior_mask, cmap='gray', vmin=0, vmax=1)
            #plt.show()

            img_vit, img_crop, mask_pp, (y_offset, x_offset) = rectangular_crop_by_factor(image, prior_bbox, extractor, template_size)
            query_scaling = np.max(prior_bbox[2:]) / template_size

            with torch.no_grad():


                # extractor using SD from TOTF model
                #desc_sd, sd_image = pose_estimator.get_SD_features(img_pp)

                # extraction using dino from TOTF model
                #desc_dino = extractor.extract_descriptors(img_pp.to(device), layer=11, facet='key', bin=False,
                #                                          include_cls=True)
                #desc_dino = desc_dino.squeeze(0).squeeze(0).detach().cpu()

                # extraction when only DinoV2 is loaded
                query_desc = extractor.extract_descriptors(img_vit.to(device), layer=11, facet='key', bin=False,
                                                     include_cls=True)
                query_desc = query_desc.squeeze(0).squeeze(0).detach().cpu().numpy()

                matched_templates = utils.find_template_cpu(query_desc, templates_desc, num_results=config['n_hypotheses'])

            # match keypoints
            template_hyps = []
            template_masks = []
            template_cams = []
            template_depths = []
            pose_params = []

            for template_match in matched_templates:
                curr_template = templates_crops[template_match[1]] # .unsqueeze(0)
                image_viz = np.array(copy.deepcopy(image))

                # Visualization template
                temp_vanilla = extractor.revert_preprocess(curr_template.squeeze(0).squeeze(0), load_size=None)
                curr_temp_viz = temp_vanilla

                #print(type(img_crop), type(curr_temp_viz))
                #f, axarr = plt.subplots(1, 2)
                #axarr[0].imshow(img_crop)
                #axarr[1].imshow(curr_temp_viz)
                #plt.imshow(img_crop, curr_temp_viz)
                #plt.show()

                with torch.no_grad():
                    points1, points2, template_pil, crop_pil = extractor.find_correspondences_fastkmeans(
                                img_crop,  # temp_vanilla,
                                temp_vanilla, #img_crop,
                                num_pairs=number_keypoints,
                                load_size=template_size)
                #print('points1: ', points1)
                #print('points2: ', points2)

                #valid_points1 = np.array(points1).astype(np.float64) * query_scaling
                #valid_points1[:, 0] += y_offset
                #valid_points1[:, 1] += x_offset

                #img_crop_viz, scene_img_viz = viz_keypoints(img_crop, temp_vanilla, points1, points2)

                #f, axarr = plt.subplots(1, 2)
                #axarr[0].imshow(img_crop_viz)
                #axarr[1].imshow(scene_img_viz)
                #plt.imshow(img_crop, curr_temp_viz)
                #plt.show()

                template_K = templates_K[template_match[1]]
                template_scaling = template_scales[template_match[1]]
                template_offset = template_offsets[template_match[1]]
                template_bbox = template_bboxes[template_match[1]]

                template_hyps.append(templates_desc[template_match[1]])
                template_masks.append(np.where(np.sum(curr_template, axis=2) > 0.0, 1, 0))
                template_cams.append(template_K)
                template_depths.append(templates_t[template_match[1]])

                # Visualize pose of template
                #R_c_q = np.array(templates_R[template_match[1]]).reshape((3, 3))
                #t_c_q = np.array(templates_t[template_match[1]])
                #colEst = (255, 200, 0)
                #model_vsd = object_models[prior_cls]
                #pts = model_vsd['pts']
                #proj_pts = R_c_q.dot(pts.T).T
                #proj_pts = proj_pts + np.repeat(t_c_q[np.newaxis, :], pts.shape[0], axis=0)
                #proj_pts = vis_utils.toPix_array(proj_pts, query_K[0], query_K[4], query_K[2], query_K[5])
                #proj_pts = proj_pts.astype(np.uint16)
                #proj_pts[:, 0] = np.where(proj_pts[:, 0] >= image_viz.shape[1], image_viz.shape[1] - 1, proj_pts[:, 0])
                #proj_pts[:, 0] = np.where(proj_pts[:, 0] < 0, 0, proj_pts[:, 0])
                #proj_pts[:, 1] = np.where(proj_pts[:, 1] >= image_viz.shape[0], image_viz.shape[0] - 1, proj_pts[:, 1])
                #proj_pts[:, 1] = np.where(proj_pts[:, 1] < 0, 0, proj_pts[:, 1])

                ###############################
                # PnP solver with depth
                R_est, t_est = get_pose_using_pnp(points1, points2, image_viz, templates_t[template_match[1]][2],
                                                                   query_offset=(y_offset, x_offset), template_offset=template_offset, K_q=query_K, K_t=template_K,
                                                                   query_factor=query_scaling, template_scaling=template_scaling)

                initial_pose = np.array(6)
                initial_pose[:3] = cv2.Rodrigues(R_est)
                initial_pose[3:] = t_est
                #print(t_est[2])
                #z = templates_t[template_match[1]][2] * (-t_est[2])
                #y = templates_t[template_match[1]][1] + t_est[1]
                #x = templates_t[template_match[1]][0] + t_est[0]
                #t_est = np.array([x, y, z])
                #t_obj = np.array(templates_t[template_match[1]]) + t_est
                #R_obj = np.linalg.inv(np.array(templates_R[template_match[1]]).reshape((3, 3)))
                ##############################

                ##############################
                # mesh visualization
                #############################
                R_c_q = R_obj
                t_c_q = t_obj
                #colEst = (np.random.randint(25, 240), np.random.randint(25, 240), np.random.randint(25, 240))
                colEst = (255, 223, 0)
                model_vsd = object_models[prior_cls]
                pts = model_vsd['pts']
                proj_pts = R_c_q.dot(pts.T).T
                proj_pts = proj_pts + np.repeat(t_c_q[np.newaxis, :], pts.shape[0], axis=0)
                proj_pts = vis_utils.toPix_array(proj_pts, query_K[0], query_K[4], query_K[2], query_K[5])
                proj_pts = proj_pts.astype(np.uint16)
                proj_pts[:, 0] = np.where(proj_pts[:, 0] >= image_viz.shape[1], image_viz.shape[1]-1, proj_pts[:, 0])
                proj_pts[:, 0] = np.where(proj_pts[:, 0] < 0, 0, proj_pts[:, 0])
                proj_pts[:, 1] = np.where(proj_pts[:, 1] >= image_viz.shape[0], image_viz.shape[0]-1, proj_pts[:, 1])
                proj_pts[:, 1] = np.where(proj_pts[:, 1] < 0, 0, proj_pts[:, 1])
                #image_viz[proj_pts[:, 1], proj_pts[:, 0], :] = colEst
                #plt.imshow(Image.fromarray(image_viz))
                #plt.show()

            #########################################
            # optimize pose using bundle adjustment
            ########################################
            # pts to object origin of templates
            n_templates = config['n_hypotheses']
            # prepare arrays for bundle adjustment
            # Project templates space to 3D
            templates_corners = np.array((n_templates, 4))
            for i_hyp, msk in enumerate(template_masks):
                tmp_K = template_cams[i_hyp]
                z_tmp = template_depths[i_hyp]
                t_height, t_width = msk.shape
                template_corners[i_hyp, 0] = ((0.0 - tmp_K[2]) * z_tmp) / tmp_K[0]
                template_corners[i_hyp, 1] = ((0.0 - tmp_K[3]) * z_tmp) / tmp_K[1]
                template_corners[i_hyp, 2] = ((t_width - tmp_K[2]) * z_tmp) / tmp_K[0]
                template_corners[i_hyp, 3] = ((t_height - tmp_K[3]) * z_tmp) / tmp_K[1]

            # Doing semantic bundle adjustment here

            print("len pose hyps: ", len(pose_params_ba))
            print("pose hyps: ", pose_params_ba)
            #
            trans_refine = featuremetric_bundle_adjustment(template_hyps, template_masks, templates_corners, query_desc, query_K, initial_pose)

            t0 = time.time()
            res = least_squares(mfzs6d_utils.compute_residuals, ba_input, verbose=2, ftol=1e-4, method='lm',
                                args=(points2d_ba, points3d_ba, query_K, image_viz))
            t1 = time.time()

            #############################
            # BOP format export
            ##########################
            '''
            eval_line = []
            sc_id = int(scene_id[0])
            eval_line.append(sc_id)
            im_id = int(image_id)
            eval_line.append(im_id)
            obj_id = int(true_cls)
            eval_line.append(obj_id)
            score = float(scores[odx])
            eval_line.append(score)
            R_bop = [str(i) for i in R_est.flatten().tolist()]
            R_bop = ' '.join(R_bop)
            eval_line.append(R_bop)
            t_bop = t_est * 1000.0
            t_bop = [str(i) for i in t_bop.flatten().tolist()]
            t_bop = ' '.join(t_bop)
            eval_line.append(t_bop)
            time_bop = float(t_img)
            eval_line.append(time_bop)
            eval_img.append(eval_line)
            '''

        print('Testing on img took ', time.time()-time_template)

    #print('Saving results in ', tmp_name = str(template['hemi']) + '_' + str(template['idx']))


    print('######################')
    print('All testing done in ', time.time() - time_global)
