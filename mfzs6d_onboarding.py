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
#import matplotlib.pyplot as plt
#from torchvision import transforms, utils
from skimage import io, transform

from mfzs6d_data_loader import HopeBOPDataset
from totf_StableDino import stabledino
from zs6d_dino2_extractor import PoseViTExtractor

from pose_utils import img_utils

import json
import argparse

def rectangular_crop_by_factor(img, mask, extractor, desc_img_size):
    bbox = img_utils.get_bounding_box_from_mask(mask)
    img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)
    mask_crop, _, _ = img_utils.make_quadratic_crop(mask, bbox)
    img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
    img_crop = Image.fromarray(img_crop)
    
    #img_prep, _, _ = extractor.preprocess(img_crop, load_size=224)
    img_vit, img_sd, _ = extractor.preprocess(img_crop, load_size=desc_img_size)

    return img_vit, img_sd, mask_crop, bbox, (y_offset, x_offset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parse args and allow override of config args

    with open(sys.argv[1]) as handle:
        config = json.loads(handle.read())
    try:
        with open(sys.argv[1]) as handle:
            config = json.loads(handle.read())
    except ValueError as err:
        print("Expected syntax: \"python mf_zs6d_onboarding.py path/to/config/file.json\" ")
        sys.exit()

    external_size = int(512)  # sd size
    external_timestep = int(50)  # sd size
    #external_raw = False if (int(sys.argv[3]) == 0) else True  # raw
    external_raw = False

    # sd
    save_folder = 'obj_' + str(config['obj_ids'][0]) + '_' + str(external_size) + '_' + str(external_timestep)
    # Dino debugging
    save_folder = 'obj_' + str(config['obj_ids'][0]) + '_dino_zs6d'
    template_save_path = os.path.join(config['template_savepath'], save_folder)
    if not os.path.exists(template_save_path):
        os.makedirs(template_save_path)

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
        print('Instantiating TOTF with: image size: %i; time frame: %i' % (external_size, external_timestep))
        pose_estimator = stabledino(use_sd=1, use_dino=1, sd_size=external_size, sd_timestep=external_timestep,
                                    sd_raw=external_raw)
        extractor = pose_estimator.extractor_dino
        desc_img_size = pose_estimator.sd_size
        print('Instantiation of TOTF done')
    elif config["extractor"] == "dino":
        print('For now instantiating DinoV2...')  # because, you know, memory issues on my high-end 2018 Dell Laptop
        extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)
        template_size = config['template_size']
        print('Instantiating DinoV2 done.')

    print('##################')
    print('Encoding templates ....')
    current_template = 1
    time_global = time.time()
    for template in dataloader:
        print('Processing template %i of %i' %(current_template, dataloader.__len__()))
        current_template += 1
        time_template = time.time()
        #io.imshow(template['image'])

        img_vit, img_sd, mask_pp, bbox, (y_offset, x_offset) = rectangular_crop_by_factor(template['image'], template['mask'], extractor, template_size)
        tmp_name = str(template['hemi']) + '_' + str(template['idx'])
        template_scaling = np.max(bbox[2:]) / template_size

        with torch.no_grad():

            # extractor using SD from TOTF model
            #desc_sd, sd_image = pose_estimator.get_SD_features(img_pp)

            # extraction using dino from TOTF model
            #desc_dino = extractor.extract_descriptors(img_pp.to(device), layer=11, facet='key', bin=False,
            #                                          include_cls=True)
            #desc_dino = desc_dino.squeeze(0).squeeze(0).detach().cpu()

            # extraction when only DinoV2 is loaded
            desc = extractor.extract_descriptors(img_vit.to(device), layer=11, facet='key', bin=False,
                                                 include_cls=True)
            desc = desc.squeeze(0).squeeze(0).detach().cpu().numpy()

        #sample = {'hemi': hemi_set, 'idx': int(entry), 'image': rgb, 'mask': msk,
        #          'obj_id': scene_gt[entry][0]['obj_id'],
        #          'R': scene_gt[entry][0]['cam_R_m2c'], 't': scene_gt[entry][0]['cam_t_m2c'],
        #          'cam_K': scene_cam[entry]['cam_K']}
        # Storing template information:
        tmp_dict = {"img_id": int(template['idx']),
                    "obj_id": int(template['obj_id']),
                    "cam_R_m2c": template['R'],
                    "cam_t_m2c": template['t'],
                    "cam_K": template['cam_K'],
                    "img_size": template['resolution'],
                    "bbox": bbox,
                    "template_scaling": template_scaling,
                    "template_offset": (y_offset, x_offset),
                    "img_crop": img_vit,
                    "img_desc": desc
                    }

        save = os.path.join(template_save_path, tmp_name + ".npy")
        np.save(save, tmp_dict)
        #with open(save + '.json', 'w') as fp:
        #    json.dump(tmp_dict, fp)
        print('Encoding and saving took', time.time()-time_template)
    print('######################')
    print('Encoding all templates took: ', time.time() - time_global)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
