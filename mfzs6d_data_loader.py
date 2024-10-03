import json
import os
from torch.utils.data import Dataset
from torchvision import datasets
from skimage import io, transform
from matplotlib import pyplot as plt


class HopeBOPDataset(Dataset):

    def __init__(self, config, transform=None):

        self.stage = config["type"]
        if self.stage not in ['onboarding_static', 'onboarding_dynamic', 'test']:
            raise ValueError('Please specify Stage. [\'onboarding_static\', \'onboarding_dynamic\', \'test\']')

        self.path = config["dataset_path"]
        self.object_classes = config["obj_ids"]
        self.transform = transform

        self.samples = []

        if self.stage == 'onboarding_static':
            self.scenes_path = os.path.join(self.path, 'hope/hope_onboarding_static/onboarding_static')
        elif self.stage == 'onboarding_dynamic':
            self.scenes_path = os.path.join(self.path, 'hope/hope_onboarding_dynamic/onboarding_dynamic/')
        elif self.stage == 'test':
            self.scenes_path = os.path.join(self.path, 'hope/hope_test_bop24/test')
            self.segmentations = config["segmentation_path"]

        if self.stage == 'onboarding_static':
            for obj_cls in self.object_classes:
                self.load_specific_obj(obj_cls)

        else:
            for scene in os.listdir(self.scenes_path):
                # HOPE has no gt
                #scene_gt_path = os.path.join(self.scenes_path, scene, 'scene_gt.json')
                scene_cam_path = os.path.join(self.scenes_path, scene, 'scene_camera.json')
                #with open(scene_gt_path) as handle:
                #    scene_gt = json.loads(handle.read())
                with open(scene_cam_path) as handle:
                    scene_cam = json.loads(handle.read())

                # looping over images
                for entry in os.listdir(os.path.join(self.scenes_path, scene, "rgb")):
                    if entry.endswith('.png'):

                        rgb = io.imread(os.path.join(self.scenes_path, scene, "rgb", entry))
                        depth = io.imread(os.path.join(self.scenes_path, scene, "depth", entry))
                        msk = None

                        if self.transform:
                            img = self.transform(img)
                            #msk = self.transform(msk)

                    sample = {'idx': int(entry[:-4]), 'image': rgb, 'depth': depth, 'scene': int(scene), 'cam_K': scene_cam[str(int(entry[:-4]))]}
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def load_specific_obj(self, obj_id):
        scene_up = 'obj_000000'[:-len(str(obj_id))] + str(obj_id) + '_up'
        scene_down = 'obj_000000'[:-len(str(obj_id))] + str(obj_id) + '_down'
        obj_scenes = [scene_up, scene_down]

        hemi = 0
        for scene in obj_scenes:

            scene_gt_path = os.path.join(self.scenes_path, scene, 'scene_gt.json')
            scene_cam_path = os.path.join(self.scenes_path, scene, 'scene_camera.json')
            with open(scene_gt_path) as handle:
                scene_gt = json.loads(handle.read())
            with open(scene_cam_path) as handle:
                scene_cam = json.loads(handle.read())

            for entry in scene_gt:

                msk = io.imread(
                    os.path.join(self.scenes_path, scene, "mask_visib",
                                 '000000'[:-len(entry)] + str(entry) + '_000000.png'))
                rgb = io.imread(os.path.join(self.scenes_path, scene, "rgb", '000000'[:-len(entry)] + str(entry) + '.jpg'))

                #io.imshow(rgb)
                #plt.show()

                if self.transform:
                    img = self.transform(img)
                    msk = self.transform(msk)

                if hemi == 0:
                    hemi_set = 'up'
                else:
                    hemi_set = 'down'
                sample = {'hemi': hemi_set, 'idx': int(entry), 'image': rgb, 'mask': msk, 'obj_id': scene_gt[entry][0]['obj_id'],
                          'R': scene_gt[entry][0]['cam_R_m2c'], 't': scene_gt[entry][0]['cam_t_m2c'],
                          'cam_K': scene_cam[entry]['cam_K'], 'resolution': [scene_cam[entry]['height'], scene_cam[entry]['width']]}
                self.samples.append(sample)
            hemi = 1
