# StableDino
Stable Diffusion vs Dino


class stabledino:
-  set_paths_and_load_data     (ZS6D, Loader , slight changes)
-  get_SD_features             (Tale of two features, Decriptor extraction, slight changes)
-  compute_pair_feature        (Tale of two features, Part of Keypoint matching , slight changes)
-  co_pca                      (Tale of two features, Part of Keypoint matching , slight changes)
-  perform_clustering          (Tale of two features, Part of Keypoint matching , slight changes)
-  cluster_and_match           (Tale of two features, Part of Keypoint matching , big changes)
-  get_pose_from_sd_cluster    (ZS6D, Part of Keypoint matching , big changes)
-  get_template_sd_knowndesc   (-, Template matching , -)
-  find_template_cpu_sd_cross  (-, Template matching , -)
-  calculate_score             (ZS6D, 15° Metric)

Template Descriptors: totf_prepare_templates_and_gt.py
- Init stabledino (currently Dino + SD)
- get_SD_features
- create UV image
- save sd-descriptor level 2,3,4,5
- save json

Inference Descriptors: totf_extract_desc_dataset.py
- Init stabledino (currently Dino + SD)
- Crop with mask
- get_SD_features
- save sd-descriptor level 2,3,4,5

Process Data (Templatematching + Keypoint Matching): totf_dinoshot_check_angle_single_desc_known.py
(Sorry for the naming, this might be misleading and will be changed in future)
(A lot of dead code inside here)
- Init stabledino (currently Dino + SD)
- set_paths_and_load_data
- get_template_sd_knowndesc
- load and process descriptors (line 164 - 191, mainly loading and matching the needed datastructur)
- compute_pair_feature
- get_pose_from_sd_cluster
- calculate_score
# ZS_SBA
