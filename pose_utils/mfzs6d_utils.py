import copy
from pose_utils import img_utils, utils, procrustes
from pose_utils import vis_utils
import open3d
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
from scipy.optimize import least_squares


def cosine_similarity(tensor1, tensor2, mask=None, eps=1e-8):
    """
    Compute cosine similarity along the third dimension of two 3D tensors.

    Arguments:
    tensor1 -- First 3D tensor of shape (batch_size, height, width, channels)
    tensor2 -- Second 3D tensor of shape (batch_size, height, width, channels)
    mask -- Optional mask (same shape as input tensors), 1 where valid, 0 where masked out
    eps -- Small epsilon to avoid division by zero

    Returns:
    cosine_sim -- 2D array of cosine similarities (batch_size, height, width)
    """
    # Compute the L2 norms of each tensor along the last dimension (axis=-1)
    norm_tensor1 = np.linalg.norm(tensor1, axis=-1, keepdims=True)
    norm_tensor2 = np.linalg.norm(tensor2, axis=-1, keepdims=True)

    # Normalize tensors to unit vectors
    normalized_tensor1 = tensor1 / (norm_tensor1 + eps)
    normalized_tensor2 = tensor2 / (norm_tensor2 + eps)

    # Compute dot product along the third dimension
    dot_product = np.sum(normalized_tensor1 * normalized_tensor2, axis=-1)

    # Apply mask if provided (masking out invalid positions)
    if mask is not None:
        dot_product *= mask

    return dot_product

# Function to project 3D points to 2D using the camera matrix and pose (R, t)
def project_points(points_3d, camera_matrix, R, t):
    """
    Project 3D points onto the image plane using the camera intrinsic matrix (K),
    rotation matrix (R), and translation vector (t).
    """

    # Apply rotation and translation
    points_2d_homo = R @ points_3d.T + t[:, np.newaxis]

    # Convert to 2D by dividing by the z component
    points_2d_homo /= points_2d_homo[2]

    # Apply intrinsic camera matrix
    points_2d = camera_matrix @ points_2d_homo[:3, :]

    return points_2d[:2].T


# Define the reprojection error function
def masked_feature_map_error(params, feature_maps_candidates, feature_maps_masks, template_3d, feature_map_query, camera_matrix):
    """
    Compute reprojection error between observed 2D points and the projected 2D points.
    """
    # Extract rotation vector (Rodrigues' format) and translation vector from params
    r_vec = params[:3]  # Rotation vector (Rodrigues format)
    t = params[3:]  # Translation vector

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(r_vec)

    # apply masking to
    feature_maps_observed = []
    for i_hyp, tmp_mask in enumerate(feature_maps_masks):
        x1 = ((template_3d[i_hyp, 0] * camera_matrix[0]) / t[2]) + camera_matrix[2]
        y1 = ((template_3d[i_hyp, 1] * camera_matrix[1]) / t[2]) + camera_matrix[3]
        x2 = ((template_3d[i_hyp, 2] * camera_matrix[0]) / t[2]) + camera_matrix[2]
        y2 = ((template_3d[i_hyp, 3] * camera_matrix[1]) / t[2]) + camera_matrix[3]

        # crop + resize


        # mask



    feature_maps_observed = project_points(feature_map_query, feature_maps_masks, template_3d, camera_matrix, R, t)

    # Compute the error (difference between observed and projected points)
    #error = (points_2d_observed - points_2d_projected).ravel()
    error = (feature_maps_observed - np.array(feature_maps_candidates)).ravel()

    return error


# Bundle adjustment function
def featuremetric_bundle_adjustment(template_maps, template_masks, templates_3d, feature_map_query, camera_matrix, initial_pose):
    """
    Perform bundle adjustment by optimizing the camera pose (rotation and translation)
    while keeping the intrinsic matrix (camera_matrix) fixed.

    Arguments:
    points_3d -- Nx3 array of 3D points
    points_2d_observed -- Nx2 array of observed 2D points
    camera_matrix -- 3x3 camera intrinsic matrix (K)
    initial_pose -- Initial guess for the pose [r_vec (Rodrigues), t]

    Returns:
    optimized_pose -- Optimized camera pose [r_vec, t]
    """
    # Use least squares optimization to minimize reprojection error
    result = least_squares(
        masked_feature_map_error,
        initial_pose,
        args=(template_maps, template_masks, templates_3d, feature_map_query, camera_matrix),
        method='lm'  # Levenberg-Marquardt method
    )

    return result


############### deprecated

def compute_residuals(params, points2d, points3d, template_diffs, cam_K, image=None):
    """Compute residuals.
    points = [cameras, keypoints, coordinates]
    """
    R_g = params[:3, :3]
    t_g = params[:3, 3]
    projections = []
    for idx, trans_t in enumerate(template_diffs):
        #R_t = trans_t[:3, :3]
        #t_t = trans_t[:3, 3]
        #R_r = R_g @ np.linalg.inv(R_t)
        #t_r = t_g - t_t
        print("params: ", params)
        print("trans_t: ", trans_t)
        print("What: ", np.linalg.inv(params) @ trans_t)
        trans_poses = np.linalg.inv(np.linalg.inv(params) @ trans_t)
        R_r = trans_poses[:3, :3]
        t_r = trans_poses[:3, 3]
        proj_pts = R_r.dot(points3d[idx].T).T
        proj_pts = proj_pts + np.repeat(t_r[np.newaxis, :], points3d[idx].shape[0], axis=0)
        projections.append(project_3d_to_2d(proj_pts, cam_K))

        print("projected points: ", proj_pts)
        # viz
        viz_keypoints(image, image2=None, points1=proj_pts, points2=points2d[idx])

    points_proj = np.concatenate(projections, axis=0)
    return (points_proj - points2d).ravel()


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def get_pose_using_gpa(points1, points2, img1, img2, query_offset, template_offset, K_q, K_t, query_factor, template_resize):

    valid_points1 = points1
    valid_points2 = points2

    valid_points1 = np.array(valid_points1).astype(np.float64) * query_factor
    valid_points1[:, 0] += query_offset[0]
    valid_points1[:, 1] += query_offset[1]
    #valid_points1 = expand_2d_keypoint_with_depth(img2, valid_points1, K_q)

    # need to scale template_resize and template_offset due to diverging intrinsics
    fy_d = (K_q[4] / K_t[4])
    fx_d = (K_q[0] / K_t[0])
    valid_points2 = np.array(points2).astype(np.float64) * template_resize
    valid_points2[:, 0] = (valid_points2[:, 0] + template_offset[0]) - K_t[5]
    valid_points2[:, 1] = (valid_points2[:, 1] + template_offset[1]) - K_t[2]
    valid_points2[:, 0] = valid_points2[:, 0] * fy_d
    valid_points2[:, 1] = valid_points2[:, 1] * fx_d
    mu_points2 = np.array(valid_points2).mean(0)
    valid_points2[:, 0] += K_q[5]
    valid_points2[:, 1] += K_q[2]
    #valid_points2[:, [0, 1]] = valid_points2[:, [1, 0]]
    #points_expanded = np.concatenate([valid_points2, np.ones((valid_points2.shape[0], 1))], axis=1)

    #valid_points2[:, [0, 1]] = valid_points2[:, [1, 0]]
    points1_with_depth = []
    points2_with_depth = []
    #for point1, point2 in zip(valid_points1, valid_points2):
    #    if not np.isnan(point1[2]) and point1[2] != 0:
    #        points1_with_depth.append(point1)
    #        points2_with_depth.append(point2)

    #print('valid_points1: ', valid_points1.shape)
    #print('valid_points2: ', valid_points2.shape)

    d, transformed_coords2, tform = procrustes.procrustes(valid_points1, valid_points2)
    R_imspace = tform["rotation"]
    s_imspace = tform["scale"]
    t_imspace = tform["translation"]

    return R_imspace, s_imspace, t_imspace, mu_points2, valid_points1, valid_points2


def get_pose_using_pnp(points1, points2, img1, img2, query_offset, template_offset, K_q, K_t, query_factor, template_scaling):
    # points1 = query in 3D
    # points2 = template keypoints

    # filter valid points
    # sThalham: maybe with depth
    valid_points1 = points1
    valid_points2 = points2

    valid_points1 = np.array(valid_points1).astype(np.float64) * query_factor
    valid_points1[:, 0] += query_offset[0]
    valid_points1[:, 1] += query_offset[1]
    valid_points1[:, [0, 1]] = valid_points1[:, [1, 0]]
    #valid_points1 = expand_2d_keypoint_with_depth(img2, valid_points1, K_q)

    # need to scale template_resize and template_offset due to diverging intrinsics
    fy_d = (K_q[4] / K_t[4])
    fx_d = (K_q[0] / K_t[0])
    valid_points2 = np.array(valid_points2).astype(np.float64) * template_scaling
    valid_points2[:, 0] = (valid_points2[:, 0] + template_offset[0]) - template_K[5]
    valid_points2[:, 1] = (valid_points2[:, 1] + template_offset[1]) - template_K[2]
    valid_points2[:, 0] = valid_points2[:, 0] * fy_d
    valid_points2[:, 1] = valid_points2[:, 1] * fx_d
    valid_points2[:, 0] += query_K[5]
    valid_points2[:, 1] += query_K[2]
    valid_points2[:, [0, 1]] = valid_points2[:, [1, 0]]

    # img_2 assumed to be a single depth value
    points2_expanded = expand_2d_keypoint_with_depth(valid_points1, img2, K_q)
    print('points2 expanded: ', points2_expanded)

    #points1_with_depth = []
    #points2_with_depth = []
    #for point1, point2 in zip(valid_points1, valid_points2):
    ##    if not np.isnan(point1[2]) and point1[2] != 0:
    #        points1_with_depth.append(point1)
    #        points2_with_depth.append(point2)

    #print('points1_with_depth: ', points1_with_depth)
    #print('points2_with_depth: ', points2_with_depth)

    #img_pnp = viz_keypoints(img1, None, points1_with_depth, points2_with_depth)
    #plt.imshow(img_pnp)
    #plt.show()

    try:
        K_PnP = np.array(K_q).reshape((3, 3))
        points_3D = np.ascontiguousarray(points2_expanded).astype(np.float64)[:, np.newaxis, :]
        points_2D = np.ascontiguousarray(valid_points1).astype(np.float64)[:, np.newaxis, :]
        print('query: ', points_3D.shape, type(points_3D.shape))
        print('template: ', points_2D.shape, type(points_2D.shape))
        print('K: ', K_PnP)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, points_2D, K_PnP,
                                                         distCoeffs=None, iterationsCount=100, reprojectionError=8.0)
    except:
        print("Solving PnP failed!")
        return None, None

    R_est, _ = cv2.Rodrigues(rvec)
    t_est = np.squeeze(tvec) # * scale_factor

    return R_est, t_est


def viz_keypoints(image1, image2=None, points1=None, points2=None):

    img1_cp = np.array(copy.deepcopy(image1))
    img2_cp = np.array(copy.deepcopy(image2))
    radius = 4

    if image2 is None:
        for kp in range(len(points1)):
            color = (np.random.randint(25, 240), np.random.randint(25, 240), np.random.randint(25, 240))
            cv2.circle(img1_cp, (int(points1[kp][1]), int(points1[kp][0])), radius, color, thickness=3, lineType=8, shift=0)
            cv2.circle(img1_cp, (int(points2[kp][1]), int(points2[kp][0])), radius, color, thickness=3, lineType=8, shift=0)

        plt.imshow(Image.fromarray(img1_cp))
        plt.show()
        #return img1_cp
    else:
        for kp in range(len(points1)):
            color = (np.random.randint(25, 240), np.random.randint(25, 240), np.random.randint(25, 240))
            cv2.circle(img1_cp, (int(points1[kp][1]), int(points1[kp][0])), radius, color, thickness=3, lineType=8, shift=0)
            cv2.circle(img2_cp, (int(points2[kp][1]), int(points2[kp][0])), radius, color, thickness=3, lineType=8, shift=0)

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img1_cp)
        axarr[1].imshow(img2_cp)
        plt.show()
        #return img1_cp, img2_cp


def project_3d_to_2d(points, cam_K):

    points2d = np.zeros((points.shape[0], 2), dtype=np.float64)

    points2d[:, 0] = ((points[:, 0] * cam_K[0]) / points[:, 2]) + cam_K[2]
    points2d[:, 1] = ((points[:, 1] * cam_K[4]) / points[:, 2]) + cam_K[2]

    return points2d


def project_2d_to_3d_with_depth(keypoints, depth, cam_K):

    # assumes points as [n, [x, y]]
    points_expanded = np.concatenate([keypoints, np.zeros((keypoints.shape[0], 1))], axis=1)
    points_expanded[:, 2] = depth[keypoints[:,0].astype(dtype=np.int32), keypoints[:,1].astype(dtype=np.int32)]
    #print("points expanded: ", points_expanded.shape)
    #print("corresponding depth: ", depth[keypoints[:,0].astype(dtype=np.int32), keypoints[:,1].astype(dtype=np.int32)])
    #print(depth[int(keypoints[0, 0]), int(keypoints[0, 1])])
    #print(depth[int(keypoints[1, 0]), int(keypoints[1, 1])])
    #print(depth[int(keypoints[2, 0]), int(keypoints[2, 1])])

    points_expanded[:, 0] = ((points_expanded[:, 0] - cam_K[2]) * points_expanded[:, 2]) / cam_K[0]
    points_expanded[:, 1] = ((points_expanded[:, 1] - cam_K[2]) * points_expanded[:, 2]) / cam_K[0]

    return points_expanded
