import os
import tqdm
import copy
import shutil
import torch
import pickle
import yaml
import numpy as np
import math
from pandas import DataFrame

from math import log2
from pathlib import Path
from easydict import EasyDict
from sklearn.cluster import KMeans
from collections import defaultdict
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils.box_utils import roiaware_pool3d_utils
from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from tools.visual_utils import open3d_vis_utils as V

from sklearn.mixture import GaussianMixture


def global_rotation_cal(gt_boxes, points, rot):

    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([rot]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([rot]))[0]
    gt_boxes[:, 6] += rot
    gt_boxes = gt_boxes[0]

    return gt_boxes, points


def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    import numpy as np
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


def random_flip_along_x(gt_boxes):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = True  # np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        # points[:, 1] = -points[:, 1]

        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes


def random_flip_along_y(gt_boxes):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = True  # np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        # points[:, 0] = -points[:, 0]

        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes  # , points


def global_rotation(gt_boxes, noise_rotation):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    # points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    # if gt_boxes.shape[1] > 7:
    #     gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
    #         np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
    #         np.array([noise_rotation])
    #     )[0][:, 0:2]

    return gt_boxes  # , points


def global_scaling(gt_boxes, noise_scale):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    # if scale_range[1] - scale_range[0] < 1e-3:
    #     return gt_boxes, points
    # noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    # points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes  # , points


def format_decimal(num):
    a, b = str(num).split('.')
    return float(a + '.' + b[:2])


def get_jnb_threshold(score_list):
    kclf = KMeans(n_clusters=2)
    data_kmeans = np.array(score_list)
    data_kmeans = data_kmeans.reshape(len(data_kmeans), -1)
    kclf.fit(data_kmeans)
    threshod = kclf.cluster_centers_.reshape(-1)
    res = np.sort(threshod)[::-1]
    res = [format_decimal(r) for r in res]
    return res


def get_GMM_threshold(score_list):
    data_kmeans = np.array(score_list)
    init = (np.max(data_kmeans) - np.min(data_kmeans)) / 2
    means_init = [[np.min(data_kmeans)], [np.min(data_kmeans) + init], [np.max(data_kmeans)]]
    weights_init = [1 / 3, 1 / 3, 1/3]
    precisions_init = [[[1.0]], [[1.0]],[[1.0]]]

    gmm = GaussianMixture(n_components=3, weights_init=weights_init,
                          means_init=means_init,
                          precisions_init=precisions_init)
    data_kmeans = data_kmeans.reshape(len(data_kmeans), -1)
    gmm.fit(data_kmeans)
    threshod = gmm.means_.reshape(-1)
    res = np.sort(threshod)[::-1]
    res = [format_decimal(r) for r in res]
    return res


def get_pseudo_label(model, dataloader, rank, cur_epoch, ckpt_save_dir, dist_test=False, result_dir=None):
    # eval one epoch dataset, and save all pred boxes

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    aug_det_annos = []
    x_len = 0.06
    y_len = 0.06
    z_len = 0.06

    for o_batch_dict in tqdm.tqdm(dataloader, desc='get_new_thr'):
        bs = o_batch_dict['batch_size']
        batch_dict = copy.deepcopy(o_batch_dict)
        load_data_to_gpu(batch_dict)
        aug_batch_dict = copy.deepcopy(batch_dict)
        batch_dict['pseudo_label_flag'] = True
        aug_batch_dict['pseudo_label_flag'] = True
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            # global augmentation
            for key in aug_batch_dict.keys():
                if 'aug_' in key:
                    temp_key = key
                    key = key.replace('aug_', '')
                    aug_batch_dict[key] = aug_batch_dict[temp_key]
            aug_pred_dicts, aug_ret_dict = model(aug_batch_dict)

            for frame in range(bs):
                cal_entr_x = []
                cal_entr_y = []
                cal_entr_z = []

                entr_box = copy.deepcopy(pred_dicts[frame]['pred_boxes'].cpu().numpy())
                points = copy.deepcopy(o_batch_dict['points'])
                point = points[points[:, 0] == frame]
                box_mask = points_in_boxes_cpu(point[:, 1:4], entr_box)

                for box_idx in range(box_mask.shape[0]):
                    cur_points = point[box_mask[box_idx] != 0][:, 1:]
                    cur_box = entr_box[box_idx]
                    cur_box = np.expand_dims(cur_box, axis=0)
                    cur_box, cur_points = global_rotation_cal(cur_box, cur_points, -cur_box[0][6])

                    # x_box = np.expand_dims(cur_box, axis=0)
                    # V.draw_scenes(points=cur_points, ref_boxes=x_box)

                    box_list_x = []
                    box_list_y = []
                    box_list_z = []
                    ex = 0
                    ey = 0
                    ez = 0
                    x_layers = int(cur_box[3] / x_len)
                    y_layers = int(cur_box[4] / y_len)
                    z_layers = int(cur_box[5] / z_len)
                    if x_layers != 0:
                        for x_layer in range(x_layers):
                            tmp_x = np.array(
                                [cur_box[0] - cur_box[3] / 2 + (x_layer + 1) * x_len, cur_box[1], cur_box[2],
                                 x_len, cur_box[4], cur_box[5], cur_box[6]])
                            tmp_x = np.expand_dims(tmp_x, axis=0)
                            box_list_x.append(tmp_x)
                    else:
                        cal_entr_x.append(0)
                        print('error')
                        continue

                    if y_layers != 0:
                        for y_layer in range(y_layers):
                            tmp_y = np.array(
                                [cur_box[0], cur_box[1] - cur_box[4] / 2 + (y_layer + 1) * y_len, cur_box[2],
                                 cur_box[3], y_len, cur_box[5], cur_box[6]])
                            tmp_y = np.expand_dims(tmp_y, axis=0)
                            box_list_y.append(tmp_y)
                    else:
                        cal_entr_y.append(0)
                        print('error')
                        continue

                    if z_layers != 0:
                        for z_layer in range(z_layers):
                            tmp_z = np.array(
                                [cur_box[0], cur_box[1], cur_box[2] - cur_box[5] / 2 + (z_layer + 1) * z_len,
                                 cur_box[3], cur_box[4], z_len, cur_box[6]])
                            tmp_z = np.expand_dims(tmp_z, axis=0)
                            box_list_z.append(tmp_z)
                    else:
                        cal_entr_z.append(0)
                        print('error')
                        continue

                    for idx_blx, box_x in enumerate(box_list_x):
                        point_mask_x = points_in_boxes_cpu(cur_points[:, 0:3], box_x)
                        layer_pts = cur_points[point_mask_x[0] != 0]
                        if layer_pts.shape[0] == 0:
                            ex += 0
                        else:
                            pi_x = layer_pts.shape[0] / cur_points.shape[0]
                            entropy = - pi_x * math.log(pi_x)
                            ex += entropy

                    for idx_bly, box_y in enumerate(box_list_y):
                        point_mask_y = points_in_boxes_cpu(cur_points[:, 0:3], box_y)
                        layer_pts = cur_points[point_mask_y[0] != 0]
                        if layer_pts.shape[0] == 0:
                            ey += 0
                        else:
                            pi_y = layer_pts.shape[0] / cur_points.shape[0]
                            entropy = - pi_y * math.log(pi_y)
                            ey += entropy

                    for idx_blz, box_z in enumerate(box_list_z):
                        point_mask_z = points_in_boxes_cpu(cur_points[:, 0:3], box_z)
                        layer_pts = cur_points[point_mask_z[0] != 0]
                        if layer_pts.shape[0] == 0:
                            ez += 0
                        else:
                            pi_z = layer_pts.shape[0] / cur_points.shape[0]
                            entropy = - pi_z * math.log(pi_z)
                            ez += entropy

                    cal_entr_x.append(ex)
                    cal_entr_y.append(ey)
                    cal_entr_z.append(ez)

                pred_dicts[frame]['entr_x'] = cal_entr_x
                pred_dicts[frame]['entr_y'] = cal_entr_y
                pred_dicts[frame]['entr_z'] = cal_entr_z

        #V.draw_scenes(points=point[:, 1:], ref_boxes=entr_box)


        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=None
        )
        for i, ann in enumerate(annos):
            ann['rot'] = batch_dict['rot'][i].cpu().numpy()
            ann['sca'] = batch_dict['sca'][i].cpu().numpy()
            ann['roi_score'] = pred_dicts[i]['roi_scores'].cpu().numpy()

        aug_annos = dataset.generate_prediction_dicts(
            aug_batch_dict, aug_pred_dicts, class_names,
            output_path=None
        )
        for i, ann in enumerate(aug_annos):
            ann['roi_score'] = aug_pred_dicts[i]['roi_scores'].cpu().numpy()

        det_annos += annos
        aug_det_annos += aug_annos

    if dist_test:
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        aug_det_annos = common_utils.merge_results_dist(aug_det_annos, len(dataset), tmpdir=result_dir / 'tmpdir1')

    return det_annos, aug_det_annos


def get_socre_pool(train_anns, test_anns, aug_test_anns):
    all_conf_score = defaultdict(list)
    all_iou_score = defaultdict(list)
    all_roi_score = defaultdict(list)

    # car_list_x = []
    # car_list_y = []
    # car_list_z = []
    # ped_list_x = []
    # ped_list_y = []
    # ped_list_z = []
    # cyc_list_x = []
    # cyc_list_y = []
    # cyc_list_z = []

    class_name = ['Car', 'Pedestrian', 'Cyclist']
    for i in range(len(train_anns)):
        cur_train_anns = train_anns[i]['annos']
        cur_test_anns = test_anns[i]
        cur_aug_test_anns = aug_test_anns[i]

        train_box = cur_train_anns['gt_boxes_lidar']
        test_box = cur_test_anns['boxes_lidar']
        aug_test_box = cur_aug_test_anns['boxes_lidar']

        if len(train_box) == 0 or len(test_box) == 0:
            continue

        # get confident score list
        iou = iou3d_nms_utils.boxes_bev_iou_cpu(train_box, test_box)
        iou_value = np.sum(iou, axis=0)
        paired_train_test_box_flag = iou_value >= 0.5

        test_name = cur_test_anns['name']
        for cat in class_name:
            cat_flag = (test_name == cat)
            # entr_x = np.array(cur_test_anns['entr_x'])[cat_flag]
            # print(entr_x,'hah')
            # entr_y = np.array(cur_test_anns['entr_y'])[cat_flag]
            # entr_z = np.array(cur_test_anns['entr_z'])[cat_flag]
            # if cat == 'Car':
            #     for val in entr_x:
            #         car_list_x.append(val)
            #     for val in entr_y:
            #         car_list_y.append(val)
            #     for val in entr_z:
            #         car_list_z.append(val)
            # if cat == 'Pedestrian':
            #     for val in entr_x:
            #         ped_list_x.append(val)
            #     for val in entr_y:
            #         ped_list_y.append(val)
            #     for val in entr_z:
            #         ped_list_z.append(val)
            # if cat == 'Cyclist':
            #     for val in entr_x:
            #         cyc_list_x.append(val)
            #     for val in entr_y:
            #         cyc_list_y.append(val)
            #     for val in entr_z:
            #         cyc_list_z.append(val)

            conf_score = cur_test_anns['score'][paired_train_test_box_flag & cat_flag]
            roi_score = cur_test_anns['roi_score'][paired_train_test_box_flag & cat_flag]
            all_conf_score[cat].extend(conf_score)
            all_roi_score[cat].extend(roi_score)
        # get iou score list
        rot = cur_test_anns['rot']
        sca = cur_test_anns['sca']

        ori_aug_test_box = random_flip_along_x(test_box)
        ori_aug_test_box = global_rotation(ori_aug_test_box, rot)
        ori_aug_test_box = global_scaling(ori_aug_test_box, sca)

        aug_train_box = random_flip_along_x(train_box)
        aug_train_box = global_rotation(aug_train_box, rot)
        aug_train_box = global_scaling(aug_train_box, sca)

        iou = iou3d_nms_utils.boxes_bev_iou_cpu(aug_train_box, aug_test_box)
        iou_value = np.sum(iou, axis=0)
        paired_aug_train_test_box_flag = iou_value >= 0.5

        paired_iou = iou3d_nms_utils.boxes_bev_iou_cpu(ori_aug_test_box, aug_test_box)
        aug_test_name = cur_aug_test_anns['name']
        for cat in class_name:
            if len(test_name) == 0 or len(aug_test_name) == 0:
                continue
            cat_flag = (test_name == cat) & paired_train_test_box_flag
            aug_cat_flag = (aug_test_name == cat) & paired_aug_train_test_box_flag
            ans = paired_iou[cat_flag]
            ans = ans[:, aug_cat_flag]
            ans = ans.reshape(-1)
            ans = ans[ans > 0]
            all_iou_score[cat].extend(list(ans))

    # df_car = DataFrame({'car_x': car_list_x, 'car_y': car_list_y, 'car_z': car_list_z})
    # df_ped = DataFrame({'ped_x': ped_list_x, 'ped_y': ped_list_y, 'ped_z': ped_list_z})
    # df_cyc = DataFrame({'cyc_x': cyc_list_x, 'cyc_y': cyc_list_y, 'cyc_z': cyc_list_z})
    #
    # df_car.to_excel('last_one_car1.xlsx', index=False)
    # df_ped.to_excel('last_one_ped1.xlsx', index=False)
    # df_cyc.to_excel('last_one_cyc1.xlsx', index=False)

    return all_conf_score, all_iou_score, all_roi_score


def merge_result(test_anns, aug_test_anns, cur_epoch, ckpt_save_dir, root_path, label_frame_idx,
                 init_score_threshold=None):
    train_pkl_root = root_path / 'data' / 'kitti' / 'kitti_infos_train.pkl'
    with open(train_pkl_root, 'rb') as f:
        train_anns = pickle.load(f)
    f.close()

    label_idx = np.loadtxt(label_frame_idx)
    assert len(train_anns) == len(test_anns)
    assert len(train_anns) == len(aug_test_anns)

    frame_idx_2_anns = {}
    for anns in test_anns:
        f_id = anns['frame_id']
        frame_idx_2_anns[f_id] = anns
    test_anns = [frame_idx_2_anns[k] for k in sorted(frame_idx_2_anns.keys())]

    frame_idx_2_anns = {}
    for anns in aug_test_anns:
        f_id = anns['frame_id']
        frame_idx_2_anns[f_id] = anns
    aug_test_anns = [frame_idx_2_anns[k] for k in sorted(frame_idx_2_anns.keys())]

    count = 0

    # conf_high, conf_low, roi_high, roi_low, iou_high, iou_low, x_low,x_high, y_low,y_high, z_low,z_high

    threshold_pool = {'Car': [0.99, 0.99, 0.95, 0.95, 0.93, 0.93, 0, 10, 0, 10, 0, 10],
                      'Pedestrian': [0.98, 0.97, 0.86, 0.91, 0.88, 0.9, 0, 10, 0, 10, 0, 10],
                      'Cyclist': [0.99, 0.99, 0.95, 0.95, 0.93, 0.93, 0, 10, 0, 10, 0, 10]}

    # all_conf_score, all_iou_score, all_roi_score = get_socre_pool_entr(
    #     copy.deepcopy(train_anns), copy.deepcopy(test_anns))

    all_conf_score, all_iou_score, all_roi_score = get_socre_pool(
        copy.deepcopy(train_anns), copy.deepcopy(test_anns), copy.deepcopy(aug_test_anns))


    for category in all_conf_score.keys():
        if category == 'Car':
            if len(all_conf_score[category]) > 2:
                threshold_pool[category][0] = get_GMM_threshold(all_conf_score[category])[0]
                threshold_pool[category][1] = get_GMM_threshold(all_conf_score[category])[1]
            if len(all_roi_score[category]) > 2:
                threshold_pool[category][2] = get_GMM_threshold(all_roi_score[category])[1]
                threshold_pool[category][3] = get_GMM_threshold(all_roi_score[category])[0]
            if len(all_iou_score[category]) > 2:
                threshold_pool[category][4] = get_GMM_threshold(all_iou_score[category])[1]
                threshold_pool[category][5] = get_GMM_threshold(all_iou_score[category])[0]
            threshold_pool[category][6] = 0
            threshold_pool[category][7] = 10
            threshold_pool[category][8] = 1
            threshold_pool[category][9] = 10
            threshold_pool[category][10] = 0
            threshold_pool[category][11] = 10

    if category == 'Pedestrian':
        if len(all_conf_score[category]) > 2:
            threshold_pool[category][0] = get_GMM_threshold(all_conf_score[category])[0] if \
            get_GMM_threshold(all_conf_score[category])[0] > 0.98 else 0.98
            threshold_pool[category][1] = get_GMM_threshold(all_conf_score[category])[1] if \
            get_GMM_threshold(all_conf_score[category])[2] > 0.95 else 0.95
        if len(all_roi_score[category]) > 2:
            threshold_pool[category][2] = get_GMM_threshold(all_roi_score[category])[1] if \
            get_GMM_threshold(all_roi_score[category])[1] > 0.92 else 0.92
            threshold_pool[category][3] = get_GMM_threshold(all_roi_score[category])[0] if \
            get_GMM_threshold(all_roi_score[category])[0] > 0.92 else 0.92
        if len(all_iou_score[category]) > 2:
            threshold_pool[category][4] = get_GMM_threshold(all_iou_score[category])[1] if \
            get_GMM_threshold(all_iou_score[category])[1] > 0.85 else 0.85
            threshold_pool[category][5] = get_GMM_threshold(all_iou_score[category])[0] if \
            get_GMM_threshold(all_iou_score[category])[0] > 0.9 else 0.9

    if category == 'Cyclist':
            if len(all_conf_score[category]) > 2:
                threshold_pool[category][0] = get_GMM_threshold(all_conf_score[category])[0] \
                    if get_GMM_threshold(all_conf_score[category])[0] >= 0.96 else 0.96
                threshold_pool[category][1] = get_GMM_threshold(all_conf_score[category])[1]
            if len(all_roi_score[category]) > 2:
                threshold_pool[category][2] = get_GMM_threshold(all_roi_score[category])[1] if \
                get_GMM_threshold(all_roi_score[category])[1] >= 0.85 else 0.85
                threshold_pool[category][3] = get_GMM_threshold(all_roi_score[category])[0]
            if len(all_iou_score[category]) > 2:
                threshold_pool[category][4] = get_GMM_threshold(all_iou_score[category])[1] if get_GMM_threshold(all_roi_score[category])[1] >= 0.8 else 0.8
                threshold_pool[category][5] = get_GMM_threshold(all_iou_score[category])[0]
            threshold_pool[category][6] = 0
            threshold_pool[category][7] = 10
            threshold_pool[category][8] = 2.1
            threshold_pool[category][9] = 2.8
            threshold_pool[category][10] = 2
            threshold_pool[category][11] = 10

    if init_score_threshold:  # debug
        car_score_thres, car_iou_thres = init_score_threshold[0], init_score_threshold[1]

    for i in tqdm.tqdm(range(len(train_anns)), desc='renew pkl'):
        cur_frame_id = train_anns[i]['point_cloud']['lidar_idx']
        if int(cur_frame_id) in label_idx:
            continue  # labeled scene do not mine pseudo-label

        cur_train_anns = train_anns[i]['annos']
        cur_test_anns = test_anns[i]
        cur_aug_test_anns = aug_test_anns[i]

        train_box = cur_train_anns['gt_boxes_lidar']
        test_box = cur_test_anns['boxes_lidar']
        aug_test_box = cur_aug_test_anns['boxes_lidar']

        if len(train_box) != 0:
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(train_box, test_box)
            iou_value = np.sum(iou, axis=0)

            iou_flag = iou_value == 0
        else:
            iou_flag = np.array([True] * len(test_box))

        if len(test_box) == 0:
            continue

        if len(aug_test_box) != 0:
            rot = cur_test_anns['rot']
            sca = cur_test_anns['sca']
            raw_test_box = copy.deepcopy(test_box)
            cur_aug_box = random_flip_along_x(raw_test_box)
            # cur_aug_box = random_flip_along_y(cur_aug_box)
            cur_aug_box = global_rotation(cur_aug_box, rot)
            cur_aug_box = global_scaling(cur_aug_box, sca)

            iou_aug_test = iou3d_nms_utils.boxes_bev_iou_cpu(cur_aug_box, aug_test_box)
            max_iou_with_test = np.max(iou_aug_test, 1)
        else:
            continue

        names = cur_test_anns['name']
        car_idx = names == 'Car'
        ped_idx = names == 'Pedestrian'
        cyc_idx = names == 'Cyclist'

        car_score_flag = cur_test_anns['score'][car_idx] > threshold_pool['Car'][0]
        ped_score_flag = cur_test_anns['score'][ped_idx] > threshold_pool['Pedestrian'][0]
        cyc_score_flag = cur_test_anns['score'][cyc_idx] > threshold_pool['Cyclist'][0]

        select_flag = np.array([True] * len(test_box))
        select_flag[car_idx] = car_score_flag
        select_flag[ped_idx] = ped_score_flag
        select_flag[cyc_idx] = cyc_score_flag

        roi_flag = np.array([True] * len(test_box))
        roi_score = cur_test_anns['roi_score']
        roi_flag[car_idx] = roi_score[car_idx] > threshold_pool['Car'][2]
        roi_flag[ped_idx] = roi_score[ped_idx] > threshold_pool['Pedestrian'][2]
        roi_flag[cyc_idx] = roi_score[cyc_idx] > threshold_pool['Cyclist'][2]

        aug_flag = np.array([True] * len(test_box))
        aug_flag[car_idx] = max_iou_with_test[car_idx] > threshold_pool['Car'][4]
        aug_flag[ped_idx] = max_iou_with_test[ped_idx] > threshold_pool['Pedestrian'][4]
        aug_flag[cyc_idx] = max_iou_with_test[cyc_idx] > threshold_pool['Cyclist'][4]

        entr_flag = np.array([True] * len(test_box))
        cur_test_anns['entr_x'] = np.array(cur_test_anns['entr_x'])
        cur_test_anns['entr_y'] = np.array(cur_test_anns['entr_y'])
        cur_test_anns['entr_z'] = np.array(cur_test_anns['entr_z'])
        entr_flag[car_idx] = cur_test_anns['entr_x'][car_idx] > threshold_pool['Car'][6]
        entr_flag[car_idx] = entr_flag[car_idx] & (cur_test_anns['entr_x'][car_idx] < threshold_pool['Car'][7])
        entr_flag[car_idx] = entr_flag[car_idx] & (cur_test_anns['entr_y'][car_idx] > threshold_pool['Car'][8])
        entr_flag[car_idx] = entr_flag[car_idx] & (cur_test_anns['entr_y'][car_idx] < threshold_pool['Car'][9])
        entr_flag[car_idx] = entr_flag[car_idx] & (cur_test_anns['entr_z'][car_idx] > threshold_pool['Car'][10])
        entr_flag[car_idx] = entr_flag[car_idx] & (cur_test_anns['entr_z'][car_idx] < threshold_pool['Car'][11])

        entr_flag[ped_idx] = cur_test_anns['entr_x'][ped_idx] > threshold_pool['Pedestrian'][6]
        entr_flag[ped_idx] = entr_flag[ped_idx] & (cur_test_anns['entr_x'][ped_idx] < threshold_pool['Pedestrian'][7])
        entr_flag[ped_idx] = entr_flag[ped_idx] & (cur_test_anns['entr_y'][ped_idx] > threshold_pool['Pedestrian'][8])
        entr_flag[ped_idx] = entr_flag[ped_idx] & (cur_test_anns['entr_y'][ped_idx] < threshold_pool['Pedestrian'][9])
        entr_flag[ped_idx] = entr_flag[ped_idx] & (cur_test_anns['entr_z'][ped_idx] > threshold_pool['Pedestrian'][10])
        entr_flag[ped_idx] = entr_flag[ped_idx] & (cur_test_anns['entr_z'][ped_idx] < threshold_pool['Pedestrian'][11])

        entr_flag[cyc_idx] = cur_test_anns['entr_x'][cyc_idx] > threshold_pool['Cyclist'][6]
        entr_flag[cyc_idx] = entr_flag[cyc_idx] & (cur_test_anns['entr_x'][cyc_idx] < threshold_pool['Cyclist'][7])
        entr_flag[cyc_idx] = entr_flag[cyc_idx] & (cur_test_anns['entr_y'][cyc_idx] > threshold_pool['Cyclist'][8])
        entr_flag[cyc_idx] = entr_flag[cyc_idx] & (cur_test_anns['entr_y'][cyc_idx] < threshold_pool['Cyclist'][9])
        entr_flag[cyc_idx] = entr_flag[cyc_idx] & (cur_test_anns['entr_z'][cyc_idx] > threshold_pool['Cyclist'][10])
        entr_flag[cyc_idx] = entr_flag[cyc_idx] & (cur_test_anns['entr_z'][cyc_idx] < threshold_pool['Cyclist'][11])

        select_flag = select_flag & roi_flag
        select_flag = select_flag & iou_flag
        # select_flag = select_flag & aug_flag
        select_flag = select_flag & entr_flag
        select_flag[ped_idx] = select_flag[ped_idx] & aug_flag[ped_idx]
        # select_flag[cyc_idx] = select_flag[cyc_idx] & aug_flag[cyc_idx]
        trund = np.sum(cur_train_anns['truncated'] > -1)
        out_anns = {}
        count += sum(select_flag)

        if sum(select_flag) > 0:

            select_fg_box = cur_test_anns['boxes_lidar']
            lidar_root = root_path / 'data' / 'kitti' / 'training' / 'velodyne'
            lidar_idx = train_anns[i]['point_cloud']['lidar_idx']
            lidar_path = lidar_root / (lidar_idx + '.bin')
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

            calib_root = root_path / 'data' / 'kitti' / 'training' / 'calib'
            calib_path = calib_root / (lidar_idx + '.txt')
            calib = calibration_kitti.Calibration(calib_path)
            pts_rect = calib.lidar_to_rect(points[:, :3])
            image_shape = train_anns[i]['image']['image_shape']

            fov_flag = get_fov_flag(pts_rect, image_shape, calib)
            points = points[fov_flag]

            points2box = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], select_fg_box)
            num_in_box = np.sum(points2box > 0, axis=1)
            num_flag = num_in_box > 0
            num_flag[ped_idx] = num_in_box[ped_idx] > 5
            num_flag[car_idx] = num_in_box[car_idx] > 5
            num_flag[cyc_idx] = num_in_box[cyc_idx] > 5

            select_flag = select_flag & num_flag

            for key in cur_train_anns:
                if key == 'score':
                    out_anns[key] = np.hstack(
                        (cur_train_anns[key][:trund], np.ones(sum(select_flag)) * -1, cur_train_anns[key][trund:]))
                elif key == 'difficulty':
                    out_anns[key] = np.hstack(
                        (cur_train_anns[key][:trund], np.zeros(sum(select_flag)), cur_train_anns[key][trund:]))
                elif key == 'index':
                    if len(cur_train_anns['index']) != 0:
                        num = max(cur_train_anns['index']) + 1
                    else:
                        num = 1
                    index = np.arange(num, num + sum(select_flag))
                    out_anns[key] = np.hstack((cur_train_anns[key][:trund], index, cur_train_anns[key][trund:]))
                elif key == 'gt_boxes_lidar':
                    out_anns[key] = np.vstack((cur_train_anns[key][:trund], cur_test_anns['boxes_lidar'][select_flag],
                                               cur_train_anns[key][trund:]))
                elif key == 'num_points_in_gt':
                    num_in_box = num_in_box[select_flag]
                    out_anns[key] = np.hstack((cur_train_anns[key][:trund], num_in_box, cur_train_anns[key][trund:]))
                elif key in ['name', 'truncated', 'occluded', 'alpha', 'rotation_y']:
                    out_anns[key] = np.hstack(
                        (cur_train_anns[key][:trund], cur_test_anns[key][select_flag], cur_train_anns[key][trund:]))
                elif key in ['bbox', 'dimensions', 'location']:
                    out_anns[key] = np.vstack(
                        (cur_train_anns[key][:trund], cur_test_anns[key][select_flag], cur_train_anns[key][trund:]))
            train_anns[i]['annos'] = out_anns

    dbinfos_path = root_path / 'data' / 'kitti' / 'kitti_dbinfos_train.pkl'
    train_info_path = root_path / 'data' / 'kitti' / 'kitti_infos_train.pkl'
    gt_path = root_path / 'data' / 'kitti' / 'gt_database'

    os.remove(dbinfos_path)
    os.remove(train_info_path)

    with open(train_info_path, 'wb') as f:
        pickle.dump(train_anns, f)
    f.close()

    shutil.rmtree(gt_path)

    dataset_cfg_root = root_path / 'tools' / 'cfgs' / 'dataset_configs' / 'kitti_dataset.yaml'

    dataset_cfg = EasyDict(yaml.load(open(dataset_cfg_root), Loader=yaml.FullLoader))
    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=root_path / 'data' / 'kitti',
        save_path=root_path / 'data' / 'kitti'
    )

    return threshold_pool


def main():
    print()


if __name__ == '__main__':
    main()

