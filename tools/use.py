import os
import numpy as np

# os.system("python test.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/pv_rcnn.yaml \
#             --batch_size 4 --ckpt_dir /home/songbur/HSSDA/output/pv_rcnn_ssl/aug_entr/ckpt --eval_all --eval_tag entr_2_16")

# os.system("python test.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/pv_rcnn.yaml \
#             --batch_size 4 --ckpt_dir /home/songbur/HSSDA/output/pv_rcnn_ssl/0.01_1/ckpt --eval_all --eval_tag 0.01_3")

# os.system("python train.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/pv_rcnn_ssl.yaml \
#             --extra_tag 0.01_1 --batch_size 4   \
#             --ckpt  /home/songbur/HSSDA/tools/pv_rcnn_0.01_1.pth \
#             --labeled_frame_idx ../data/kitti/semi_supervised_data_3dioumatch/scene_0.01/1/label_idx.txt"
#              )

# os.system("python train.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/pv_rcnn_ssl.yaml \
#             --extra_tag aug_entr --batch_size 1   \
#             --ckpt  /home/songbur/HSSDA/output/pv_rcnn_ssl/gmm3_last_MOD/ckpt/checkpoint_epoch_159.pth \
#             --labeled_frame_idx ../data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/label_idx.txt"
#             )

# os.system("python train.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/pv_rcnn_ssl.yaml \
#             --extra_tag test --batch_size 1   \
#             --ckpt  /home/songbur/HSSDA/output/pv_rcnn_ssl/gmm3_last_MOD/ckpt/checkpoint_epoch_159.pth \
#             --labeled_frame_idx ../data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/label_idx.txt"
#              )

os.system("python train.py --cfg_file /home/songbur/HSSDA/tools/cfgs/kitti_models/voxel_rcnn_3classes_ssl.yaml \
            --extra_tag VOX_002 --batch_size 2   \
            --ckpt  /home/songbur/HSSDA/002_vox_ini1.pth \
             --labeled_frame_idx ../data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/label_idx.txt"
             )