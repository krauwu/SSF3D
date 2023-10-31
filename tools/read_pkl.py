import pickle

def read_pkl():
    with open('/home/songbur/HSSDA/data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/kitti_infos_train.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)

read_pkl()