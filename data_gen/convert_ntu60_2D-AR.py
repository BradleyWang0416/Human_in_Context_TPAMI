import os
import pickle
import numpy as np
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from lib.utils.viz_skel_seq import viz_skel_seq_anim

############################# customize #############################
label_stride = 2
############################# customize #############################

source_dataset_path = "data/source_data/NTU60_HRNET.pkl"
label_subset = np.arange(0, 60, label_stride)
if label_stride > 1:
    save_path = f"data/partial_data/NTU60_2DAR/Actions{len(label_subset)}_LabelStride{label_stride}_2DAR"
elif label_stride == 1:
    save_path = f"data/NTU60_2DAR"

def main():
    dataset = read_pkl(source_dataset_path)
    annotations = dataset['annotations']        # annotations: dict list, len=56578
    xsub_train = dataset['split']['xsub_train']
    xsub_val = dataset['split']['xsub_val']

    train_sample_cnt = 0
    test_sample_cnt = 0
    
    for sample_idx, sample in tqdm(enumerate(annotations)):

        label = sample['label']
        if label not in label_subset:
            continue
        
        data = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])  # (1 or 2, T, 17, 2)

        data = human_tracking(data)
        data = coco2h36m(data)      # (1 or 2, T, 17, 2)
        # resample_id = resample(ori_len=sample['total_frames'], target_len=243, randomness=False)
        # data = data[:, resample_id]     # (1 or 2, 243, 17, 2)

        keypoint_score = sample['keypoint_score'][..., None]
        data = np.concatenate((data, keypoint_score), axis=-1)
        # data = np.concatenate((data, np.zeros_like(keypoint_score)), axis=-1)     # (1 or 2, T, 17, 3)
        # data = data.astype(np.float32)      # (1 or 2, T, 17, 3)

        M, T, J, C = data.shape
        data = data - data[0, :, 0, :].reshape(1, T, 1, C)

        viz_seq = {'1st person': data[0], '2nd person': data[1]} if data.shape[0]==2 else {'1st person': data[0]}
        viz_skel_seq_anim(viz_seq, if_print=1, fig_title=f'sample{sample_idx} | label{label}', file_name=f'sample{sample_idx:05d}_label{label:02d}_length{T:03d}', file_folder='tmp/NTU60/frame280-300')


        if sample['frame_dir'] in xsub_train:
            # save_clips(subset_name='train', root_path=save_path, sample=data, label=label, i=train_sample_cnt)
            train_sample_cnt += 1

        elif sample['frame_dir'] in xsub_val:
            # save_clips(subset_name='test', root_path=save_path, sample=data, label=label, i=test_sample_cnt)
            test_sample_cnt += 1

    print(f'Cross subject training sample count: {train_sample_cnt}')
    print(f'Cross subject testing sample count: {test_sample_cnt}')


def resample(ori_len, target_len, replay=False, randomness=False):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def save_clips(subset_name, root_path, sample, label, i):
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_dict = {
        "data_input": sample,   # (1 or 2, T, 17, 3). 每个样本 T 大小不一定相同
        "data_label": label    # 一个属于0到59闭区间的整数
    }
    with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
        pickle.dump(data_dict, myprofile)


def human_tracking(x):
    M, T = x.shape[:2]
    if M==1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0,1:] - x[0,:-1], axis=-1), axis=-1)        # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0,1:] - x[1,:-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        sel = np.cumsum(diff0 > diff1) % 2
        sel = sel[:,None,None]
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0,1:] = x[1,1:] * sel + x[0,1:] * (1-sel)
        x_new[1,1:] = x[0,1:] * sel + x[1,1:] * (1-sel)
        return x_new


def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam


def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y


if __name__ == '__main__':
    main()
