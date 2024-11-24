# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)
    
class DataReaderH36M_3D(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_root = dt_root
        self.dt_file = dt_file
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        else:
            trainset = np.concatenate((trainset, np.zeros(trainset.shape)[:,:,0:1]), axis=-1)
            testset = np.concatenate((testset, np.zeros(testset.shape)[:,:,0:1]), axis=-1)
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2
            
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2
            
        return train_labels, test_labels
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_hw(self):
#       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        mean_2d = np.mean(train_data, axis=(0, 1)) # (3,)
        std_2d = np.std(train_data, axis=(0, 1))   # (3,)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        mean_3d = np.mean(train_labels, axis=(0, 1)) # (3,)
        std_3d = np.std(train_labels, axis=(0, 1))   # (3,)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # ipdb.set_trace()
        num_train = train_data.shape[0]
        num_test = test_data.shape[0]
        return train_data, test_data, train_labels, test_labels, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d
    
    def denormalize(self, test_data):
#       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw)
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data # [n_clips, -1, 17, 3]

class DataReaderH36M_MESH(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl',
                 return_3d=True, use_global_orient=True):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_root = dt_root
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.return_3d = return_3d
        self.use_global_orient = use_global_orient
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        else:
            trainset = np.concatenate((trainset, np.zeros(trainset.shape)[:,:,0:1]), axis=-1)
            testset = np.concatenate((testset, np.zeros(testset.shape)[:,:,0:1]), axis=-1)
        return trainset, testset
    
    def read_3d(self):
        train_smpl_pose = self.dt_dataset['train']['smpl_pose'][::self.sample_stride, :]
        train_smpl_shape = self.dt_dataset['train']['smpl_shape'][::self.sample_stride, :]
        test_smpl_pose = self.dt_dataset['test']['smpl_pose'][::self.sample_stride, :]
        test_smpl_shape = self.dt_dataset['test']['smpl_shape'][::self.sample_stride, :]

        from third_party.motionbert.lib.utils.utils_smpl import SMPL
        import torch
        smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)
        train_smpl = smpl(
            betas=torch.from_numpy(train_smpl_shape).float(),
            body_pose=torch.from_numpy(train_smpl_pose).float()[:, 3:],
            global_orient=torch.from_numpy(train_smpl_pose).float()[:, :3] if self.use_global_orient else torch.zeros_like(torch.from_numpy(train_smpl_pose).float()[:, :3]),
            pose2rot=True
        )

        train_verts = train_smpl.vertices.detach()
        J_regressor = smpl.J_regressor_h36m                                    # [17, 6890]
        J_regressor_batch = J_regressor[None, :].expand(train_verts.shape[0], -1, -1).to(train_verts.device)
        train_3d = torch.einsum('bij,bjk->bik', J_regressor_batch, train_verts)
        train_3d = train_3d - train_3d[:, :1, :]                       # chunk_3d: (T,17,3)

        test_smpl = smpl(
            betas=torch.from_numpy(test_smpl_shape).float(),
            body_pose=torch.from_numpy(test_smpl_pose).float()[:, 3:],
            global_orient=torch.from_numpy(test_smpl_pose).float()[:, :3],
            pose2rot=True
        )
        test_verts = test_smpl.vertices.detach()
        J_regressor = smpl.J_regressor_h36m                                    # [17, 6890]
        J_regressor_batch = J_regressor[None, :].expand(test_verts.shape[0], -1, -1).to(test_verts.device)
        test_3d = torch.einsum('bij,bjk->bik', J_regressor_batch, test_verts)
        test_3d = test_3d - test_3d[:, :1, :]                       # chunk_3d: (T,17,3)

        return train_3d.detach().cpu().numpy(), test_3d.detach().cpu().numpy()
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        mean_2d = np.mean(train_data, axis=(0, 1))
        std_2d = np.std(train_data, axis=(0, 1))
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)

        if self.return_3d:
            train_labels, test_labels = self.read_3d()
            mean_3d = np.mean(train_labels, axis=(0, 1))
            std_3d = np.std(train_labels, axis=(0, 1))
            train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]
        else:
            smpl_pose_train = self.dt_dataset['train']['smpl_pose'][::self.sample_stride][split_id_train]                       # (N, T, 72)
            smpl_shape_train = self.dt_dataset['train']['smpl_shape'][::self.sample_stride][split_id_train]                     # (N, T, 10)
            smpl_pose_test = self.dt_dataset['test']['smpl_pose'][::self.sample_stride][split_id_test]                          # (N, T, 72)
            smpl_shape_test = self.dt_dataset['test']['smpl_shape'][::self.sample_stride][split_id_test]                        # (N, T, 10)
            train_labels = {'pose': smpl_pose_train, 'shape': smpl_shape_train}
            test_labels = {'pose': smpl_pose_test, 'shape': smpl_shape_test}
            mean_3d = None
            std_3d = None

        num_train = train_data.shape[0]
        num_test = test_data.shape[0]
        return train_data, test_data, train_labels, test_labels, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d

