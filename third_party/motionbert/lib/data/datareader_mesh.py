import numpy as np
import os, sys
import copy
import pickle
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
import train
import torch

class DataReaderMesh(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/mesh', dt_file = 'pw3d_det.pkl', res=[1920, 1920],
                 return_3d=True, use_global_orient=True):
        self.split_id_train = None
        self.split_id_test = None
        self.dt_root = dt_root
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))

        '''
        证明 /home/wxs/Skeleton-in-Context-tpami/data_icl_gen/processed_data/PW3D/mesh_det_pw3d.pkl 的数据帧率是30Hz:
        cnt = {}
        for i in range(len(self.dt_dataset['train']['source'])):
            source = self.dt_dataset['train']['source'][i]
            if source not in tttt:
                tttt[source] = 0
            cnt[source] += 1
        得到的 cnt: 
         {   'courtyard_arguing_000': 765, 'courtyard_backpack_000': 1262, 'courtyard_basketball_000': 418, 'courtyard_bodyScannerMotions_000': 1257, 
                'courtyard_box_000': 1041, 'courtyard_capoeira_000': 388, 'courtyard_captureSelfies_000': 677, 'courtyard_dancing_010': 309, 'courtyard_giveDirections_000': 848, 
                'courtyard_golf_000': 604, 'courtyard_goodNews_000': 431, 'courtyard_jacket_000': 1236, 'courtyard_laceShoe_000': 931, 'courtyard_rangeOfMotions_000': 594, 
                'courtyard_relaxOnBench_000': 546, 'courtyard_relaxOnBench_010': 842, 'courtyard_shakeHands_000': 361, 'courtyard_warmWelcome_000': 562, 'outdoors_climbing_000': 1224, 
                'outdoors_climbing_010': 1061, 'outdoors_climbing_020': 874, 'outdoors_freestyle_000': 498, 'outdoors_slalom_000': 328, 'outdoors_slalom_010': 369, 'courtyard_arguing_001': 765, 
                'courtyard_basketball_001': 468, 'courtyard_capoeira_001': 435, 'courtyard_captureSelfies_001': 693, 'courtyard_dancing_011': 331, 'courtyard_giveDirections_001': 848, 
                'courtyard_goodNews_001': 431, 'courtyard_rangeOfMotions_001': 601, 'courtyard_shakeHands_001': 391, 'courtyard_warmWelcome_001': 343
                }
        又发现3DPW源数据中的 ./sequenceFiles/train/courtyard_arguing_00.pkl 文件的 poses2d 对应765帧, 而 poses_60Hz 对应1530帧
        '''

        # self.pw3d_info = pickle.load(open(, 'rb'), encoding='latin1')
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.res = res
        self.return_3d = return_3d
        self.use_global_orient = use_global_orient
        
    def read_2d(self):
        if self.res is not None:
            res_w, res_h = self.res
            offset = [1, res_h / res_w]
        else:
            res = np.array(self.dt_dataset['train']['img_hw'])[::self.sample_stride].astype(np.float32)
            res_w, res_h = res.max(1)[:, None, None], res.max(1)[:, None, None]
            offset = 1
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)    # [N, 17, 2] 
        # res_w, res_h = self.res
        trainset = trainset / res_w * 2 - offset
        testset = testset / res_w * 2 - offset
        if self.read_confidence:
            train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
            test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
            if len(train_confidence.shape)==2: 
                train_confidence = train_confidence[:,:,None]
                test_confidence = test_confidence[:,:,None]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        else:
            trainset = np.concatenate((trainset, np.zeros_like(trainset[..., :1])), axis=2)
            testset = np.concatenate((testset, np.zeros_like(testset[..., :1])), axis=2)
        return trainset, testset
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                          
        self.split_id_train = split_clips(vid_list_train, self.n_frames, self.data_stride_train)  
        self.split_id_test = split_clips(vid_list_test, self.n_frames, self.data_stride_test)  
        return self.split_id_train, self.split_id_test
    
    def read_3d(self):
        train_smpl_pose = self.dt_dataset['train']['smpl_pose'][::self.sample_stride, :]
        train_smpl_shape = self.dt_dataset['train']['smpl_shape'][::self.sample_stride, :]
        test_smpl_pose = self.dt_dataset['test']['smpl_pose'][::self.sample_stride, :]
        test_smpl_shape = self.dt_dataset['test']['smpl_shape'][::self.sample_stride, :]

        from third_party.motionbert.lib.utils.utils_smpl import SMPL
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
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()                                              # (N, 17, 3)
        mean_2d = np.mean(train_data, axis=(0, 1))
        std_2d = np.std(train_data, axis=(0, 1))
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]        # (n, T, 17, 3)

        if self.return_3d:
            train_labels, test_labels = self.read_3d()
            mean_3d = np.mean(train_labels, axis=(0, 1))
            std_3d = np.std(train_labels, axis=(0, 1))
            train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]
        else:
            smpl_pose_train = self.dt_dataset['train']['smpl_pose'][::self.sample_stride, :][split_id_train]             # (n, T, 72)
            smpl_shape_train = self.dt_dataset['train']['smpl_shape'][::self.sample_stride, :][split_id_train]           # (n, T, 10)
            smpl_pose_test = self.dt_dataset['test']['smpl_pose'][::self.sample_stride, :][split_id_test]
            smpl_shape_test = self.dt_dataset['test']['smpl_shape'][::self.sample_stride, :][split_id_test]
            train_labels = {'pose': smpl_pose_train, 'shape': smpl_shape_train}
            test_labels = {'pose': smpl_pose_test, 'shape': smpl_shape_test}
            mean_3d = None
            std_3d = None
        num_train = len(train_data)
        num_test = len(test_data)
        return train_data, test_data, train_labels, test_labels, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d

    def get_mean_std_2d(self):
        train_2d, _= self.read_2d()
        mean = np.mean(train_2d, axis=(0, 1))
        std = np.std(train_2d, axis=(0, 1))
        return mean, std
    
    def get_mean_std_3d(self):
        train_3d, _ = self.read_3d()
        mean = np.mean(train_3d, axis=(0, 1))
        std = np.std(train_3d, axis=(0, 1))
        return mean, std
    



class DataReaderAMASS_MESH(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/mesh', dt_file = 'pw3d_det.pkl',
                 return_3d=True):
        self.split_id_train = None
        self.split_id_test = None
        self.dt_root = dt_root
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))
        # train:
        #   joint3d: list (len=8080) of numpy array (N,17,3)
        #   vid_list
        #   vid_len_list
        #   smpl_param: list (len=8080) of dict
        #       smpl_param[x]:
        #           'root_orient': (N, 3)
        #           'pose_body': (N, 63)
        #           'pose_hand': (N, 90)
        #           'trans': (N, 3)
        #           'betas': (N, 16)
        #           'dmpls': (N, 8)
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.return_3d = return_3d
    
    def read_3d(self):
        joint3d_train = self.dt_dataset['train']['joint3d']
        joint3d_test = self.dt_dataset['test']['joint3d']
        return np.vstack(joint3d_train), np.vstack(joint3d_test)      # (5728109,17,3)
    
    def read_smpl(self):
        smpl_train = self.dt_dataset['train']['smpl_param']
        smpl_pose_train = []
        for i in range(len(smpl_train)):
            smpl_pose_train_ = np.concatenate([smpl_train[i]['root_orient'].cpu().numpy(),
                                    smpl_train[i]['pose_body'].cpu().numpy(),
                                    smpl_train[i]['pose_hand'].cpu().numpy()], axis=-1)
            smpl_pose_train.append(smpl_pose_train_)
        smpl_pose_train = np.vstack(smpl_pose_train)   # (5728109,156)
        
        smpl_test = self.dt_dataset['test']['smpl_param']
        smpl_pose_test = []
        for i in range(len(smpl_test)):
            smpl_pose_test_ = np.concatenate([smpl_test[i]['root_orient'].cpu().numpy(),
                                    smpl_test[i]['pose_body'].cpu().numpy(),
                                    smpl_test[i]['pose_hand'].cpu().numpy()], axis=-1)
            smpl_pose_test.append(smpl_pose_test_)
        smpl_pose_test = np.vstack(smpl_pose_test)   # (5728109,156)
                        
        return smpl_pose_train, smpl_pose_test


    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['vid_list']
        vid_list_test = self.dt_dataset['test']['vid_list']
        self.split_id_train = split_clips(vid_list_train, self.n_frames, self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):
        split_id_train, split_id_test = self.get_split_id()

        joint3d_train, joint3d_test = self.read_3d()          # (5728109,17,3); (1882446,17,3)
        smpl_pose_train, smpl_pose_test = self.read_smpl()    # (5728109,156); (1882446,156)

        mean_3d = np.mean(joint3d_train, axis=(0,1))
        std_3d = np.std(joint3d_train, axis=(0,1))
        mean_2d, std_2d = mean_3d.copy(), std_3d.copy()
        mean_2d[-1] = 0
        std_2d[-1] = 0

        train_joint3d, test_joint3d = joint3d_train[split_id_train], joint3d_test[split_id_test]            # (89678,32,17,3); (15516,32,17,3)
        train_smplpose, test_smplpose = smpl_pose_train[split_id_train], smpl_pose_test[split_id_test]      # (89678,32,156); (15516,32,156)
        num_train = len(train_joint3d)
        num_test = len(test_joint3d)

        train_joint2d, test_joint2d = train_joint3d[..., :2], test_joint3d[..., :2]
        if self.read_confidence:
            train_confidence = np.ones_like(train_joint2d)[...,0:1]
            test_confidence = np.ones_like(test_joint2d)[...,0:1]
            train_joint2d = np.concatenate((train_joint2d, train_confidence), axis=-1)
            test_joint2d = np.concatenate((test_joint2d, test_confidence), axis=-1)
        else:
            train_confidence = np.zeros_like(train_joint2d)[...,0:1]
            test_confidence = np.zeros_like(test_joint2d)[...,0:1]
            train_joint2d = np.concatenate((train_joint2d, train_confidence), axis=-1)
            test_joint2d = np.concatenate((test_joint2d, test_confidence), axis=-1)
        if self.return_3d:
            return train_joint2d, test_joint2d, train_joint3d, test_joint3d, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d
        else:
            return train_joint2d, test_joint2d, {'pose': train_smplpose, 'shape': None}, {'pose': test_smplpose, 'shape': None}, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d
        