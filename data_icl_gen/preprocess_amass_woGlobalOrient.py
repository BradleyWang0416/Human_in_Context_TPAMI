import torch
import numpy as np
import os
from os import path as osp
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))
from third_party.motionbert.human_body_prior.body_model.body_model import BodyModel
import copy
import pickle
import pandas as pd

df = pd.read_csv('data_icl_gen/processed_data/AMASS/fps.csv', sep=',',header=None)
fname_list = list(df[0][1:])

processed_dir = 'data_icl_gen/processed_data/AMASS/AMASS_fps60'
J_reg_dir = 'third_party/motionbert/data/AMASS/J_regressor_h36m_correct.npy'
all_motions = 'data_icl_gen/processed_data/AMASS/all_motions_fps60.pkl'

file = open(all_motions, 'rb')
motion_data = pickle.load(file)
J_reg = np.load(J_reg_dir)
all_joints_train = []
all_joints_test = []
vid_list_train = []
vid_list_test = []
vid_len_list_train = []
vid_len_list_test = []
smpl_param_train = []
smpl_param_test = []
gender_train = []
gender_test = []
cnt_train = 0
cnt_test = 0
cnt_global = 0

max_len = 2916
scale_factor = 1
real2cam = np.array([[1, 0, 0], 
                     [0, 0, 1], 
                     [0, -1, 0]], dtype=np.float64)

def gender_to_digit(gender):
    if gender == 'male':
        return 0
    elif gender == 'female':
        return 1

with open('data_icl_gen/processed_data/AMASS/clip_list.csv', 'w') as f:
    print('clip_id, data_split, local_id, fname, clip_len', file=f)
    for i, bdata in enumerate(motion_data):
        if i%200==0:
            print(i, 'seqs done.')
        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subject_gender = bdata['gender']
        if (str(subject_gender) != 'female') and (str(subject_gender) != 'male'):
            subject_gender = 'female'

        bm_fname = osp.join('third_party/motionbert/data/AMASS/body_models/smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = osp.join('third_party/motionbert/data/AMASS/body_models/dmpls/{}/model.npz'.format(subject_gender))

        # number of body parameters
        num_betas = 16
        # number of DMPL parameters
        num_dmpls = 8

        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
        time_length = len(bdata['trans'])
        num_slice = time_length // max_len

        for sid in range(num_slice+1):
            start = sid*max_len
            end = min((sid+1)*max_len, time_length)
            body_parms = {
                'root_orient': torch.zeros_like(torch.Tensor(bdata['poses'][start:end, :3])).to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(bdata['poses'][start:end, 3:66]).to(comp_device), # controls the body
                'pose_hand': torch.Tensor(bdata['poses'][start:end, 66:]).to(comp_device), # controls the finger articulation
                'trans': torch.Tensor(bdata['trans'][start:end]).to(comp_device), # controls the global body position
                'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=(end-start), axis=0)).to(comp_device), # controls the body shape. Body shape is static
                'dmpls': torch.Tensor(bdata['dmpls'][start:end, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
            }
            body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
            mesh = body_trans_root.v.cpu().numpy()
            kpts = np.dot(J_reg, mesh)    # (17,T,3)


            kpts = np.transpose(kpts, (1,0,2))        # (T,17,3)
            kpts = kpts @ real2cam
            kpts *= scale_factor

            dataset_name = fname_list[i].split('_')[1]
            if dataset_name in ['ACCAD','MPI','CMU','Eyes','KIT','EKUT','TotalCapture','TCD']:
                all_joints_train.append(kpts)
                vid_list_train = vid_list_train + [cnt_train] * kpts.shape[0]
                vid_len_list_train.append(kpts.shape[0])
                smpl_param_train.append({k: v.cpu().numpy() for k, v in body_parms.items()})
                gender_train.append(np.array([gender_to_digit(subject_gender) for _ in range(kpts.shape[0])]))
                print(cnt_global, ',', 'train', ',', cnt_train, ',', fname_list[i], ',', end-start, file=f)
                cnt_train += 1
            elif dataset_name in ['BioMotionLab']:
                all_joints_test.append(kpts)
                vid_list_test = vid_list_test + [cnt_test] * kpts.shape[0]
                vid_len_list_test.append(kpts.shape[0])
                smpl_param_test.append({k: v.cpu().numpy() for k, v in body_parms.items()})
                gender_test.append(np.array([gender_to_digit(subject_gender) for _ in range(kpts.shape[0])]))
                print(cnt_global, ',', 'test', ',', cnt_test, ',', fname_list[i], ',', end-start, file=f)
                cnt_test += 1
            else:
                raise ValueError('Unknown dataset name: {}'.format(dataset_name))
            cnt_global += 1
    fileName = open('data_icl_gen/processed_data/AMASS/amass_joints_h36m_60_wSMPL_cpu_woGlobalOrient.pkl','wb')

    all_joints = {
        'train': {
            'joint3d': all_joints_train,
            'vid_list': vid_list_train,
            'vid_len_list': vid_len_list_train,
            'smpl_param': smpl_param_train,
            'gender': gender_train
            },
        
        'test': {
            'joint3d': all_joints_test,
            'vid_list': vid_list_test,
            'vid_len_list': vid_len_list_test,
            'smpl_param': smpl_param_test,
            'gender': gender_test
            }
    }
    pickle.dump(all_joints, fileName)
    print(len(all_joints))
