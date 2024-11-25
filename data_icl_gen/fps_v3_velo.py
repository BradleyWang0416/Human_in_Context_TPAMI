import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import cProfile
import pstats
import time

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def farthest_point_sampling(points, presave_folder, max_len=999999):
    print('max number of points: ', max_len)
    # [46726,32,17,3]
    rest_indices = list(range(points.shape[0]))
    sampled_indices = [0]
    rest_indices.remove(sampled_indices[0])
    start_time = time.time()


    while len(rest_indices) > 0 and len(sampled_indices) < max_len:

        # rest_points = points[rest_indices]  # [rest_num,32,17,3]
        # sampled_points = points[sampled_indices]    # [sampled_num,32,17,3]

        rest_num = len(rest_indices)
        sampled_num = len(sampled_indices)

        rest_batch_size = 2048
        rest_batches_id = [rest_indices[i:i + rest_batch_size] for i in range(0, rest_num, rest_batch_size)]
        sampled_batch_size = 2048
        sampled_batches_id = [sampled_indices[i:i + sampled_batch_size] for i in range(0, sampled_num, sampled_batch_size)]


        MIN_DIST = []
        for rest_batch_id in rest_batches_id:
            rest_batch = points[rest_batch_id]  # [rest_batch,32,17,3]
            rest_velo = rest_batch[:, 1:] - rest_batch[:, :-1]  # [rest_batch,31,17,3]
            rest_velo = torch.norm(rest_velo, dim=-1)  # [rest_batch,31,17]
            rest_velo_avg = rest_velo.mean(-1)  # [rest_batch,31]

            min_dist = torch.full((len(rest_batch),), float('inf')).cuda()   # [rest_batch]

            for sampled_batch_id in sampled_batches_id:
                sampled_batch = points[sampled_batch_id]  # [sampled_batch,32,17,3]
                sampled_velo = sampled_batch[:, 1:] - sampled_batch[:, :-1]  # [sampled_batch,31,17,3]
                sampled_velo = torch.norm(sampled_velo, dim=-1)  # [sampled_batch,31,17]
                sampled_velo_avg = sampled_velo.mean(-1)  # [sampled_batch,31]

                # dist = torch.norm(rest_batch.unsqueeze(1) - sampled_batch.unsqueeze(0), dim=-1)   # [rest_batch, sampled_batch, 32, 17]
                # dist = dist.mean(-1).mean(-1)   # [rest_batch, sampled_batch]

                dist = torch.abs(rest_velo_avg.unsqueeze(1) - sampled_velo_avg.unsqueeze(0))  # [rest_batch, sampled_batch, 31]
                dist = dist.mean(-1)    # [rest_batch, sampled_batch]
                
                min_dist = torch.min(min_dist, dist.min(1)[0])  # [rest_batch]
            MIN_DIST.append(min_dist.cpu().numpy())
        MIN_DIST = np.concatenate(MIN_DIST)  # [rest_num]
        argmax_pos = MIN_DIST.argmax().item()

        ## For double-checking
        # dist_ = torch.norm(rest_points.cuda().unsqueeze(1) - sampled_points.cuda().unsqueeze(0), dim=-1)    # [rest_num, sampled_num, 32, 17]
        # dist_ = dist_.mean(-1).mean(-1).cpu()        # [rest_num, sampled_num]
        # min_dist_ = dist_.min(1)[0]   # [rest_num=1999,]
        # argmax_pos_ = min_dist_.argmax().item()
        
        max_idx = rest_indices[argmax_pos]
        sampled_indices.append(max_idx)
        del rest_indices[argmax_pos]
        if len(sampled_indices) % 50 == 0:
            time_cost = time.time() - start_time
            start_time = time.time()
            print(f'Sampled {len(sampled_indices):d}/{len(points)} points. Cost {time_cost:.2f} sec')

        if len(sampled_indices) % 1000 == 0:
            fps_sorted_indices = np.array(sampled_indices)
            np.save(os.path.join(presave_folder, 'fps_sorted_indices.npy'), fps_sorted_indices)
            # fps_sorted_sliced_data = points[fps_sorted_indices].cpu().numpy().astype(np.float32)
            # np.save(os.path.join(presave_folder, 'fps_sorted_sliced_data.npy'), fps_sorted_sliced_data)
            print(f'Saved')
    
    fps_sorted_indices = np.array(sampled_indices)
    np.save(os.path.join(presave_folder, 'fps_sorted_indices.npy'), fps_sorted_indices)
    # fps_sorted_sliced_data = points[fps_sorted_indices].cpu().numpy().astype(np.float32)
    # np.save(os.path.join(presave_folder, 'fps_sorted_sliced_data.npy'), fps_sorted_sliced_data)
    print(f'Saved at the end')
    
    return sampled_indices

def main():
    # presave_folder = 'data_icl_gen/presave_data/ver5_ICL/AMASS/nframes32 - samplestride1 - datastridetrain128 - datastridetest256 - readconfidence0 - useglobalorient1 - returnskel3d_1 - returnsmpl_1 - filename_AMASS'
    presave_folder = 'data_icl_gen/presave_data/ver5_ICL/H36M_MESH/nframes32 - samplestride1 - datastridetrain16 - datastridetest32 - readconfidence0 - useglobalorient1 - returnskel3d_1 - returnsmpl_1 - filename_mesh_det_h36m_EXTENDED'
    # presave_folder = 'data_icl_gen/presave_data/ver5_ICL/PW3D_MESH/nframes32 - samplestride1 - datastridetrain16 - datastridetest32 - readconfidence0 - useglobalorient1 - returnskel3d_1 - returnsmpl_1 - filename_mesh_det_pw3d_EXTENDED'

    presave_file = os.path.join(presave_folder, 'sliced_data.pkl')  # [2500,22,3]
    with open(presave_file, 'rb') as f:
        sliced_data = pickle.load(f)
    # train:
    #   joint2d: (46726, 32, 17, 3)
    #   joint3d: (46726, 32, 17, 3)
    #   smpl_pose: (46726, 32, 72)
    #   smpl_shape: (46726, 32, 10)
    seq3d = sliced_data['train']['joint3d']
    seq3d = torch.from_numpy(seq3d).cuda()         # [46726,32,17,3]

    sampled_indices = farthest_point_sampling(seq3d, presave_folder)

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()