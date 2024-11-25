#%%
import os
import argparse

from requests import head
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='ckpt/default', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    # parser.add_argument('-v', '--visualize', action='store_true', help='whether to activate visualization')
    parser.add_argument('-v', '--visualize', default='', type=str, help='whether to activate visualization')
    parser.add_argument('-g', '--eval_generalization', action='store_true', help='whether to evaluate generalization ability')
    parser.add_argument('-debug', '--quick_debug', action='store_true', help='whether to quickly debug')
    parser.add_argument('-gpu', default='0', type=str, help='assign the gpu(s) to use')
    parser.add_argument('-bs', '--batch_size', default=None, type=int, help='batch size')
    parser.add_argument('-clip_len', default=None, type=int, help='clip length')
    parser.add_argument('-max_len', default=None, type=int, help='max length')
    parser.add_argument('-epochs', default=None, type=int, help='number of epochs')
    parser.add_argument('-no_eval', default=None, type=int)
    parser.add_argument('-stage', default='', type=str, help='stage')
    parser.add_argument('-classifier_type', default='task', type=str)
    parser.add_argument('-num_class', default=4, type=int)
    parser.add_argument('-use_presave_data', default=None, type=int)
    parser.add_argument('-normalize_2d', default=None, type=int)
    parser.add_argument('-normalize_3d', default=None, type=int)
    parser.add_argument('-aug', default=None, type=int)
    parser.add_argument('-aug_shuffle_joints', default=None, type=int)
    parser.add_argument('-use_task_id_as_prompt', action='store_true')
    parser.add_argument('-dumb_task', default=None, type=str, metavar='LIST', help='dumb task')
    parser.add_argument('-reverse_query_prompt', action='store_true')
    parser.add_argument('-shuffle_batch', action='store_true')
    parser.add_argument('-fix_prompt', default='', type=str)
    parser.add_argument('-reverse_query_prompt_per_iter', action='store_true')
    parser.add_argument('-aug2D', action='store_true')
    parser.add_argument('-out', default='', type=str)
    parser.add_argument('-deactivate_prompt_branch', action='store_true')
    parser.add_argument('-gpu0_bs', default=0, type=int)
    parser.add_argument('-train_simultaneously', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('-mp', '--master_port', type=str, default='12356', help='master port for DistributedDataParallel')
    parser.add_argument('-use_context', type=str, default='')

    parser.add_argument('-apply_attnmask', action='store_true')
    parser.add_argument('-data_efficient', action='store_true')
    parser.add_argument('-vertex_x1000', action='store_true')
    parser.add_argument('-fully_connected_graph', action='store_true')
    opts = parser.parse_args()
    # opts, _ = parser.parse_known_args()       # 在ipynb中要用这行
    return opts

opts = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

import sys
import numpy as np
import shutil
import errno
import tensorboardX
from time import time
import random
import prettytable
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import cycle
from collections import Counter
import importlib
from functools import partial
from datetime import datetime
import pytz
import pandas as pd

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.dataset_2DAR import ActionRecognitionDataset2D, get_AR_labels, collate_fn_2DAR
from lib.model.loss import *
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.data_parallel import BalancedDataParallel

from funcs_and_classes.Non_AR.eval_generalization.ver0 import evaluate_generalization

def import_class(class_name):
    mod_str, _sep, class_str = class_name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def import_function(func_name=None):
    """
    动态导入指定的函数。
    
    参数:
    - func_name: 一个字符串，表示函数的全限定名，如 "mymodule.my_function"
    
    返回:
    - 导入的函数对象
    """    
    # 分割模块名和函数名
    module_name, func_name = func_name.rsplit('.', 1)
    
    # 动态导入模块
    module = importlib.import_module(module_name)
    
    # 获取函数对象
    func = getattr(module, func_name)
    
    return func


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, eval_dict, no_print=False):
    if not no_print:
        print('\tSaving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'eval_dict' : eval_dict
    }, chk_path)


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_with_config(rank, world_size, args, opts):
    setup(rank, world_size, opts.master_port)

    # Import specified classes and functions
    if 'AR' in args.tasks:
        ## dataset AR
        dataset_action_recognition_VER = args.func_ver.get('dataset_action_recognition', 1)
        dataset_action_recognition = import_class(class_name=f'funcs_and_classes.AR.dataset_AR.ver{dataset_action_recognition_VER}.Dataset_ActionRecognition')
        ## evaluate AR
        evaluate_action_recognition_VER = args.func_ver.get('evaluate_action_recognition', 2)
        evaluate_action_recognition = import_function(func_name=f'funcs_and_classes.AR.eval_AR.ver{evaluate_action_recognition_VER}.evaluate_action_recognition')
        ## train epoch AR
        train_epoch_action_recognition_VER = args.func_ver.get('train_epoch_action_recognition', 2)
        train_epoch_action_recognition = import_function(func_name=f'funcs_and_classes.AR.train_epoch.ver{train_epoch_action_recognition_VER}.train_epoch')

    if len([task for task in args.tasks if task not in ['AR']]) > 0:
        ## dataset non-AR
        dataset_VER = args.func_ver.get('dataset_non_AR', 1)
        DATASET = import_class(class_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.MotionDatasetICL')
        ## evaluate non-AR
        evaluate_VER = args.func_ver.get('evaluate_non_AR', 1)
        if evaluate_VER in ['2_ICL', '3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
            evaluate = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate')
            if opts.stage == 'classifier':
                evaluate = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_classifier')
        else:
            evaluate_future_pose_estimation = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_future_pose_estimation')
            evaluate_motion_completion = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_motion_completion')
            evaluate_motion_prediction = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_motion_prediction')
            evaluate_pose_estimation = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_pose_estimation')
        ## train epoch non-AR
        train_epoch_VER = args.func_ver.get('train_epoch_non_AR', 1)
        train_epoch = import_function(func_name=f'funcs_and_classes.Non_AR.train_epoch.ver{train_epoch_VER}.train_epoch')
        if opts.stage == 'classifier':
            train_epoch = import_function(func_name=f'funcs_and_classes.Non_AR.train_epoch.ver{train_epoch_VER}.train_classifier_epoch')
        
    ## model name
    model_name = args.func_ver.get('model_name', 'M00_SiC_dynamicTUP')
    if opts.stage == 'classifier':
        model_name = model_name + '_classifier'
    try: 
        model_class = import_class(class_name=f'lib.model.{model_name}.Skeleton_in_Context')
    except ModuleNotFoundError:
        model_class = import_class(class_name=f'lib.model.model_old.{model_name}')
    
    if hasattr(model_class, 'prepare_motion'):
        setattr(args, 'prepare_motion_function', model_class.prepare_motion)
        if rank == 0: print(f'\nOverriding function... Using <prepare_motion> function from [{model_name}] instead of default from dataset ...')
    if hasattr(model_class, 'preprocess'):
        setattr(args, 'preprocess_function', model_class.preprocess)
        if rank == 0: print(f'\nOverriding function... Using <preprocess> function from [{model_name}] instead of default from dataset ...')
    


    train_writer = None
    if rank == 0:
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    if rank == 0:  
        print('\nLoading dataset...')
    trainloader_params = {
        #   'batch_size': args.batch_size // world_size ,
          'batch_size': args.batch_size,
          'num_workers': 0,
          'pin_memory': True,
        #   'prefetch_factor': 4,
        #   'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.test_batch_size,
          'num_workers': 0,
          'pin_memory': True,
        #   'prefetch_factor': 4,
        #   'persistent_workers': True
    }


    train_loader = {}

    if len([task for task in args.tasks if task not in ['AR']]) > 0:
        if dataset_VER in ['3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL']:
            train_dataset = DATASET(args, data_split='train', rank=rank)
            try:
                collate_func = import_function(func_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.collate_func')
                trainloader_params.update({'collate_fn': collate_func})
            except:
                if rank == 0: print(f'\tNo customized <collate_func> found for ver{dataset_VER}. Use default collate_fn.')
        elif dataset_VER in ['8_ICL']:
            train_dataset = DATASET(args, data_split='train', rank=rank)
            try:
                collate_func = import_function(func_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.collate_func_train')
                trainloader_params.update({'collate_fn': collate_func})
            except:
                if rank == 0: print(f'\tNo customized <collate_func> found for ver{dataset_VER}. Use default collate_fn.')
        else:
            train_dataset = DATASET(args, data_split='train')
        if rank == 0: print('\tTraining (non-AR) sample count:', len(train_dataset))
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader['non_AR'] = DataLoader(train_dataset, sampler=train_sampler, **trainloader_params)

        if dataset_VER in [0]:
            test_dataset = DATASET(args, data_split='test', prompt_list=train_dataset.prompt_list)
            if rank == 0: print('\tTesting (non-AR) sample count:', len(test_dataset))
        elif dataset_VER in [1]:
            test_dataset = DATASET(args, data_split='test')
            if rank == 0: print('\tTesting (non-AR) sample count:', len(test_dataset))
        elif dataset_VER in ['2_ICL']:
            test_dataset = {}
            dataset_task_info_test = args.dataset_task_info['test']
            for dataset_name in dataset_task_info_test:
                for eval_task in dataset_task_info_test[dataset_name]:
                    test_dataset[(dataset, eval_task)] = DATASET(args, data_split='test', prompt_list=train_dataset.prompt_list, TASK=eval_task, DATASET=dataset)
        elif dataset_VER in ['3_ICL']:
            test_dataset = {}
            dataset_task_info_test = args.dataset_task_info['test']
            for dataset_name in dataset_task_info_test:
                for eval_task in dataset_task_info_test[dataset_name]:
                    test_dataset[(dataset_name, eval_task)] = DATASET(args, data_split='test', TASK=eval_task, DATASET_NAME=dataset_name)
        elif dataset_VER in ['4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
            test_dataset = {}
            dataset_task_info_test = args.dataset_task_info['test']
            for dataset_name in dataset_task_info_test:
                for eval_task in dataset_task_info_test[dataset_name]:
                    if dataset_name not in train_dataset.sliced_data_dict:
                        train_dataset.sliced_data_dict[dataset_name] = None
                    test_dataset_params = {
                        'data_split': 'test', 
                        'TASK': eval_task, 
                        'DATASET_NAME': dataset_name, 
                        'SLICED_DATA': train_dataset.sliced_data_dict[dataset_name],
                        'rank': rank
                    }
                    if hasattr(train_dataset, 'prompt_log'):
                        test_dataset_params['PROMPT_LOG'] = train_dataset.prompt_log
                    test_dataset[(dataset_name, eval_task)] = DATASET(args, **test_dataset_params)


    ################################################### AR ###################################################
    if 'AR' in args.tasks:
        train_dataset_AR = dataset_action_recognition(**args.train_feeder_args_ntu)
        if rank == 0:
            print(f'Training (AR) sample count: {len(train_dataset_AR)}')
        train_loader_params_AR = {
            'dataset': train_dataset_AR // world_size,
            'batch_size': args.batch_size_ntu,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': False  # Shuffle should be False when using DistributedSampler
        }
        if dataset_action_recognition_VER in [1,2,3,4]:
            collate_func = import_function(func_name=f'funcs_and_classes.AR.dataset_AR.ver{dataset_action_recognition_VER}.collate_func')
            train_loader_params_AR.update({'collate_fn': collate_func})

        train_sampler_AR = DistributedSampler(train_dataset_AR, num_replicas=world_size, rank=rank)
        train_loader['AR'] = DataLoader(train_dataset_AR, sampler=train_sampler_AR, **train_loader_params_AR)

        test_dataset_AR = dataset_action_recognition(prompt_data=train_dataset_AR.data, prompt_label=train_dataset_AR.label, **args.test_feeder_args_ntu)
        if rank == 0:
            print(f'Testing (AR) sample count: {len(test_dataset_AR)}')
        test_loader_params_AR = {
            'dataset': test_dataset_AR,
            'batch_size': args.test_batch_size_ntu,
            'num_workers': 32,
            'pin_memory': True,
            'shuffle': False  # Shuffle should be False when using DistributedSampler
        }
        if dataset_action_recognition_VER in [1,2,3]:
            test_loader_params_AR.update({'collate_fn': collate_func})
        elif dataset_action_recognition_VER in [4]:
            collate_func_test = import_function(func_name=f'funcs_and_classes.AR.dataset_AR.ver{dataset_action_recognition_VER}.collate_func_test')
            test_loader_params_AR.update({'collate_fn': collate_func_test})

        test_sampler_AR = DistributedSampler(test_dataset_AR, num_replicas=world_size, rank=rank)
        test_loader_AR = DataLoader(test_dataset_AR, sampler=test_sampler_AR, **test_loader_params_AR)
    ################################################### AR ###################################################

    test_loader = {}
    if dataset_VER in ['2_ICL']:
        for (dataset, task) in test_dataset.keys():
            sampler = DistributedSampler(test_dataset[(dataset, task)], num_replicas=world_size, rank=rank)
            test_loader[(dataset, task)] = DataLoader(test_dataset[(dataset, task)], sampler=sampler, **testloader_params)
    elif dataset_VER in ['3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL']:
        try:
            collate_func = import_function(func_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.collate_func')
            testloader_params.update({'collate_fn': collate_func})
        except:
            if rank == 0: print(f'\tNo customized <collate_func> found for ver{dataset_VER}. Use default collate_fn.')
        for (dataset, task) in test_dataset.keys():
            sampler = DistributedSampler(test_dataset[(dataset, task)], num_replicas=world_size, rank=rank)
            test_loader[(dataset, task)] = DataLoader(test_dataset[(dataset, task)], sampler=sampler, **testloader_params)
    elif dataset_VER in ['8_ICL']:
        try:
            collate_func = import_function(func_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.collate_func_test')
            testloader_params.update({'collate_fn': collate_func})
        except:
            if rank == 0: print(f'\tNo customized <collate_func> found for ver{dataset_VER}. Use default collate_fn.')
        for (dataset, task) in test_dataset.keys():
            sampler = DistributedSampler(test_dataset[(dataset, task)], num_replicas=world_size, rank=rank)
            test_loader[(dataset, task)] = DataLoader(test_dataset[(dataset, task)], sampler=sampler, **testloader_params)
    else: # TODO: DDP modified
        for task in args.tasks:
            if task == 'AR':
                test_loader[task] = DataLoader(**test_loader_params_AR)
            else:
                test_loader[task] = DataLoader(Subset(test_dataset, test_dataset.global_idx_list[task]), **testloader_params) 

    eval_dict = {}
    if dataset_VER in ['2_ICL', '3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
        for (dataset, eval_task) in test_loader.keys():
            if task in ['AR', '2DAR']:
                eval_dict[(dataset, eval_task)] = {'max_acc': 0, 'best_epoch': 1000000}
            else:
                eval_dict[(dataset, eval_task)] = {'min_err': 1000000, 'best_epoch': 1000000}
    else:
        for task in args.tasks:
            if task in ['AR', '2DAR']:
                eval_dict[task] = {'max_acc': 0, 'best_epoch': 1000000}
            else:
                eval_dict[task] = {'min_err': 1000000, 'best_epoch': 1000000}
    eval_dict['all'] = {'min_err': 1000000, 'best_epoch': 1000000}


    if dataset_VER in ['3_ICL', '4_ICL']:
        datareader_pose_estimation = None
        if 'H36M_3D' in args.dataset_task_info['test'] and 'PE' in args.dataset_task_info['test']['H36M_3D']:
            datareader_pose_estimation = DataReaderH36M(n_frames=args.clip_len*2, sample_stride=args.dataset_config['H36M_3D']['sample_stride'],
                                                            data_stride_train=args.dataset_config['H36M_3D']['data_stride_train'], data_stride_test=args.dataset_config['H36M_3D']['data_stride_test'], 
                                                            dt_root='', dt_file=args.dataset_file['H36M_3D'])
    elif dataset_VER in ['5_ICL', '6_ICL', '7_ICL', '8_ICL']:
        datareader_pose_estimation = None
        if 'H36M_3D' in args.dataset_task_info['test'] and 'PE' in args.dataset_task_info['test']['H36M_3D']:
            n_frames = args.dataset_config[dataset_name].get('clip_len', args.clip_len) * 2
            datareader_pose_estimation = DataReaderH36M(n_frames=n_frames, sample_stride=args.dataset_config['H36M_3D']['sample_stride'],
                                                        data_stride_train=args.dataset_config['H36M_3D']['data_stride']['train'], data_stride_test=args.dataset_config['H36M_3D']['data_stride']['test'], 
                                                        dt_root='', dt_file=args.dataset_file['H36M_3D'])
    elif 'PE' in args.tasks:
        datareader_pose_estimation = DataReaderH36M(n_frames=args.data.clip_len, sample_stride=args.data.sample_stride, 
                                                            data_stride_train=args.data.train_stride, data_stride_test=args.data.clip_len, 
                                                            dt_root=args.data.root_path, dt_file=args.data.source_file_h36m)

    if rank == 0:
        print('\nLoading model...')
        print(f'\tModel: {model_name}')

    if 'model_mesh' in model_name:
        model_config = {
            'args': args,
            'num_frame': args.clip_len,
            'num_joints': 17,
            'in_chans': 3,
            'hidden_dim': 512,
            'depth': 8,
            'num_heads': 8,
            'mlp_ratio': 2,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0.1,
            'norm_layer': partial(nn.LayerNorm, eps=1e-6)
        }
    elif 'MixSTE' in model_name:
        model_config = {
            "num_frame": args.maxlen,
            "num_joints": args.data.num_joints,
            "in_chans": 3,
            "embed_dim_ratio": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 2,

            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0,
            "attn_drop_rate": 0,
            "drop_path_rate": 0.1,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "is_train": True,

            "prompt_enabled": True,
            "sqrt_alphas_cumprod": None,
            "sqrt_one_minus_alphas_cumprod": None,

            "prompt_gt_as_condition": False,
            "use_text": False,
            "fuse_prompt_query": 'add'
        }
    elif 'TTT' in model_name:
        model_config = {
            "num_frame": args.maxlen,
            "num_joints": 17,
            "in_chans": 3,
            "embed_dim_ratio": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 2,

            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0,
            "attn_drop_rate": 0,
            "drop_path_rate": 0.1,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        }
    else:
        model_config = {
            "args": args, 
            "dim_in": args.dim_in,
            "dim_out": args.dim_out,
            "dim_feat": args.dim_feat,
            "dim_rep": args.dim_rep,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "maxlen": args.maxlen,
            "num_joints": args.data.num_joints
        }

    if opts.stage == 'classifier':
        model_config.update({'num_class': opts.num_class})
    if 'maxlen' in model_name:
        model_config.update({'max_clip_len': args.max_len})
        assert args.max_len >= args.clip_len
    model_backbone = model_class(**model_config)

    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    if rank == 0:
        print(f'\tTrainable parameter count: {model_params/1000000}M')

    # if torch.cuda.is_available(): # 暂时弃用
    #     if args.gpu0_bs > 0:
    #         model_backbone = BalancedDataParallel(args.gpu0_bs, model_backbone)
    #     else:
    #         model_backbone = nn.DataParallel(model_backbone)
    #     model_backbone = model_backbone.cuda()
    
    model_backbone = model_backbone.to(rank)
    model_backbone = nn.parallel.DistributedDataParallel(model_backbone, device_ids=[rank], find_unused_parameters=True)

    chk_filename = os.path.join(opts.checkpoint, "latest_epoch", "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        if rank == 0:
            print('\nLoading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=f'cuda:{rank}')
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone


    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0

        if rank == 0:
            print(f'\nTraining on {[[key, len(loader)] for key, loader in train_loader.items()]} batches for {args.epochs} epochs. batch size: {args.batch_size}')

        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                if rank == 0:
                    print('\nWARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'eval_dict' in checkpoint and checkpoint['eval_dict'] is not None:
                eval_dict = checkpoint['eval_dict']
            else:
                if rank == 0:
                    print('\nWARNING: this checkpoint does not contain <eval_dict>. The <eval_dict> will be reinitialized.')

        summary_table = prettytable.PrettyTable()
        if dataset_VER in ['2_ICL', '3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
            summary_table.field_names = ['Epoch'] + [f'{dataset} | {eval_task}' for (dataset, eval_task) in test_loader.keys()]
        else:
            summary_table.field_names = ['Epoch'] + [metric for task in args.tasks for metric in args.task_metrics[task]]

        training_start_time = time()
        epoch_to_eval = [ep for ep in range(0, st + 10, 1)] + \
                        [ep for ep in range(st + 10, args.epochs - 20, 2)] + \
                        [ep for ep in range(args.epochs - 20, args.epochs, 2)]
        for epoch in range(st, args.epochs):
            train_sampler.set_epoch(epoch)
            if rank == 0:
                print(f'[{epoch + 1} start]')
            start_time = time()

            if train_epoch_VER in ['5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                losses = {'joint': {}}
                for loss_name, loss_weight in args.losses.items():
                    if loss_weight == 0:
                        continue
                    losses['joint'].update({loss_name: {'loss_logger': AverageMeter(),
                                                        'loss_weight': loss_weight,
                                                        'loss_function': globals()[loss_name]}})
                losses.update({'joint_total': AverageMeter()})

                use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
                if use_smpl:
                    from third_party.motionbert.lib.model.loss_mesh import MeshLoss
                    criterion_mesh = MeshLoss(loss_type=args.Mesh.loss_type)
                    losses.update({'mesh_criterion': criterion_mesh})
                    losses.update({'mesh': {}})
                    for loss_name, loss_weight in args.Mesh.losses.items():
                        if loss_weight == 0:
                            continue
                        losses['mesh'].update({loss_name: {'loss_logger': AverageMeter(),
                                                           'loss_weight': loss_weight}})
                    losses.update({'mesh_total': AverageMeter()})
                    losses.update({'all_total': AverageMeter()})
            else:
                losses = {}
                for loss_name, loss_weight in args.losses.items():
                    if loss_name == 'mpjpe':
                        loss_function = loss_mpjpe
                    elif loss_name == 'n_mpjpe':
                        loss_function = n_mpjpe
                    elif loss_name == 'velocity':
                        loss_function = loss_velocity
                    elif loss_name == 'limb_var':
                        loss_function = loss_limb_var
                    elif loss_name == 'limb_gt':
                        loss_function = loss_limb_gt
                    elif loss_name == 'angle':
                        loss_function = loss_angle
                    elif loss_name == 'angle_velocity':
                        loss_function = loss_angle_velocity
                    else:
                        raise ValueError('Unknown loss type.')
                    losses[loss_name] = {'loss_logger': AverageMeter(),
                                         'loss_weight': loss_weight,
                                         'loss_function': loss_function}
                losses['total'] = {'loss_logger': AverageMeter()}
            if opts.stage == 'classifier':
                losses = {
                    'mpjpe': {'loss_logger': AverageMeter(), 'loss_weight': 1, 'loss_function': nn.CrossEntropyLoss()},
                    'total': {'loss_logger': AverageMeter()}
                }

            if 'AR' in train_loader.keys():
                train_epoch_action_recognition(args, model_pos, train_loader, losses, optimizer, epoch=epoch, if_viz=opts.visualize)
            if 'non_AR' in train_loader.keys():
                if opts.stage == 'classifier':
                    train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug, classifier_type=opts.classifier_type)
                    evaluate(args, test_loader, model_pos, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug, classifier_type=opts.classifier_type)
                else:
                    train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug, rank=rank)


            # Post-epoch evaluation and loss logger update
            if rank == 0: #只在0号卡上进行测试、记录
                if args.no_eval:
                    elapsed = (time() - start_time) / 60
                    print(f"[{epoch+1} end] Time cost: {elapsed:.2f}min \t| lr: {lr:.8f} \t| train loss: {losses['mpjpe']['loss_logger'].avg}")
                else:
                    if epoch in epoch_to_eval:
                        epoch_eval_results = {}
                        if evaluate_VER in  ['2_ICL']:
                            summary_table_collection = {}
                            for (dataset, eval_task) in test_loader.keys():
                                err, summary_table_single = evaluate(args, test_loader[(dataset, eval_task)], datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                                epoch_eval_results[(dataset, eval_task)] = err
                                summary_table_collection[(dataset, eval_task)] = summary_table_single
                                train_writer.add_scalar(f'{dataset} | {eval_task}', err, epoch + 1)

                            summary_table.add_row([epoch+1] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])
                        elif evaluate_VER in ['3_ICL']:
                            head_r1 = ['Epoch', 'lr', 'train_loss']
                            head_r2 = [' ', ' ', ' ']
                            head_r3 = [' ', ' ', ' ']
                            ret_log = [epoch + 1, f"{lr:.8f}", f"{losses['mpjpe']['loss_logger'].avg:.5f}"]
                            summary_table_collection = {}
                            for dataset in dataset_task_info_test:
                                for eval_task in dataset_task_info_test[dataset]:
                                    err, summary_table_single, header, results = evaluate(args, test_loader, datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                    
                                    epoch_eval_results[(dataset, eval_task)] = err
                                    summary_table_collection[(dataset, eval_task)] = summary_table_single
                                    train_writer.add_scalar(f'{dataset} | {eval_task}', err, epoch + 1)

                                    head_r2 = head_r2 + [eval_task] + [' '] * (len(header))
                                    head_r3 = head_r3 + ['Avg'] + header
                                    ret_log = ret_log + [err] + results

                                head_r1 = head_r1 + [dataset] + [' '] * (len(head_r2) - len(head_r1) - 1)
                                assert len(head_r1) == len(head_r2) == len(ret_log)

                            summary_table.add_row([epoch+1] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])

                            summary_table_epoch = prettytable.PrettyTable()
                            summary_table_epoch._validate_field_names = lambda *a, **k: None
                            summary_table_epoch.field_names = ['Epoch', 'lr', 'train_loss'] + sum([[dataset_name] + [''] * (len(dataset_task_info_test[dataset_name])-1) for dataset_name in dataset_task_info_test], [])
                            summary_table_epoch.add_row(['', '', ''] + [eval_task for (dataset, eval_task) in test_loader.keys()])
                            summary_table_epoch.add_row([epoch+1, f"{lr:.8f}", f"{losses['mpjpe']['loss_logger'].avg:.5f}"] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])
                            summary_table_epoch.float_format = ".2"
                            print(summary_table_epoch)
                        elif evaluate_VER in ['4_ICL']:
                            epoch_eval_results_smpl = {}

                            head_r1 = ['Epoch', 'lr', 'train_loss']
                            head_r2 = [' ', ' ', ' ']
                            head_r3 = [' ', ' ', ' ']
                            ret_log = [epoch + 1, f"{lr:.8f}", f"{losses['mpjpe']['loss_logger'].avg:.5f}"]
                            summary_table_collection = {}
                            for dataset in dataset_task_info_test:
                                for eval_task in dataset_task_info_test[dataset]:
                                    err, summary_table_single, header, results, err_smpl = evaluate(args, test_loader, datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                    
                                    epoch_eval_results[(dataset, eval_task)] = err
                                    epoch_eval_results_smpl[(dataset, eval_task)] = err_smpl
                                    summary_table_collection[(dataset, eval_task)] = summary_table_single
                                    train_writer.add_scalar(f'{dataset} | {eval_task}', err, epoch + 1)

                                    head_r2 = head_r2 + [eval_task] + [' '] * (len(header))
                                    head_r3 = head_r3 + ['Avg'] + header
                                    ret_log = ret_log + [err] + results

                                head_r1 = head_r1 + [dataset] + [' '] * (len(head_r2) - len(head_r1) - 1)
                                assert len(head_r1) == len(head_r2) == len(ret_log)

                            summary_table.add_row([epoch+1] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])

                            summary_table_epoch = prettytable.PrettyTable()
                            summary_table_epoch._validate_field_names = lambda *a, **k: None
                            summary_table_epoch.field_names = ['Epoch', 'lr', 'train_loss'] + [''] + sum([[dataset_name] + [''] * (len(dataset_task_info_test[dataset_name])-1) for dataset_name in dataset_task_info_test], [])
                            summary_table_epoch.add_row([epoch+1, f"{lr:.8f}", f"{losses['mpjpe']['loss_logger'].avg:.5f}"] + [''] + [eval_task for (dataset, eval_task) in test_loader.keys()])
                            summary_table_epoch.add_row(['', '', ''] + ['MPJPE'] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])
                            summary_table_epoch.add_row(['', '', ''] + ['MPVE'] + [epoch_eval_results_smpl[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])
                            summary_table_epoch.float_format = ".2"
                            print(summary_table_epoch)                
                        elif evaluate_VER in ['5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                            head_r1 = ['Epoch', 'lr', 'JOINT']
                            head_r2 = [' ', ' ', ' ']
                            head_r3 = [' ', ' ', ' ']
                            ret_log = [epoch + 1, f"{lr:.8f}", f"{losses['joint_total'].avg:.5f}"]

                            if use_smpl:
                                head_r1 = head_r1 + ['MESH', 'ALL']
                                head_r2 = head_r2 + [' ', ' ']                            
                                head_r3 = head_r3 + [' ', ' ']
                                ret_log = ret_log + [f"{losses['mesh_total'].avg:.5f}", f"{losses['all_total'].avg:.5f}"]

                            PrettyTableFieldNames = head_r1
                            PrettyTable_TrainStat = ret_log
                            PrettyTable_Task = []
                            PrettyTable_Dataset = []
                            PrettyTable_Joint = []
                            if use_smpl:
                                PrettyTable_Smpl = []

                            for dataset in dataset_task_info_test:
                                for eval_id, eval_task in enumerate(dataset_task_info_test[dataset]):
                                    
                                    eval_results = evaluate(args, test_loader, datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                                    if use_smpl:
                                        err_avg_joint, err_full_joint, header_full_joint, err_avg_mesh, err_full_mesh, header_full_mesh, _ = eval_results
                                    else:
                                        err_avg_joint, err_full_joint, header_full_joint, _ = eval_results
                                    
                                    epoch_eval_results[(dataset, eval_task)] = err_avg_joint
                                    train_writer.add_scalar(f'{dataset} | {eval_task} | joint', err_avg_joint, epoch + 1)
                                    if use_smpl:
                                        train_writer.add_scalar(f'{dataset} | {eval_task} | mesh', err_avg_mesh, epoch + 1)

                                    head_r2 = head_r2 + [eval_task] + [' '] * (len(header_full_joint))
                                    head_r3 = head_r3 + ['Avg'] + header_full_joint
                                    ret_log = ret_log + [err_avg_joint] + err_full_joint
                                    if use_smpl:
                                        head_r2 = head_r2 + [' '] * (len(header_full_mesh))
                                        head_r3 = head_r3 + header_full_mesh
                                        ret_log = ret_log + err_full_mesh

                                    PrettyTable_Dataset += [dataset] if eval_id == 0 else ['      ']
                                    PrettyTable_Task += [eval_task]
                                    PrettyTable_Joint += [err_avg_joint]
                                    if use_smpl:
                                        PrettyTable_Smpl += [err_avg_mesh]
                                    

                                head_r1 = head_r1 + [dataset] + [' '] * (len(head_r2) - len(head_r1) - 1)
                                assert len(head_r1) == len(head_r2) == len(ret_log)


                            summary_table_epoch = prettytable.PrettyTable()
                            summary_table_epoch._validate_field_names = lambda *a, **k: None
                            summary_table_epoch.field_names = PrettyTableFieldNames + PrettyTable_Dataset + PrettyTable_Dataset
                            summary_table_epoch.add_row(PrettyTable_TrainStat + PrettyTable_Task + PrettyTable_Task)
                            summary_table_epoch.add_row(['']*len(PrettyTableFieldNames) + PrettyTable_Joint + PrettyTable_Smpl)
                            summary_table_epoch.float_format = ".2"
                            print(summary_table_epoch)

                            summary_table.add_row([epoch+1] + [epoch_eval_results[(dataset, eval_task)] for (dataset, eval_task) in test_loader.keys()])
                        else:
                            if 'PE' in args.tasks:
                                e1, e2, summary_table_PE = evaluate_pose_estimation(args, model_pos, test_loader['PE'], datareader_pose_estimation, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['PE e1'] = e1; epoch_eval_results['PE e2'] = e2
                                train_writer.add_scalar('PE Error P1', e1, epoch + 1)
                                train_writer.add_scalar('PE Error P2', e2, epoch + 1)
                            if 'FPE' in args.tasks:
                                e1FPE, summary_table_FPE = evaluate_future_pose_estimation(args, test_loader['FPE'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['FPE'] = e1FPE
                                train_writer.add_scalar('FPE MPJPE', e1FPE, epoch + 1)

                            if 'MP' in args.tasks:
                                mpjpe, summary_table_MP = evaluate_motion_prediction(args, test_loader['MP'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['MP'] = mpjpe
                                train_writer.add_scalar('MP MPJPE', mpjpe, epoch + 1)

                            if 'MC' in args.tasks:
                                min_err_mc, summary_table_MC = evaluate_motion_completion(args, test_loader['MC'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['MC'] = min_err_mc
                                train_writer.add_scalar('MC MPJPE', min_err_mc, epoch + 1)
                            
                            if '2DAR' in args.tasks:
                                acc_top1, summary_table_2DAR = evaluate_action_recognition2D(args, test_loader['2DAR'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['2DAR'] = acc_top1
                                train_writer.add_scalar('2DAR acc top1', acc_top1, epoch + 1)

                            if 'AR' in args.tasks:
                                acc_top1_AR, summary_table_AR = evaluate_action_recognition(args, test_loader['AR'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                                epoch_eval_results['AR'] = acc_top1_AR
                                train_writer.add_scalar('AR acc top1', acc_top1_AR, epoch + 1)

                            summary_table.add_row([epoch+1] + [epoch_eval_results[metric] for task in args.tasks for metric in args.task_metrics[task]])
                    
                    if evaluate_VER in ['5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                        for loss_name, loss_dict in losses['joint'].items():
                            train_writer.add_scalar('Joint | '+loss_name, loss_dict['loss_logger'].avg, epoch + 1)
                        train_writer.add_scalar('Joint | Total', losses['joint_total'].avg, epoch + 1)
                        if use_smpl:
                            for loss_name, loss_dict in losses['mesh'].items():
                                train_writer.add_scalar('Mesh | '+loss_name, loss_dict['loss_logger'].avg, epoch + 1)
                            train_writer.add_scalar('Mesh | Total', losses['mesh_total'].avg, epoch + 1)
                            train_writer.add_scalar('All | Total', losses['all_total'].avg, epoch + 1)
                    else:
                        for loss_name, loss_dict in losses.items():
                            train_writer.add_scalar(loss_name, loss_dict['loss_logger'].avg, epoch + 1)

                # Save checkpoints
                # if epoch in epoch_to_eval:
                #     chk_path = os.path.join(opts.checkpoint, 'all_epochs', f'epoch_{epoch}.bin')
                #     save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, eval_dict, no_print=True)

                chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch', 'latest_epoch.bin')
                save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, eval_dict, no_print=True)

                if dataset_VER in ['2_ICL', '3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                    chk_path_best = {(dataset, eval_task): os.path.join(opts.checkpoint, 'best_epochs', f'best_epoch_{dataset}_{eval_task}.bin') for (dataset, eval_task) in test_loader.keys()}
                else:
                    chk_path_best = {task: os.path.join(opts.checkpoint, f'best_epoch_{task}.bin') for task in args.tasks}
                chk_path_best['all'] = os.path.join(opts.checkpoint, 'best_epoch_all.bin')


                # Save best checkpoint according to global best 
                if not args.no_eval:
                    if epoch in epoch_to_eval:
                        if dataset_VER in ['2_ICL']:
                            for (dataset, eval_task) in test_loader.keys():
                                err_new = epoch_eval_results[(dataset, eval_task)]
                                if err_new < eval_dict[(dataset, eval_task)]['min_err']:
                                    eval_dict[(dataset, eval_task)]['min_err'] = err_new
                                    eval_dict[(dataset, eval_task)]['best_epoch'] = epoch + 1
                                    save_checkpoint(chk_path_best[(dataset, eval_task)], epoch, lr, optimizer, model_pos, eval_dict)
                                
                            for (dataset, eval_task) in test_loader.keys():
                                print(summary_table_collection[(dataset, eval_task)])
                        elif dataset_VER in ['3_ICL', '4_ICL', '5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                            print(f'\tSaving best checkpoints:', end=' ')
                            for (dataset, eval_task) in test_loader.keys():
                                err_new = epoch_eval_results[(dataset, eval_task)]
                                if err_new < eval_dict[(dataset, eval_task)]['min_err']:
                                    eval_dict[(dataset, eval_task)]['min_err'] = err_new
                                    eval_dict[(dataset, eval_task)]['best_epoch'] = epoch + 1
                                    save_checkpoint(chk_path_best[(dataset, eval_task)], epoch, lr, optimizer, model_pos, eval_dict, no_print=True)
                                    print(f'[{dataset},{eval_task}]', end=' ')
                            print('\n')
                            

                            if epoch == st:
                                exp_info = np.array([
                                                    'EXP_INFO', ' ', ' ', 
                                                    'PID', os.getpid(), ' ', ' ',
                                                    ' ', ' ', ' '.join(sys.argv), ' ', ' ',
                                                    'Checkpoint', opts.checkpoint, ' ', ' ', 
                                                    'Config', opts.config, ' ', ' '
                                                    ])
                                place_holder = np.array([' '] * (len(exp_info)))

                                df = pd.DataFrame(np.vstack([exp_info, place_holder]))
                                if not os.path.exists(os.path.join(opts.checkpoint, f'PID_{os.getpid()}.csv')):
                                    df.to_csv(os.path.join(opts.checkpoint, f'PID_{os.getpid()}.csv'), index=False, header=False)
                                else:
                                    with open(os.path.join(opts.checkpoint, f'PID_{os.getpid()}.csv'), 'a') as f:
                                        f.write('\n')
                                        df.to_csv(f, index=False, header=False)

                                head_r1 = np.array(head_r1)
                                head_r2 = np.array(head_r2)
                                head_r3 = np.array(head_r3)

                                df = pd.DataFrame(np.vstack([head_r1, head_r2, head_r3]))
                                with open(os.path.join(opts.checkpoint, f'PID_{os.getpid()}.csv'), 'a') as f:
                                        df.to_csv(f, index=False, header=False)

                            for i in range(len(ret_log)):
                                if isinstance(ret_log[i], float):
                                    ret_log[i] = round(ret_log[i], 2)
                            ret_log = np.array(ret_log)
                            df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
                            with open(os.path.join(opts.checkpoint, f'PID_{os.getpid()}.csv'), 'a') as f:
                                df.to_csv(f, index=False, header=False)
                            
                                
                        else:
                            if 'PE' in args.tasks and e1 < eval_dict['PE']['min_err']:
                                eval_dict['PE']['min_err'] = e1
                                eval_dict['PE']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['PE'], epoch, lr, optimizer, model_pos, eval_dict)
                                
                            if 'MP' in args.tasks and mpjpe < eval_dict['MP']['min_err']:
                                eval_dict['MP']['min_err'] = mpjpe
                                eval_dict['MP']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['MP'], epoch, lr, optimizer, model_pos, eval_dict)
                            
                            if 'FPE' in args.tasks and e1FPE < eval_dict['FPE']['min_err']:
                                eval_dict['FPE']['min_err'] = e1FPE
                                eval_dict['FPE']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['FPE'], epoch, lr, optimizer, model_pos, eval_dict)
                            
                            if 'MC' in args.tasks and min_err_mc < eval_dict['MC']['min_err']:
                                eval_dict['MC']['min_err'] = min_err_mc
                                eval_dict['MC']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['MC'], epoch, lr, optimizer, model_pos, eval_dict)

                            if '2DAR' in args.tasks and acc_top1 > eval_dict['2DAR']['max_acc']:
                                eval_dict['2DAR']['max_acc'] = acc_top1
                                eval_dict['2DAR']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['2DAR'], epoch, lr, optimizer, model_pos, eval_dict)
                            
                            if 'AR' in args.tasks and acc_top1_AR > eval_dict['AR']['max_acc']:
                                eval_dict['AR']['max_acc'] = acc_top1_AR
                                eval_dict['AR']['best_epoch'] = epoch + 1
                                save_checkpoint(chk_path_best['AR'], epoch, lr, optimizer, model_pos, eval_dict)

                            try:
                                if (e1 + mpjpe + e1FPE + min_err_mc - acc_top1) / 4 < eval_dict['all']['min_err']:
                                    eval_dict['all']['min_err'] = (e1 + mpjpe + e1FPE + min_err_mc) / 4
                                    eval_dict['all']['best_epoch'] = epoch + 1
                                    save_checkpoint(chk_path_best['all'], epoch, lr, optimizer, model_pos, eval_dict)
                            except:
                                pass


                            # Print evaluation results
                            if 'PE' in args.tasks:
                                print(summary_table_PE)
                            if 'MP' in args.tasks:
                                print(summary_table_MP)
                            if 'MC' in args.tasks:
                                print(summary_table_MC)
                            if 'FPE' in args.tasks:
                                print(summary_table_FPE)
                            if '2DAR' in args.tasks:
                                print(summary_table_2DAR)
                            if 'AR' in args.tasks:
                                print(summary_table_AR)


                    elapsed = (time() - start_time) / 60
                    print(f"[{epoch+1} end] Time cost: {elapsed:.2f}min \t| lr: {lr:.8f}")

                if opts.quick_debug:
                    if epoch > st + 4: break

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            
        print(f"Training took {(time() - training_start_time) / 3600 :.2f}h")

        if not args.no_eval and rank == 0:
            summary_table.float_format = ".2"
            print("All results:")
            print(summary_table)
            print("Best results:")

    if rank == 0:
        if opts.eval_generalization:
            evaluate_generalization(args, test_loader, model_pos, if_viz=opts.visualize, if_debug=opts.quick_debug)

        if opts.evaluate:
            epoch = checkpoint['epoch']
            if opts.stage == 'classifier':
                evaluate(args, test_loader, model_pos, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug, classifier_type=opts.classifier_type)
            else:
                if dataset_VER in ['2_ICL']:
                    for (dataset, eval_task) in test_loader.keys():
                        _, summary_table_single = evaluate(args, test_loader[(dataset, eval_task)], datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                        print(summary_table_single)
                elif dataset_VER in ['3_ICL', '4_ICL']:
                    for dataset in dataset_task_info_test:
                        for eval_task in dataset_task_info_test[dataset]:
                            _, summary_table_single, _, _ = evaluate(args, test_loader, datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                            print(summary_table_single)
                elif dataset_VER in ['5_ICL', '6_ICL', '7_ICL', '8_ICL']:
                    for dataset in dataset_task_info_test:
                        for eval_task in dataset_task_info_test[dataset]:
                            evaluation_results = evaluate(args, test_loader, datareader_pose_estimation, model_pos, dataset, eval_task, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                            summary_table = evaluation_results[-1]
                            print(summary_table)
                else:
                    if 'PE' in args.tasks:
                        _, _, summary_table_PE = evaluate_pose_estimation(args, model_pos, test_loader['PE'], datareader_pose_estimation, epoch=epoch, if_debug=opts.quick_debug)
                        print(summary_table_PE)
                    if 'MP' in args.tasks:
                        _, summary_table_MP = evaluate_motion_prediction(args, test_loader['MP'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                        print(summary_table_MP)
                    if 'MC' in args.tasks:
                        _, summary_table_MC = evaluate_motion_completion(args, test_loader['MC'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                        print(summary_table_MC)
                    if 'FPE' in args.tasks:
                        _, summary_table_FPE = evaluate_future_pose_estimation(args, test_loader['FPE'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                        print(summary_table_FPE)
                    if '2DAR' in args.tasks:
                        _, summary_table_2DAR = evaluate_action_recognition2D(args, test_loader['2DAR'], model_pos, epoch=epoch, if_debug=opts.quick_debug)
                        print(summary_table_2DAR)
                    if 'AR' in args.tasks:
                        _, summary_table_AR = evaluate_action_recognition(args, test_loader['AR'], model_pos, epoch=epoch, if_viz=opts.visualize, if_debug=opts.quick_debug)
                        print(summary_table_AR)


    if rank == 0:
        train_writer.close()
    cleanup()

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    for key, value in opts.__dict__.items():
        if key in args.__dict__ and value is not None:
            setattr(args, key, value)
            print(f'Overriding existing arg: args.{key} = {value}')
        elif key not in args.__dict__ and value is not None and (value != '' or key == 'visualize'):
            setattr(args, key, value)
            print(f'Adding new arg: args.{key} = {value}')

    # assign args.data
    assert '.bin' not in opts.checkpoint
    if args.use_partial_data:
        args.data = args.partial_data
    else:
        args.data = args.full_data

    # print info
    print(f'\nConfigs: {args}')
    print('\npython ' + ' '.join(sys.argv))
    print('\nPID: ', os.getpid())

    # create checkpoint folder
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    
    # create subfolders for storing checkpoints
    # CREATE 'ALL_EPOCHS' FOLDER ON HARD DISK
    ROOT_DIR_HARDDISK = os.path.dirname(os.path.abspath(__file__)).replace('wxs', 'wxs/wxs')
    assert not opts.checkpoint.startswith('/')
    try:
        os.makedirs(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'all_epochs'))
        print('Created folder:', os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'all_epochs'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'all_epochs'))
        
    # LINK 'ALL_EPOCHS' FOLDER BACK TO SSD
    try:
        os.symlink(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'all_epochs'), os.path.join(opts.checkpoint, 'all_epochs'))
        print('Linked folder:', os.path.join(opts.checkpoint, 'all_epochs'))
    except OSError as e:
        print(e)
    
    # CREATE 'BEST_EPOCHS' FOLDER ON HARD DISK
    try:
        os.makedirs(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'best_epochs'))
        print('Created folder:', os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'best_epochs'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'best_epochs'))
    
    # LINK 'BEST_EPOCHS' FOLDER BACK TO SSD
    try:
        os.symlink(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'best_epochs'), os.path.join(opts.checkpoint, 'best_epochs'))
        print('Linked folder:', os.path.join(opts.checkpoint, 'best_epochs'))
    except OSError as e:
        print(e)
    
    # CREATE 'LATEST_EPOCH' FOLDER ON HARD DISK
    try:
        os.makedirs(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'latest_epoch'))
        print('Created folder:', os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'latest_epoch'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', os.path.join(opts.checkpoint, 'latest_epoch'))
    
    # LINK 'LATEST_EPOCH' FOLDER BACK TO SSD
    try:
        os.symlink(os.path.join(ROOT_DIR_HARDDISK, opts.checkpoint, 'latest_epoch'), os.path.join(opts.checkpoint, 'latest_epoch'))
        print('Linked folder:', os.path.join(opts.checkpoint, 'latest_epoch'))
    except OSError as e:
        print(e)

    # link out file to checkpoint folder
    if opts.out:
        try:
            if opts.out.endswith('.out'):
                os.link(os.path.join('out', opts.out), os.path.join(opts.checkpoint, opts.out))
            else:
                os.link(os.path.join('out', f'{opts.out}.out'), os.path.join(opts.checkpoint, f'{opts.out}.out'))
        except:
            print('\nFailed to create symlink for out file.')

    # create config backup
    if not opts.evaluate:
        if not os.path.exists(os.path.join(opts.checkpoint, os.path.basename(opts.config))):
            shutil.copy(opts.config, opts.checkpoint)
        else:
            new_config_filename = os.path.splitext(os.path.basename(opts.config))[0] \
                                    + '_' \
                                    + '{0:D%Y%m%dT%H%M%S}'.format(datetime.now(pytz.timezone('Etc/GMT-8'))) \
                                    + os.path.splitext(os.path.basename(opts.config))[1]
            new_config_filepath = os.path.join(opts.checkpoint, new_config_filename)
            shutil.copy(opts.config, new_config_filepath)


    

    world_size = torch.cuda.device_count()
    try:
        mp.spawn(train_with_config, args=(world_size, args, opts), nprocs=world_size, join=True)
    except:
        cleanup()

if __name__ == "__main__":
    # Set stdout and stderr to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    os.environ['PYTHONUNBUFFERED'] = '1' # 防止多进程写入log导致阻塞
    main()
