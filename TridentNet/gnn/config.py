"""
Fix Me: define your own network, result log and global parameters for this project
"""
import json
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from models.tridentNet import TridentNet
import os
import torch
import torch.distributed as dist
import numpy as np
import random

def weights_init_uniform(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        if layer.bias != None:
            nn.init.constant_(layer.bias, 0.)

# Multiple gpu configuration 
# New arguments: rank, word_size, local_rank, distributed, dist_backend, device
def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器(GPU?)数，RANK对应第几台机器(GPU?)，rank = 0 的主机为 master 节点
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if'RANK'in os.environ and 'WORLD_SIZE'in os.environ:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])# LOCAL_RANK代表某个机器上第几块GPU
        torch.cuda.set_device(args.local_rank)  # 对当前进程指定使用的GPU
        args.dist_backend = 'nccl'# 通信后端，nvidia GPU推荐使用NCCL
        dist.init_process_group(backend=args.dist_backend)
        args.device = torch.device("cuda", args.local_rank) # 获取GPU
        print(f"[init] == local rank: {args.local_rank}, global rank: {args.rank} ==")
        dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



def init_task_specified_params(args):    
    # network 
    args.particleNetSettings =  {
            "conv_params": [
                (16, (64, 64, 64)),
                (16, (128, 128, 128)),
                (16, (256, 256, 256)),
                (16, (512, 512, 512)),
                (16, (512, 512, 512)),
                (16, (512, 512, 512)),
            ],
            "fc_params": [
                (0.1, 256),
                (0.05, 32),
            ],
            "input_features": 5,
            "output_classes": 3,
        }
    args.net = TridentNet(args.particleNetSettings, args.device)
    
    args.net.apply(weights_init_uniform)
    # result log
    args.reslog = {'epochs': [], 'lr' : [], 'train_slice': [], 'test_slice': [],
        'train_time': [], 'test_time': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    # Loss function & optimizer
    # args.criterion = nn.CrossEntropyLoss(reduction='sum') # The recorded loss is in fact mean loss. refer train_test.train_one_epoch: res['train_loss'].append 
    args.criterion = nn.MSELoss(reduction='none')
    

    # Files and labels
    args.fileList = args.fileList
    args.labelList = args.labelList


# Get arguments from command line
def get_args():
    # Input arguments
    parser = argparse.ArgumentParser(description="Convolutional Neuron Network trainning by Cen Mo")
    # Basic setting
    parser.add_argument('--seed', type=int, default=42, help='random seed, default=15')
    parser.add_argument('--logDir', type=str, default='./', help='output directory, default=./')

    # Input data setting
    parser.add_argument('--fileList', type=str, nargs='*', default=None, help='intput file list')
    parser.add_argument('--labelList', type=int, nargs='*', default=None, help='intput label list')
    parser.add_argument('--num_classes', type=int, default=3, help='number of image classes, default=3')
    parser.add_argument('--num_channels', type=int, default=3, help='number of image channels, default=3')

    # Trainning parameters
    parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs, default=6')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size, default=128')
    parser.add_argument('--data_size', type=int, default=-1, help='data size per class, default=-1')
    parser.add_argument('--num_slices_train', type=int, default=1, help='split train data into n slices, default=1')
    parser.add_argument('--num_slices_test', type=int, default=1, help='split test data into n slices, default=1')
    parser.add_argument('--num_slices_apply', type=int, default=1, help='split apply data into n slices, default=1')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
    parser.add_argument('--apply_only', type=int, default=0, choices=[0,1], help='apply only mode, default=0')

    # Use trained model
    parser.add_argument('--pre_train', type=int, default=0, choices=[0, 1], help='use pre train epoch, def=0, 0 means no, 1 means yes')
    parser.add_argument('--pre_net', type=str, default="./net.pt", help="pre-trained network")
    parser.add_argument('--pre_log', type=str, default="./train-result.json", help="log directory for pre-trainning")

    # Extra parameters needed by specific task
    parser.add_argument('--extra_param', type=str, nargs='*', default=None, help=' Extra parameters needed by specific task')
   
    args = parser.parse_args()
    return args


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def custom_formatter(x):
    return f'{x:.2f}'

def initialize():
    # Set the formatter to the custom formatter function
    np.set_printoptions(formatter={'float_kind': custom_formatter})
    # Set random seed
    seed_everything(args.seed)
    
    # Multiple GPU configuration
    # New arguments: rank, word_size, local_rank, distributed, dist_backend
    init_distributed_mode(args)
    
    # Task specified parameters is obtained here
    init_task_specified_params(args)
    
    # Do not print net
    params_to_print = {key: value for key, value in args.__dict__.items() if key not in ['net']}
    params_to_print
    print(params_to_print,flush=True)

   
# Receive args from main.py
args = get_args()
initialize()



