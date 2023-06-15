"""
Fix Me: define your own dataloader with the function: get_dataloader
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch_geometric
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
import copy
import warnings
import random
import math
from config import args

# constants
c = 0.2998
n_water = 1.35  # for pure water
c_n = c / n_water
costh = 1 / n_water
tanth = math.sqrt(1 - costh*costh) / costh
sinth = costh * tanth


def get_dataloader(loaderType, data_slice_id, num_slices, data_size, batch_size):
    loader = None
    if loaderType=="train":
        dataId = data_slice_id
        trainset = MyDataset(dataId, dataAug=True)
        loader = torch_geometric.loader.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    elif loaderType=='test':
        dataId = args.num_slices_train + data_slice_id
        testset = MyDataset(dataId, dataAug=False)
        loader = torch_geometric.loader.DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif loaderType=='apply':
        dataId = args.num_slices_train+args.num_slices_test + data_slice_id
        applyset = MyDataset(dataId, dataAug=False)
        loader = torch_geometric.loader.DataLoader(applyset, batch_size=batch_size, shuffle=False)
    return loader


# class MyDataset(torch_geometric.data.InMemoryDataset):
#     def __init__(self, data_id, transform=None, pre_transform=None, pre_filter=None):
#         self.data_dir = args.fileList[0]
#         # self.data_dir = '/lustre/collider/mocen/project/hailing/machineLearning/data/signal_16Jan2023/gnn/'
#         super().__init__(self.data_dir, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(f'{self.data_dir}/xyz_{data_id}.pt')

#     @property
#     def processed_file_names(self):
#         return [self.data_dir]


class MyDataset(torch_geometric.data.Dataset):
    def __init__(self, data_id, transform=None, pre_transform=None, pre_filter=None, dataAug=False):
        self.data_dir = args.fileList[0]
        super().__init__(self.data_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(f'{self.data_dir}/xyz_{data_id}.pt')
        self.datAug = dataAug

    @property
    def processed_file_names(self):
        return [self.data_dir]


    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            self.size = len(value) - 1
            return (2 * (len(value) - 1) )if self.datAug else (len(value) - 1)
        return 0

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])
            
        dataAug = idx // self.size
        idx = idx % self.size


        data = copy.deepcopy(separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        ))

        # Data augmentation
        dataAug = int(random.random()*20) if self.datAug else dataAug
        dataAug1 = dataAug % 4
        dataAug2 = dataAug // 4
        binary = bin(dataAug1)
        if int(binary[-1])==1:
            data.pos[:,2] *= -1
            data.y[:,2] *= -1
            data.inject[:,2] *= -1
        if dataAug1>1 and int(binary[-2])==1:
            data.pos[:,0] *= -1
            data.y[:,0] *= -1
            data.inject[:,0] *= -1
        
        theta = dataAug2 * 2*np.pi/5.
        csin, sin = np.cos(theta), np.sin(theta)
        trans=torch.tensor([[csin,-sin],[sin,csin]]).type(torch.float)
        data.pos[:,[0,1]] = torch.mm( data.pos[:,[0,1]], trans)
        data.y[:,[0,1]] = torch.mm( data.y[:,[0,1]], trans)
        data.inject[:,[0,1]] = torch.mm( data.inject[:,[0,1]], trans)

        # nhits for weight
        data.nhits = data.x[:,0]
        # data.weight = data.x[:,0]
        # weight = data.nhits.sum() / 10
        # weight = weight if weight < 10 else 10
        # data.weight = torch.sqrt(data.nhits).view(-1,1)
        data.weight = (data.nhits).view(-1,1)

        # # Feature transform
        # nhits
        data.x[:, 0] = torch.clamp(data.x[:, 0], min=0, max=100) 
        # data.x[:, 0] = torch.log(torch.clamp(data.x[:, 0], min=0, max=100) + 1) / 5
        data.x[:, 0] = (torch.clamp(data.x[:, 0], min=0, max=100))
        data.x[:, 1:] = torch.clamp(data.x[:, 1:], min=0, max=1e4)
        # data.x[:, 1:] = torch.log(1 + data.x[:, 1:])+0.1 # minimum time is 0
        data.x[:, 1:] = data.x[:, 1:] / 1. * c

        data.inject = data.inject/1
        data.pos = data.pos/1
        data.x = torch.cat([data.x, data.pos], dim=1)

        # self._data_list[idx] = copy.copy(data)
        return  data

    @property
    def data(self) -> Any:
        msg1 = ("It is not recommended to directly access the internal "
                "storage format `data` of an 'InMemoryDataset'.")
        msg2 = ("The given 'InMemoryDataset' only references a subset of "
                "examples of the full dataset, but 'data' will contain "
                "information of the full dataset.")
        msg3 = ("The data of the dataset is already cached, so any "
                "modifications to `data` will not be reflected when accessing "
                "its elements. Clearing the cache now by removing all "
                "elements in `dataset._data_list`.")
        msg4 = ("If you are absolutely certain what you are doing, access the "
                "internal storage via `InMemoryDataset._data` instead to "
                "suppress this warning. Alternatively, you can access stacked "
                "individual attributes of every graph via "
                "`dataset.{attr_name}`.")
        msg = msg1
        if self._indices is not None:
            msg += f' {msg2}'
        if self._data_list is not None:
            msg += f' {msg3}'
            self._data_list = None
        msg += f' {msg4}'

        warnings.warn(msg)
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._data_list = None

def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node
