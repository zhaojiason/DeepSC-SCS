# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = 'data/txt/'
        with open(data_dir + 'data_demo/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)


    def __getitem__(self, index):
        sents = self.data[index]
        return  sents

    def __len__(self):
        return len(self.data)

def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return  torch.from_numpy(sents)

# class EurDataset(Dataset):
#     def __init__(self, data, split='train'):
#         self.data = data
#         self.split = split

#     def __getitem__(self, index):
#         sents = self.data[index]
#         return sents

#     def __len__(self):
#         return len(self.data)

# def collate_data(batch):
#     batch_size = len(batch)
#     max_len = max(map(lambda x: len(x), batch))  # get the max length of sentence in current batch
#     sents = np.zeros((batch_size, max_len), dtype=np.int64)
#     sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

#     for i, sent in enumerate(sort_by_len):
#         length = len(sent)
#         sents[i, :length] = sent  # padding the questions

#     return torch.from_numpy(sents)

# def load_and_split_data(data_dir, train_ratio=0.8):
#     with open(data_dir + 'data_demo/data.pkl', 'rb') as f:
#         data = pickle.load(f)

#     random.shuffle(data)
#     split_index = int(len(data) * train_ratio)
#     train_data = data[:split_index]
#     test_data = data[split_index:]
#     return train_data, test_data

# # 使用自定义的数据集划分
# data_dir = 'data/txt/'
# train_data, test_data = load_and_split_data(data_dir, train_ratio=0.8)
