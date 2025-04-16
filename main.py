# -*- coding: utf-8 -*-

import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/AWGN_data1', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, Rician, Suzuki and Nakagami')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    total_loss = 0.0
    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
        total_loss += loss

    # Return the average training loss
    return total_loss / len(train_iterator)
def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
    return total/len(test_iterator)

if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = 'data/processed_data_1/' + args.vocab_file
    
    """ 准备检查点目录 """
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    """ 准备数据集 """
    # 加载词汇表
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 构建 token_to_idx 和 idx_to_token 的映射
    token_to_idx = vocab
    idx_to_token = {v: k for k, v in token_to_idx.items()}

    # 提取特殊标记的索引
    pad_idx = token_to_idx.get('[PAD]', None)
    start_idx = token_to_idx.get('[CLS]', None)
    end_idx = token_to_idx.get('[SEP]', None)
    unk_idx = token_to_idx.get('[UNK]', None)
    num_vocab = len(token_to_idx)

    """ 定义模型和优化器 """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    initNetParams(deepsc)
    
    """ 断点续训配置 """
    best_checkpoint_path = os.path.join(args.checkpoint_path, 'best_checkpoint.pth')
    last_checkpoint_path = os.path.join(args.checkpoint_path, 'last_checkpoint.pth')  # 单一文件用于恢复
    final_checkpoint = os.path.join(args.checkpoint_path, 'final_checkpoint.pth')
    start_epoch = 0
    record_acc = float('inf')

    """ 加载最近检查点（如果存在） """
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path)
        deepsc.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        record_acc = checkpoint['record_acc']
        print(f'Resuming from epoch {start_epoch}...')

    try:
        """ 训练循环（每个epoch后保存检查点） """
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            
            # 训练
            train_loss = train(epoch, args, deepsc, mi_net)
            
            # 验证
            avg_acc = validate(epoch, args, deepsc)
            
            # 将训练和验证损失写入日志文件
            log_file = args.log_file if hasattr(args, 'log_file') else 'training_log.txt'
            with open(log_file, 'a') as f:
                f.write(f'Checkpoint directory:{args.checkpoints_path}\n')
                f.write(f'Epoch: {epoch+1}; Type: Train; Loss: {train_loss:.5f}\n')
                f.write(f'Epoch: {epoch+1}; Type: VAL; Loss: {avg_acc:.5f}\n')
            
            # 保存最佳模型
            if avg_acc < record_acc:
                torch.save(deepsc.state_dict(), best_checkpoint_path)
                record_acc = avg_acc
                print(f'Epoch {epoch+1}: New best model saved')

            # 每个epoch后保存恢复点（覆盖写入）
            torch.save({
                'epoch': epoch,
                'model_state_dict': deepsc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'record_acc': record_acc
            }, last_checkpoint_path)
            print(f'Epoch {epoch+1}: Checkpoint saved')

        """ 正常完成训练后清理临时检查点 """
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
            print('Training completed, temporary checkpoints cleared')

    except (KeyboardInterrupt, Exception) as e:
        """ 中断时保留检查点并提示恢复方法 """
        print(f'\nTraining interrupted at epoch {epoch}: {str(e)}')
        print(f'Last checkpoint saved to {last_checkpoint_path}, resume training with this file.')

    finally:
        """ 无论是否中断，最终保存模型 """
        torch.save(deepsc.state_dict(), final_checkpoint)
        print(f'Final model saved to {final_checkpoint}')


    




