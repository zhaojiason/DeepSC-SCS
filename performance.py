# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import time
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def performance(args, SNR, net):
    # 仅初始化一次数据加载器
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, 
                             num_workers=0, pin_memory=True,
                             collate_fn=collate_data, shuffle=False)
    StoT = SeqtoText(token_to_idx, end_idx)
    
    net.eval()
    with torch.no_grad():
        # 仅测试第一个SNR点（可修改为特定值）
        snr = SNR[0]
        noise_std = SNR_to_noise(snr)
        
        # 获取第一个batch的数据
        sample_batch = next(iter(test_iterator))
        sents = sample_batch.to(device)
        
        # 生成预测
        out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH,
                           pad_idx, start_idx, args.channel)
        
        # 解码文本
        generated = [StoT.sequence_to_text(seq) for seq in out.cpu().numpy().tolist()]
        target = [StoT.sequence_to_text(seq) for seq in sents.cpu().numpy().tolist()]
        
        # 仅取第一条样本展示
        print("\n=== 文本对比示例 ===")
        print(f"Target:    {target[1]}")
        print(f"Generated: {generated[1]}\n")
        
        # 计算全局BLEU分数（可选）
        bleu_score_1gram = BleuScore(1, 0, 0, 0)
        return bleu_score_1gram.compute_blue_score(generated, target)

def interactive_performance(args, SNR, net):
    StoT = SeqtoText(token_to_idx, end_idx)
    net.eval()
    
    with torch.no_grad():
        # 使用第一个SNR值
        snr = SNR[0]
        noise_std = SNR_to_noise(snr)
        
        while True:
            # 获取用户输入
            user_input = input("\n请输入测试文本（输入'exit'退出）: ")
            if user_input.lower() == 'exit':
                break
            
            try:
                # 将文本转换为模型输入格式
                seq = StoT.text_to_sequence(user_input, add_start_token=True, add_end_token=True)
                
                # 转换为张量并添加batch维度
                seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                # 生成预测结果
                out = greedy_decode(net, seq_tensor, noise_std, args.MAX_LENGTH,
                                  pad_idx, start_idx, args.channel)
                
                # 解码生成文本
                generated = StoT.sequence_to_text(out.cpu().numpy().tolist()[0])
                
                # 清理填充符号和结束符
                generated = generated.split('<END>')[0].replace('<START>','').replace('<PAD>', '').strip()
                
                # 显示结果
                print("\n=== 文本生成结果 ===")
                print(f"[原始输入] {user_input}")
                print(f"[生成结果] {generated}")
                
            except Exception as e:
                print(f"处理出错: {str(e)}")
                continue

        print("\n已退出交互测试模式")
        return 0  # 返回0作为占位，原BLEU功能已移除

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [20]
    start_time = time.time()
    
    # 动态检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载词汇表
    args.vocab_file = 'data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    # ============== 修复1：设备感知的优化逻辑 ==============
    if torch.cuda.is_available():
        # GPU专用优化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        loader_stream = torch.cuda.Stream()
    else:
        # CPU优化：设置并行线程并禁用流操作
        torch.set_num_threads(os.cpu_count())
        loader_stream = None

    # 初始化模型结构
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    best_model_path = os.path.join(args.checkpoint_path, 'best_checkpoint.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    # ============== 修复2：设备感知的map_location ==============
    def load_checkpoint():
        return torch.load(
            best_model_path,
            map_location=device,  # 直接指定目标设备
            mmap=True if device.type == 'cpu' else False,
            weights_only=True
        )
    # ============== 修复3：异步加载逻辑分离 ==============
    checkpoint = None
    if loader_stream:
        # GPU异步加载
        with torch.cuda.stream(loader_stream):
            checkpoint = load_checkpoint()
        torch.cuda.current_stream().wait_stream(loader_stream)
    else:
        # CPU同步加载
        checkpoint = load_checkpoint()

    # ============== 修复4：兼容CPU的参数传输 ==============
    try:
        checkpoint = torch.load(best_model_path, weights_only=False)
        deepsc.load_state_dict(checkpoint)
    except RuntimeError as e:
        # 自动修复常见键名不匹配问题
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        deepsc.load_state_dict(checkpoint, strict=False)
        print(f"加载警告: {str(e)}")

    # ============== 修复5：混合精度条件执行 ==============
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                bleu_score = performance(args, SNR, deepsc)
        else:
            # 进行用户交互测试
            interactive_performance(args, SNR, deepsc)
            bleu_score = performance(args, SNR, deepsc)
    
    print(f"BLEU Score: {bleu_score}")