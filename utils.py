# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os
import math
import torch
from torch.distributions import Gamma
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from models.mutual_info import sample_batch, mutual_information
from transformers import BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram  weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_bleu_score(self, real, predicted):
        score = []
        smoothing_function = SmoothingFunction()
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2,
                                       weights=(self.w1, self.w2,
                                                self.w3, self.w4),
                                       smoothing_function=smoothing_function.method1))
        # return the average sentence bleu score
        avg_score = sum(score)/len(score)
        return round(avg_score, 5)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 按照index将input重新排列
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        # if step <= 3000 :
        #     lr = 1e-3

        # if step > 3000 and step <=9000:
        #     lr = 1e-4

        # if step>9000:
        #     lr = 1e-5

        lr = self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return lr

    def weight_decay(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step <= 3000:
            weight_decay = 1e-3

        if step > 3000 and step <= 9000:
            weight_decay = 0.0005

        if step > 9000:
            weight_decay = 1e-4

        weight_decay = 0
        return weight_decay


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(
            zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        self.vocb_dictionary = vocb_dictionary

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)

    def text_to_sequence(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 分词
        tokens = tokenizer.tokenize(text)
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        # 转换为ID序列
        token_ids = []
        for token in tokens:
            if token in self.vocb_dictionary:
                token_ids.append(self.vocb_dictionary[token])
            else:
                token_ids.append(self.vocb_dictionary['[UNK]'])
        return token_ids


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(
            1/2), size=Tx_sig.shape).to(device)
        H_imag = torch.normal(0, math.sqrt(
            1/2), size=Tx_sig.shape).to(device)
        H = torch.sqrt(H_real**2 + H_imag**2)
        Tx_sig = Tx_sig * H
        Rx_sig = self.AWGN(Tx_sig, n_var)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=Tx_sig.shape).to(device)
        H_imag = torch.normal(mean, std, size=Tx_sig.shape).to(device)
        H = torch.sqrt(H_real**2 + H_imag**2)
        Tx_sig = Tx_sig * H
        Rx_sig = self.AWGN(Tx_sig, n_var)
        return Rx_sig

    def Suzuki(self, Tx_sig, n_var, sigma_shadow=4):
        # Suzuki信道结合了瑞利衰落和对数正态阴影衰落
        # 生成瑞利衰落信道系数
        H_rayleigh = torch.sqrt(torch.normal(0, math.sqrt(1/2), size=Tx_sig.shape).to(device)**2 +
                                torch.normal(0, math.sqrt(1/2), size=Tx_sig.shape).to(device)**2)
        # 生成对数正态阴影衰落
        shadow = torch.exp(torch.normal(
            0, sigma_shadow, size=Tx_sig.shape).to(device) / 10)
        # 应用Suzuki衰落
        Tx_sig = Tx_sig * H_rayleigh * shadow
        Rx_sig = self.AWGN(Tx_sig, n_var)
        return Rx_sig

    def Nakagami(self, Tx_sig, n_var, m=2):
        # 生成Nakagami-m信道系数
        gamma_dist = Gamma(concentration=m, rate=1.0)
        gamma_samples = gamma_dist.sample(Tx_sig.shape).to(device)
        H = torch.sqrt(gamma_samples / m)
        # 应用Nakagami-m衰落
        Tx_sig = Tx_sig * H
        Rx_sig = self.AWGN(Tx_sig, n_var)
        return Rx_sig


def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


def create_masks(src, trg, padding_idx):

    # [batch, 1, seq_len]
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)

    # [batch, 1, seq_len]
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(device), combined_mask.to(device)


def loss_function(x, trg, padding_idx, criterion):

    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask

    return loss.mean()


def PowerNormalize(x):

    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std


def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(
        trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)

    # y_est = x +  torch.matmul(n, torch.inverse(H))
    # loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine
    # loss = loss_function(pred, trg_real, pad)

    loss.backward()
    opt.step()

    return loss.item()


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    # [batch, 1, seq_len]
    src_mask = (
        src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()


def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(
        trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)

    return loss.item()


def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    # [batch, 1, seq_len]
    src_mask = (
        src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # channel_enc_output = model.blind_csi(channel_enc_output)

    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        # [batch, 1, seq_len]
        trg_mask = (
            outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        look_ahead_mask = subsequent_mask(
            outputs.size(1)).type(torch.FloatTensor)
        # print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


def beam_search_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel,
                       beam_size=15, length_penalty=0.2, end_symbol=2):
    # 初始化信道参数
    channels = Channels()
    device = src.device
    batch_size = src.size(0)

    # 编码器处理（保持原始实现）
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    # 信道传输（根据类型选择）
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Invalid channel type")

    memory = model.channel_decoder(Rx_sig)

    # 束搜索初始化（每个样本独立处理）
    beams = [([start_symbol], 0.0)] * batch_size
    memory = memory.repeat_interleave(
        beam_size, dim=0) if beam_size > 1 else memory

    for step in range(max_len):
        all_candidates = []
        for idx, (seq, score) in enumerate(beams):
            if seq[-1] == end_symbol or len(seq) >= max_len:
                all_candidates.append((seq, score))
                continue

            # 准备解码输入
            outputs = torch.tensor(seq).unsqueeze(0).to(device)
            current_memory = memory[idx:idx +
                                    1] if beam_size == 1 else memory[idx*beam_size:(idx+1)*beam_size]

            # 创建解码掩码
            trg_mask = (outputs == padding_idx).unsqueeze(-2).float()
            look_ahead_mask = subsequent_mask(
                outputs.size(1)).float().to(device)
            combined_mask = torch.max(trg_mask, look_ahead_mask)

            # 解码步骤
            dec_output = model.decoder(
                outputs, current_memory, combined_mask, None)
            logits = model.dense(dec_output[:, -1, :])

            # 概率处理
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)

            # 扩展候选序列
            for i in range(beam_size):
                next_token = top_indices[0, i].item()
                new_log_prob = score + top_log_probs[0, i].item()

                # 长度惩罚
                length_pen = math.pow(
                    (5.0 + len(seq) + 1) / 6.0, length_penalty)
                new_score = new_log_prob / length_pen

                new_seq = seq + [next_token]
                all_candidates.append((new_seq, new_score))

        # 排序并选择top-k候选
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_size]

    # 选择最高分序列并截断结束符
    best_seq = beams[0][0]
    if end_symbol in best_seq:
        best_seq = best_seq[:best_seq.index(end_symbol)]
    return torch.tensor(best_seq).unsqueeze(0).to(device)
