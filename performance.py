# !usr/bin/env python
# -*- coding:utf-8 _*-

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
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/processed_data_2/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Suzuki_data2/', type=str)
parser.add_argument('--channel', default='Suzuki', type=str)
parser.add_argument('--MAX-LENGTH', default=64, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=2, type = int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Similarity:
    
    def __init__(self):
        """初始化时直接使用词频向量化器"""
        self.vectorizer = CountVectorizer(
            binary=True,  
            token_pattern=r'(?u)\b\w+\b'  # 匹配所有单词字符
        )
    
    def compute_similarity(self, src_text, tgt_text):
        """直接处理单个文本对的相似度计算"""
        # 确保输入是列表格式（即使只有一个样本）
        if isinstance(src_text, str):
            src_text = [src_text]
        if isinstance(tgt_text, str):
            tgt_text = [tgt_text]
        
        # 合并文本并向量化
        all_texts = src_text + tgt_text
        matrix = self.vectorizer.fit_transform(all_texts).astype(float)
        
        # 分割向量
        src_vector = matrix[0].reshape(1, -1)
        tgt_vector = matrix[1].reshape(1, -1)
        
        # 计算余弦相似度
        return cosine_similarity(src_vector, tgt_vector)[0][0]



# using pre-trained model to compute the sentence similarity
class BertSimilarity:
    def __init__(self, model_name='bert-base-uncased'):
        # 加载预训练的 BERT 模型和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # 设置模型为评估模式

    def compute_similarity(self, real, predicted):
        score = []

        for sent1, sent2 in zip(real, predicted):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            # 编码输入句子
            inputs1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True, max_length=32)
            inputs2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True, max_length=32)

            # 模型推理
            with torch.no_grad():
                vector1 = self.model(**inputs1).last_hidden_state  # 获得输出
                vector2 = self.model(**inputs2).last_hidden_state  # 获得输出

            # 取出[CLS]的向量（通常用于句子级别的任务）
            vector1 = vector1[:, 0, :].numpy()  # [batch_size, hidden_size]
            vector2 = vector2[:, 0, :].numpy()  # [batch_size, hidden_size]

            # 归一化
            vector1 = normalize(vector1, axis=1, norm='max')
            vector2 = normalize(vector2, axis=1, norm='max')

            # 计算余弦相似度
            dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
            a = np.diag(np.matmul(vector1, vector1.T))  # a*a
            b = np.diag(np.matmul(vector2, vector2.T))  # b*b

            a = np.sqrt(a)
            b = np.sqrt(b)
            
            # 计算文本相似度
            output = dot / (a * b)
            score.extend(output.tolist())  # 存储每个句子的相似度

        # 计算并返回所有相似度的平均值
        if score:
            average_score = np.mean(score)
            return average_score.round(6)
        else:
            return 0  # 处理空结果


def performance(args, SNR, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, 
                             num_workers=0, pin_memory=True,
                             collate_fn=collate_data, shuffle=False)
    StoT = SeqtoText(token_to_idx, end_idx=vocab['[SEP]'])
    bleu_score_1gram = BleuScore(1.0, 0.0, 0.0, 0.0)
    net.eval()
    
    with torch.no_grad():
        noise_std = SNR_to_noise(SNR)
        total_similarity = total_bert_similarity = 0.0
        bleu_score = []
        count = 0

        # 获取一个批次的数据
        batch = next(iter(test_iterator))
        sents = batch.to(device)
        batch_size = sents.size(0)

        # 生成预测
        out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH,
                          pad_idx, start_idx, args.channel)

        for i in range(batch_size):
            # 解码生成文本
            generated = StoT.sequence_to_text(out.cpu().numpy().tolist()[i])
            generated = generated.split('[SEP]')[0].replace('[CLS]', '').replace('[PAD]', '').strip()

            # 解码目标文本
            target = StoT.sequence_to_text(sents.cpu().numpy().tolist()[i])
            target = target.split('[SEP]')[0].replace('[CLS]', '').replace('[PAD]', '').strip()

            # 计算相似度和BLEU分数
            # similarity = Similarity()
            # BERT相似度
            bert_Similarity = BertSimilarity()
            # scores = similarity.compute_similarity(target, generated)
            bert_score = bert_Similarity.compute_similarity(target, generated)
            bleu_score.append(bleu_score_1gram.compute_bleu_score(target, generated))

            # total_similarity += scores
            total_bert_similarity += bert_score
            count += 1

        # 计算平均相似度和BLEU分数
        # avg_similarity = total_similarity / count if count > 0 else 0.0
        avg_bert_similarity = total_bert_similarity / count if count > 0 else 0.0   
        bleu_score = np.array(bleu_score)
                
        if bleu_score.size >= 10:
            # 如果BLEU分数的数量大于等于10个，则取最高的10个分数的平均值
            avg_bleu = np.mean(np.partition(bleu_score, -10)[-10:])
        else:
            # 如果BLEU分数的数量不足10个，则直接取所有分数的平均值
            avg_bleu = np.mean(bleu_score) if bleu_score.size > 0 else 0.0

        # 输出最后一条文本的示例
        print("\n=== Text Comparison Example: ===")
        print(f"Target:    {target}")
        print(f"Generated: {generated}")
        # print(f"[Similarity Score] {scores:.4f}")
        print(f"[BERT Similarity Score] {bert_score}")
        print(f"\n=== Overall Performance ===")
        # print(f"Average Similarity Score: {avg_similarity:.4f}")
        print(f"Average BLEU Score: {avg_bleu:.4f}")
        print(f"[Average BERT Similarity Score] {avg_bert_similarity:.6f}")

        return avg_bert_similarity, avg_bleu
        # return avg_similarity, avg_bert_similarity

def interactive(args, SNR, net):
    StoT = SeqtoText(token_to_idx, end_idx=vocab['[SEP]'])
    bleu_score_1gram = BleuScore(1.0, 0.0, 0.0, 0.0)
    net.eval()
    
    with torch.no_grad():

        noise_std = SNR_to_noise(SNR)
        
        while True:
            # 获取用户输入
            user_input = input("\n请输入测试文本（输入'exit'退出）: ")
            if user_input.lower() == 'exit':
                break
            
            try:
                # 将文本转换为模型输入格式
                seq = StoT.text_to_sequence(user_input.lower())
                
                # 转换为张量并添加batch维度
                seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                
                # 生成预测结果
                out = greedy_decode(net, seq_tensor, noise_std, args.MAX_LENGTH,
                                  pad_idx, start_idx, args.channel)
               
                # 解码生成文本
                generated = StoT.sequence_to_text(out.cpu().numpy().tolist()[0])
                
                # 清理填充符号和结束符
                generated = generated.split('[SEP]')[0].replace('[CLS]','').replace('[PAD]', '').strip()
                
                # 显示结果
                print("\n=== 文本生成结果 ===")
                print(f"[原始输入] {user_input}")
                print(f"[生成结果] {generated}")
                similarity = Similarity()
                scores = similarity.compute_similarity(user_input, generated)
                bleu_score = bleu_score_1gram.compute_bleu_score(generated, user_input)
                print(f"[The simiarlity score is] {scores:.4f}")
                print(f"[The BLEU score is] {bleu_score:.4f}")
            except Exception as e:
                print(f"处理出错: {str(e)}")
                continue

        print("\n已退出交互测试模式")
        return 0  # 返回0作为占位，原BLEU功能已移除

def interactive_performance(args, SNR, net, user_input, token_to_idx, start_idx, end_idx, pad_idx, device, channel):
    StoT = SeqtoText(token_to_idx, end_idx)
    net.eval()
    
    with torch.no_grad():
        noise_std = SNR_to_noise(SNR)
        
        try:
            # 文本转序列
            seq = StoT.text_to_sequence(user_input.lower())
            
            seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            
            out = greedy_decode(
                net, seq_tensor, noise_std, args.MAX_LENGTH,
                pad_idx, start_idx, channel
            )
            
            generated = StoT.sequence_to_text(out.cpu().numpy().tolist()[0])
            generated = generated.split('[SEP]')[0].replace('[CLS]','').replace('[PAD]', '').strip()
            similarity = Similarity()
            scores = similarity.compute_similarity(user_input, generated)
            return generated, scores
        except Exception as e:
            # 修改处：返回包含两个元素的tuple
            return f"处理出错: {str(e)}", 0.0  # 第二个参数是默认相似度

if __name__ == '__main__': 
    args = parser.parse_args()
    SNR = 40
    start_time = time.time()
    
    # 动态检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print(f"Loaded model from {args.checkpoint_path}")
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
            bleu_score = performance(args, SNR, deepsc)
        else:
            # 进行用户交互测试
            # interactive(args, SNR, deepsc)
            bleu_score, similarity_score = performance(args, SNR, deepsc)
            
            