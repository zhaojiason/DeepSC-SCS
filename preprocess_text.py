# !usr/bin/env python
# -*- coding:utf-8 _*-

import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='', type=str)
parser.add_argument('--output-train-dir', default='processed_data_1/train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='processed_data_1/test_data.pkl', type=str)
parser.add_argument('--output-vocab', default='processed_data_1/vocab.json', type=str)
parser.add_argument('--input-encoding', default='utf-8', type=str, help='Encoding for input files')
parser.add_argument('--output-encoding', default='utf-8', type=str, help='Encoding for output files')

args = parser.parse_args()

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    # change to lower letter
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

def save_clean_sentences(sentence, save_path, encoding='utf-8'):
    with open(save_path, 'wb') as f:
        pickle.dump(sentence, f)
    print('Saved: %s' % save_path)

def process(text_path, encoding='utf-8'):
    with open(text_path, 'r', encoding=encoding) as fop:
        raw_data = fop.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [normalize_string(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    return raw_data_input

def main(args):
    data_dir = 'pre_data/'
    args.input_data_dir = os.path.join(data_dir, args.input_data_dir)
    args.output_train_dir = os.path.join(data_dir, args.output_train_dir)
    args.output_test_dir = os.path.join(data_dir, args.output_test_dir)
    args.output_vocab = os.path.join(data_dir, args.output_vocab)

    print(args.input_data_dir)
    sentences = []
    print('Preprocess Raw Text')
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith('.txt'): continue
        process_sentences = process(os.path.join(args.input_data_dir, fn), encoding=args.input_encoding)
        sentences += process_sentences

    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1
    sentences = list(a.keys())
    print('Number of sentences: {}'.format(len(sentences)))

    # 使用BERT的分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print('Tokenize sentences')
    tokenized_sentences = []
    for sentence in tqdm(sentences):
        tokens = tokenizer.tokenize(sentence)
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokenized_sentences.append(tokens)

    # 构建词汇表
    print('Build Vocab')
    vocab = tokenizer.vocab
    # 添加特殊标记
    for token, idx in SPECIAL_TOKENS.items():
        vocab[token] = idx

    print('Number of words in Vocab: {}'.format(len(vocab)))

    # 保存词汇表
    if args.output_vocab != '':
        with open(args.output_vocab, 'w', encoding=args.output_encoding) as f:
            json.dump(vocab, f)

    print('Start encoding txt')
    results = []
    for tokens in tqdm(tokenized_sentences):
        # 将分词结果转换为ID
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        results.append(token_ids)

    print('Writing Data')
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]
    save_clean_sentences(train_data, args.output_train_dir, encoding=args.output_encoding)
    save_clean_sentences(test_data, args.output_test_dir, encoding=args.output_encoding)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)