from collections import Counter
import io
import json
import numpy as np
import torch
from tqdm import tqdm
import Constants

# https://github.com/Smerity/keras_snli/blob/master/snli_rnn.py
def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = extract_tokens_from_binary_parse(data['sentence1_binary_parse'])
        s2 = extract_tokens_from_binary_parse(data['sentence2_binary_parse'])
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)
    
def get_data(fn, isTrain=True, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    token_counter = Counter()
    if isTrain:
        for _, s1, s2 in tqdm(raw_data):
            token_counter.update(s1+s2)
    print("left 95% length: {}, right 95% length: {}".format(sorted([len(x) for x in left])[int(len(left) * 0.95)], 
        sorted([len(x) for x in right])[int(len(right) * 0.95)]))

    Y = np.array([Constants.LABELS[l] for l, s1, s2 in raw_data])

    return left, right, Y, token_counter

def build_vocab(token_counter):
    print("There are {} unique words. ".format(len(token_counter)))
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = Constants.PAD_IDX 
    token2id['<unk>'] = Constants.UNK_IDX
    return token2id, id2token

def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for idx in range(len(tokens_data[0])):
        s1_index_list = [token2id[token] if token in token2id else Constants.UNK_IDX for token in tokens_data[0][idx]]
        s2_index_list = [token2id[token] if token in token2id else Constants.UNK_IDX for token in tokens_data[1][idx]]
        indices_data.append([s1_index_list, s2_index_list, tokens_data[2][idx]])
    return indices_data

def load_vectors(fname, id2token):
    """
    fasttext wiki-news-300d-1M.vec
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in id2token:
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def get_pretrain_emb(pretrained, token, notPretrained):
    if token == '<pad>':
        notPretrained.append(0)
        return [0] * 300
    if token in pretrained:
        notPretrained.append(0)
        return pretrained[token]
    else:
        notPretrained.append(1)
        return [0] * 300