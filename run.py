#!/usr/bin/env python
# _*_coding:utf-8_*_
# Author   :    Junhui Yu
# Time     :    2023/2/28 15:40


import logging
from flair.datasets import DataLoader
from tqdm import tqdm
import json
from feature_encoder import Encoder
from self_data import MyCorpus

logger = logging.getLogger("YUNLP")

corpus = MyCorpus("ace05", True, 128)

metadata = json.load(open('data/ace05/vocab.json', 'r', encoding='utf-8'))

args = {
    'pos_dim': 50,
    'char_dim': 50,
    'word2vec': 'glove',
    'chars_list': metadata['char'],
    'pos_list': metadata['pos'],
    'pretrain_model': r'D:\BERT\bert-base-uncased',
    'model_max_length': 128,
}

loader = DataLoader(
    dataset=corpus.train,
    shuffle=True,
    drop_last=False,
    batch_size=1,
)

t = tqdm(loader)

encoder = Encoder(args)
print(encoder)

for data in t:
    out = encoder(data)
    print(out)
    break
