# 给BERT补充其他特征的编码器

## 前言

特征工程是传统的机器学习基石，然而，随着BERT等预训练语言模型的发展，文本语义表示得到了极大的改善，本文将实践一种多特征编码器。

## 1、POS特征生成

### 代码：

```
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_pos_tag(sentence):
    doc = nlp(sentence)
    print(list(doc.sents))
    pos = []
    for word in list(doc.sents)[0]:
        pos.append(word.tag_)
    return pos

if __name__ == '__main__':
    print(spacy_pos_tag('Iraq Blames Market Blast on Coalition .'))

```

### 输出：

```
[Iraq Blames Market Blast on Coalition .]
['NNP', 'NNP', 'NNP', 'NNP', 'IN', 'NNP', '.']
```

## 2、NLP中常见的特征

- Word特征
- Token特征
- Char特征
- POS特征

## 3、数据处理

本文与ACE2005数据集为例，处理该数据的格式样例如下：

```json
[
    {
        "tokens": ["Their", "military", "service", "goes", "back", "to", "the", "Vietnam", "era", "."],
        "pos": ["PRP$", "JJ", "NN", "VBZ", "RB", "TO", "DT", "NNP", "NN", "."],
        "entities": [
            {"type": "PER", "start": 0, "end": 1},
            {"type": "ORG", "start": 1, "end": 2},
            {"type": "GPE", "start": 7, "end": 8}
        ],
        "ltokens": ["WOODRUFF", "We", "know", "that", "some", "of", "the", "American", "troops", "now", "fighting",
                    "in", "Iraq", "are", "longtime", "veterans", "of", "warfare", ",", "probably", "not", "most", ",",
                    "but",
                    "some", "."],
        "rtokens": ["Others", ",", "though", ",", "are", "novices", "."]
    },
    ...
]

```

词汇表样例：

```json
{
  "pos": [
    "FW",
    "VBZ",
    "PRP",
    "DT",
    "VBD",
    "IN",
    "NNS",
    ".",
    "CD",
    "``",
    "WDT",
    "''",
    "POS",
    "VBN",
    "WP",
    "RBS",
    "NNPS",
    "JJS",
    "RB",
    ":",
    "JJR",
    "PRP$",
    "WP$",
    "TO",
    "NN",
    "WRB",
    "VB",
    "-RRB-",
    "JJ",
    "MD",
    "$",
    "-LRB-",
    "VBP",
    "RP",
    "RBR",
    "LS",
    "EX",
    "NNP",
    "CC",
    "#",
    ",",
    "SYM",
    "PDT",
    "UH",
    "VBG"
  ],
  "char": [
    "o",
    "8",
    "/",
    ">",
    "^",
    "@",
    "D",
    "Y",
    "?",
    "c",
    ".",
    "Q",
    "R",
    "0",
    "e",
    "W",
    "5",
    "X",
    "p",
    "r",
    "M",
    "C",
    "|",
    "i",
    "Z",
    "l",
    "9",
    "j",
    "4",
    "$",
    "%",
    "y",
    "P",
    "s",
    "6",
    "G",
    "!",
    "m",
    "S",
    "I",
    "*",
    "h",
    ",",
    "k",
    "_",
    "a",
    "B",
    "v",
    "d",
    "N",
    "1",
    "7",
    "-",
    "q",
    "'",
    "2",
    "w",
    ":",
    "<",
    "3",
    ";",
    "t",
    "u",
    "+",
    "J",
    "&",
    "`",
    "F",
    "=",
    "V",
    "f",
    "L",
    "U",
    "T",
    "n",
    "x",
    "b",
    "K",
    "#",
    "E",
    "g",
    "H",
    "O",
    "z",
    "A"
  ],
  "type": [
    "FAC",
    "VEH",
    "ORG",
    "GPE",
    "PER",
    "WEA",
    "LOC"
  ]
}
```



## 4、编码器代码实现

### 4.1.数据集构建代码

**1、self_data.py**

```python
#!/usr/bin/env python
# _*_coding:utf-8_*_
# Author   :    Junhui Yu
# Time     :    2023/2/28 16:54

import json
import logging
from typing import List, TextIO, Union
from flair.data import Corpus, FlairDataset, Sentence
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm

logger = logging.getLogger("YUNLP")


class MySentence(Sentence):
    def __init__(self, text: Union[str, List[str]], entities: list[dict] = None, *args, **kargs):
        super(MySentence, self).__init__(text, *args, **kargs)
        self.entities = entities
        self._nested_pairs = None

    @property
    def sentence_tokens(self):
        return list(map(lambda a: a.text, self.tokens))

    @property
    def previous_tokens(self):
        return list(map(lambda a: a.text, self.previous_sentence().tokens))

    @property
    def next_tokens(self):
        return list(map(lambda a: a.text, self.next_sentence().tokens))


class MyDataset(FlairDataset):
    def __init__(self, file: TextIO, name: str):
        self.name = name
        data_list = json.load(file)
        self._sentences = self.format(data_list)

    def format(self, data_list: list[dict]) -> list[MySentence]:
        sentences = list()
        for data in tqdm(data_list):
            sentence = MySentence(data['tokens'])
            sentence._previous_sentence = MySentence(data['ltokens'])
            sentence._next_sentence = MySentence(data['rtokens'])
            sentence.add_label('pos', data['pos'])
            if 'tags' in data.keys():
                sentence.add_label('tags', data['tags'])
            sentence.entities = data['entities']
            sentences.append(sentence)

        return sentences

    def add_data(self, sentences: list[Sentence]):
        self._sentences += sentences

    @property
    def sentences(self):
        return self._sentences

    def __getitem__(self, index: int) -> MySentence:
        return self._sentences[index]

    def __len__(self) -> int:
        return len(self._sentences)

    def is_in_memory(self) -> bool:
        return True


class MyCorpus(Corpus):
    def __init__(self, dataset_name: str, concat: bool = False, max_length: int = None):
        dataset_path = f"data/{dataset_name}"
        try:
            super(MyCorpus, self).__init__(name=dataset_name)
        except RuntimeError:
            pass

        self._train: MyDataset = self.create_dataset(dataset_path, 'train')
        self._dev: MyDataset = self.create_dataset(dataset_path, 'dev')
        self._test: MyDataset = self.create_dataset(dataset_path, 'test')

        with open(f"{dataset_path}/vocab.json", encoding='utf-8') as file:
            metadata = json.load(file)
        types = metadata.pop('type')
        types_idx = range(len(types))
        self.metadata = {
            'types2idx': dict(zip(types, types_idx)),
            'idx2types': dict(zip(types_idx, types)),
            'chars_list': metadata['char'],
            'pos_list': metadata['pos']
        }

        if concat:
            self._train.add_data(self._dev.sentences)
            self._dev = None

        if max_length is not None:
            self.filter(lambda a: len(a) < max_length, ['train'])

    @staticmethod
    def create_dataset(dataset_path: str, prefix: str):
        try:
            with open(f"{dataset_path}/{prefix}.json", encoding="utf-8") as file:
                dataset = MyDataset(file, prefix)
        except FileNotFoundError as _:
            return None
        return dataset

    def filter(self, condition: callable, dataset_names: list[str] = None):
        if dataset_names is None:
            dataset_names = ['train']
        for name in dataset_names:
            dataset = self.__dict__[f'_{name}']
            self.__dict__[f'_{name}'] = self._filter(dataset, condition)
        logger.info(self)

    @staticmethod
    def _filter(dataset: MyDataset, condition: callable) -> Dataset:
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0

        for sentence in dataset:
            if condition(sentence):
                non_empty_sentence_indices.append(index)
            else:
                empty_sentence_indices.append(index)
            index += 1
        subset = Subset(dataset, non_empty_sentence_indices)
        return subset

    @property
    def datasets(self) -> list[MyDataset]:
        datasets = map(lambda a: self.__dict__[f'_{a}'], ['train', 'dev', 'test'])
        return list(filter(lambda a: a is not None, datasets))

    @property
    def train(self) -> MyDataset:
        return self._train

    @property
    def dev(self) -> MyDataset:
        return self._dev

    @property
    def test(self) -> MyDataset:
        return self._test

```

### 4.2、编码器模型代码

```python
#!/usr/bin/env python
# _*_coding:utf-8_*_
# Author   :    Junhui Yu
# Time     :    2023/2/28 13:51

from typing import Optional
import logging
import torch
import transformers
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchtext import vocab
from typing import Union
import config
from self_data import MySentence

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger("YUNLP")


def build_tokenizer(pretrain_model: str, cache_dir: str):
    return Tokenizer({
        'return_tensors': 'pt', 'padding': True
    },
        AutoTokenizer.from_pretrained(pretrain_model, cache_dir=cache_dir, padding_side='right')
    )


class Tokenizer:
    xila_table = str.maketrans(
        "âêîôûŷäëïöüÿñ",
        "aeiouyaeiouyn"
    )
    special_tokens = {
        '“': '"',
        '”': '"',
        '‘': ''',
        '’': ''',
        '—': '-',
        '…': '...',
        '……': '...',
        '�': '?'
    }

    def __init__(self, encode_kargs: dict, tokenizer: PreTrainedTokenizerFast = None):
        self.encode_kargs = encode_kargs
        self.tokenizer = tokenizer

    def __call__(self, batch_texts: Union[list[str], list[list[str]]]):
        batch_texts = list(map(self.preprocess, batch_texts))
        return self.tokenizer(batch_texts, is_split_into_words=True, **self.encode_kargs)

    def batch_decode(self, *args, **kargs):
        batch_texts = self.tokenizer.batch_decode(*args, **kargs)
        return [text.split(" ") for text in batch_texts]

    @classmethod
    def preprocess(cls, raw_text: list[str]):
        text = list(map(lambda a: a.translate(cls.xila_table), raw_text))
        text = list(map(lambda a: cls.special_tokens.get(a, a), text))
        return text


class Encoder(nn.Module):
    def __init__(self, embedding_kargs: dict):
        super(Encoder, self).__init__()
        embedding_kargs.update({'cache_dir': f"./vec/"})
        self.embedding = FusionEmbedding(**embedding_kargs)
        self.hidden_size = self.embedding.token2vec.pretrain.config.hidden_size
        embedding_length = self.embedding.embedding_length

        self.out_rnn = nn.GRU(embedding_length, embedding_length, bidirectional=True, batch_first=True)
        self.transforms = nn.Sequential(
            nn.Linear(2 * embedding_length, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(self, batch_sentences: list[MySentence]) -> tuple[Tensor, Tensor]:
        lengths = torch.as_tensor(list(map(len, batch_sentences)))

        batch_embeds, batch_masks = self.embedding(batch_sentences)
        batch_embeds = pack_padded_sequence(batch_embeds, lengths, batch_first=True, enforce_sorted=False)
        batch_embeds = pad_packed_sequence(self.out_rnn(batch_embeds)[0], batch_first=True)[0]

        batch_hiddens = self.transforms(batch_embeds)
        return batch_hiddens, batch_masks


class FusionEmbedding(nn.Module):
    def __init__(
            self,
            cache_dir: str,
            model_max_length: int,
            pretrain_model: str,
            pos_dim: Optional[int],
            char_dim: Optional[int],
            word2vec: Optional[str],
            chars_list: Optional[list[str]],
            pos_list: Optional[list[str]],
    ):
        super(FusionEmbedding, self).__init__()
        self.token2vec = Token2Vec(cache_dir, pretrain_model, model_max_length)
        self._embedding_length = self.token2vec.hidden_size

        if word2vec is not None:
            self.word2vec = Word2Vec(cache_dir, word2vec)
            self._embedding_length += self.word2vec.word_dim

        if char_dim is not None and chars_list is not None:
            self.char2vec = Char2Vec(chars_list, char_dim)
            self._embedding_length += char_dim * 2

        if pos_dim is not None and pos_list is not None:
            self.pos2vec = Pos2Vec(pos_list, pos_dim)
            self._embedding_length += pos_dim

    def forward(self, batch_sentences: list[MySentence]) -> tuple[Tensor, Tensor]:
        token2vec, mask = self.token2vec(batch_sentences)
        embeds = [token2vec]

        if hasattr(self, 'word2vec'):
            embeds.append(self.word2vec(batch_sentences, token2vec.device))

        if hasattr(self, 'char2vec'):
            embeds.append(self.char2vec(batch_sentences))

        if hasattr(self, 'pos2vec'):
            embeds.append(self.pos2vec(batch_sentences))

        batch_embeds = torch.cat(embeds, dim=-1)
        return batch_embeds, mask

    @property
    def embedding_length(self):
        return self._embedding_length


class Token2Vec(nn.Module):
    def __init__(self, cache_dir: str, pretrain_model: str, model_max_length: int):
        super(Token2Vec, self).__init__()
        self.model_max_length = model_max_length
        self.model_max_length_copy = model_max_length

        cache_dir = cache_dir + pretrain_model
        self.tokenizer = build_tokenizer(pretrain_model, cache_dir)
        self.pretrain = transformers.AutoModel.from_pretrained(pretrain_model, cache_dir=cache_dir)

    def forward(self, batch_sentences: list[MySentence]) -> tuple[Tensor, Tensor]:
        batch_context = self.span_context(batch_sentences)
        lengths = list(map(len, batch_context))

        encoding = self.tokenizer(batch_context).to(next(self.parameters()).device)
        output = self.pretrain(output_hidden_states=True, **encoding)
        hidden_states = torch.stack(output.hidden_states[-4:], dim=-1)
        hidden_state = torch.mean(hidden_states, dim=-1)

        token_embeds, sub_lengths = list(), list()
        for i, length in enumerate(lengths):
            for j in range(length):
                s, e = encoding.word_to_tokens(i, j)
                token_embeds.append(hidden_state[i, s:e])
                sub_lengths.append(e - s)

        sub_lengths = torch.as_tensor(sub_lengths, device=hidden_state.device)
        token_embeds = pad_sequence(token_embeds, padding_value=0)
        token_embeds = torch.sum(token_embeds, dim=0) / sub_lengths.unsqueeze(-1)

        token_embeds = token_embeds.split(lengths, dim=0)
        token_embeds = pad_sequence(token_embeds, batch_first=True)
        return self.span_select(token_embeds, batch_sentences)

    def span_context(self, batch_sentences: list[MySentence]) -> list[list[str]]:
        batch_context = list()
        for sentence in batch_sentences:
            context = sentence.sentence_tokens
            if len(context) + len(sentence.next_sentence()) < self.model_max_length:
                context = context + sentence.next_tokens

                if len(sentence.previous_sentence()) + len(context) < self.model_max_length:
                    context = sentence.previous_tokens + context
                    offset = len(sentence.previous_sentence())
                    sentence.start_pos, sentence.end_pos = offset, offset + len(sentence)
                else:
                    offset = self.model_max_length - len(context)
                    context = sentence.previous_tokens[-offset:] + context
                    sentence.start_pos, sentence.end_pos = offset, offset + len(sentence)
            else:
                sentence.start_pos, sentence.end_pos = 0, len(sentence)

            batch_context.append(context)
        return batch_context

    @staticmethod
    def span_select(batch_embeds: Tensor, batch_sentences: list[MySentence]) -> tuple[Tensor, Tensor]:
        hiddens, mask = list(), list()
        for sentence, embeds in zip(batch_sentences, batch_embeds):
            s, e = sentence.start_pos, sentence.end_pos
            hiddens.append(embeds[s:e])
            mask.append(torch.ones(e - s, device=embeds.device, dtype=torch.bool))
        hiddens = pad_sequence(hiddens, batch_first=True)
        mask = pad_sequence(mask, padding_value=False, batch_first=True)
        return hiddens, mask

    @property
    def hidden_size(self):
        return self.pretrain.config.hidden_size

    def train(self, mode: bool = True):
        self.model_max_length = self.model_max_length_copy
        super().train(mode)

    def eval(self):
        self.model_max_length = 512
        super().eval()


class Word2Vec(nn.Module):
    def __init__(self, cache_dir: str, word2vec: str):
        super().__init__()
        self.word2vec = word2vec
        if word2vec == 'glove':
            self._word_dim = 50
            self.vectors = vocab.GloVe(name='6B', dim=50, cache=cache_dir + "glove")

    def forward(self, batch_sentences: list[MySentence], device: str) -> Tensor:
        embeds = list(map(self.get_vectors_by_tokens, batch_sentences))
        return pad_sequence(embeds, batch_first=True).to(device)

    def get_vectors_by_tokens(self, sentence: MySentence):
        if self.word2vec in ['glove', 'chinese']:
            return self.vectors.get_vecs_by_tokens(sentence.sentence_tokens)
        elif self.word2vec == 'bio':
            indices = [self.word2idx.get(token, 0) for token in sentence.sentence_tokens]
            return self.embeds(torch.as_tensor(indices, device=self.embeds.weight.device))

    @property
    def word_dim(self):
        return self._word_dim


class Char2Vec(nn.Module):
    def __init__(self, chars_list: list[str], char_dim: int):
        super().__init__()
        self.char2idx = dict(zip(chars_list, range(len(chars_list))))
        self.char2vec = nn.Embedding(len(self.char2idx), char_dim)
        self.char_rnn = nn.GRU(char_dim, char_dim, bidirectional=True, batch_first=True)

    def forward(self, batch_sentences: list[MySentence]) -> Tensor:
        device = self.char2vec.weight.device
        lengths = list(map(len, batch_sentences))

        indices, char_lengths = list(), list()
        for sentence in batch_sentences:
            for token in sentence:
                char_text = token.text
                indices.append(torch.as_tensor([self.char2idx[c] for c in char_text], device=device))
                char_lengths.append(len(char_text))

        char_embeds = self.char2vec(pad_sequence(indices))
        char_lengths = torch.as_tensor(char_lengths, device=char_embeds.device)

        char_embeds = pack_padded_sequence(char_embeds, char_lengths.cpu(), enforce_sorted=False)
        char_embeds = pad_packed_sequence(self.char_rnn(char_embeds)[0], padding_value=0)[0]
        char_embeds = torch.sum(char_embeds, dim=0) / char_lengths.unsqueeze(-1)

        char_embeds = torch.split(char_embeds, lengths, dim=0)
        return pad_sequence(char_embeds, batch_first=True)


class Pos2Vec(nn.Module):
    def __init__(self, pos_list: list[str], pos_dim: int):
        super().__init__()
        self.pos2idx = dict(zip(pos_list, range(len(pos_list))))
        self.pos2vec = nn.Embedding(len(self.pos2idx), pos_dim)

    def forward(self, batch_sentences: list[MySentence]) -> tuple[Tensor, Tensor]:
        device = self.pos2vec.weight.device
        indices = list()
        for sentence in batch_sentences:
            pos_tags = sentence.get_labels('pos')[0].value
            pos_indices = list(map(lambda a: self.pos2idx[a], pos_tags))
            indices.append(torch.as_tensor(pos_indices, dtype=torch.long, device=device))
        return self.pos2vec(pad_sequence(indices).T)

```

## 5、模型结构与测试

1. 编码器模型结构

   ```
   Encoder(
     (embedding): FusionEmbedding(
       (token2vec): Token2Vec(
         (pretrain): BertModel(
           (embeddings): BertEmbeddings(
             (word_embeddings): Embedding(30522, 768, padding_idx=0)
             (position_embeddings): Embedding(512, 768)
             (token_type_embeddings): Embedding(2, 768)
             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
             (dropout): Dropout(p=0.1, inplace=False)
           )
           (encoder): BertEncoder(
             (layer): ModuleList(
               (0): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (1): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (2): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (3): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (4): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (5): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (6): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (7): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (8): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (9): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (10): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
               (11): BertLayer(
                 (attention): BertAttention(
                   (self): BertSelfAttention(
                     (query): Linear(in_features=768, out_features=768, bias=True)
                     (key): Linear(in_features=768, out_features=768, bias=True)
                     (value): Linear(in_features=768, out_features=768, bias=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                   (output): BertSelfOutput(
                     (dense): Linear(in_features=768, out_features=768, bias=True)
                     (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                     (dropout): Dropout(p=0.1, inplace=False)
                   )
                 )
                 (intermediate): BertIntermediate(
                   (dense): Linear(in_features=768, out_features=3072, bias=True)
                   (intermediate_act_fn): GELUActivation()
                 )
                 (output): BertOutput(
                   (dense): Linear(in_features=3072, out_features=768, bias=True)
                   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                   (dropout): Dropout(p=0.1, inplace=False)
                 )
               )
             )
           )
           (pooler): BertPooler(
             (dense): Linear(in_features=768, out_features=768, bias=True)
             (activation): Tanh()
           )
         )
       )
       (word2vec): Word2Vec()
       (char2vec): Char2Vec(
         (char2vec): Embedding(85, 50)
         (char_rnn): GRU(50, 50, batch_first=True, bidirectional=True)
       )
       (pos2vec): Pos2Vec(
         (pos2vec): Embedding(45, 50)
       )
     )
     (out_rnn): GRU(968, 968, batch_first=True, bidirectional=True)
     (transforms): Sequential(
       (0): Linear(in_features=1936, out_features=768, bias=True)
       (1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
     )
   )
   ```

2. 测试代码

   ```python
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
   
   ```

   

3. 输出

   ```
   (tensor([[[-0.6931, -0.6064, -0.3340,  ...,  0.8820,  2.0498,  0.7626],
            [-0.4846, -0.3343,  0.8345,  ...,  0.3118,  2.0231,  0.4113],
            [-1.3114, -0.7566,  0.1274,  ...,  0.2121,  1.9899,  0.1858],
            ...,
            [-0.4345,  0.6677,  0.0272,  ..., -0.1907,  1.5457, -1.8503],
            [-0.1394,  0.6579,  1.1406,  ..., -0.4034,  1.7865, -0.8288],
            [-0.7872,  0.0053,  1.2921,  ..., -1.2903,  1.2204, -0.6980]]],
          grad_fn=<NativeLayerNormBackward0>), tensor([[True, True, True, True, True, True, True, True, True, True]]))
   ```

   

## 6、相关工具

bert-base：https://huggingface.co/bert-base-uncased

glove：http://nlp.stanford.edu/data/glove.6B.zip

pos embedding：使用pytorch自带的embedding

## 结论

本文实践了一种多特征的编码器，在NLP的下游任务上经过测试有性能的提升。注：具体任务具体分析，不一定完全能大幅度提升性能，但是也不会降性能。