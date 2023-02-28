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
