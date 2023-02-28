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
