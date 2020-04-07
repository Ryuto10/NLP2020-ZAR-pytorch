import random
from math import ceil
from typing import List

import json
import numpy as np
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import h5py

OFFSETS = 10 ** 5
MASK = "[MASK]"
random.seed(2020)
np.random.seed(2020)


class SameSentenceLengthBatchIterator(object):
    """Generate mini-batches with the same sentence length."""

    def __init__(self, dataset: List, batch_size: int = 128, sort_key=None):
        """
        Args:
            dataset: List of dataset.
            batch_size: Size of minibatch.
            sort_key: The function corresponding to the key of sort.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key

        self._n_instances = len(self.dataset)
        self._n_batches = None
        self._order = None
        self._batch_order = None
        self._current_position = 0

        self._setting_order()
        self._setting_batch_order(self._order)

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_position == self._n_batches:
            self._current_position = 0
            raise StopIteration

        indices = self._batch_order[self._current_position]
        batch = [self.dataset[i] for i in indices]
        self._current_position += 1

        return self._batch_processor(batch)

    def shuffle(self):
        """Shuffle the order of the data."""
        order = [random.sample(indices, len(indices)) for indices in self._order]
        self._setting_batch_order(order)
        self._batch_order = random.sample(self._batch_order, len(self._batch_order))

    def _batch_processor(self, batch):
        """Change the minibatch size data to the format used by the model."""
        raise NotImplementedError

    def _setting_order(self):
        """Sort and set the order of dataset according to 'self.sort_key'."""
        sort_values = [self.sort_key(instance) for instance in self.dataset]
        self._order = []
        chunk = []
        sort_value = None

        for idx in np.argsort(sort_values):
            if chunk and sort_values[idx] != sort_value:
                self._order.append(chunk)
                chunk = []
            chunk.append(idx)
            sort_value = sort_values[idx]
        if chunk:
            self._order.append(chunk)

    def _setting_batch_order(self, order: List):
        """Split the dataset into minibatch sizes."""
        self._batch_order = []
        for indices in order:
            if len(indices) > self.batch_size:
                for i in range(ceil(len(indices) / self.batch_size)):
                    self._batch_order.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
            else:
                self._batch_order.append(indices)
        self._n_batches = len(self._batch_order)
        assert sum(len(indices) for indices in self._batch_order) == self._n_instances


class NtcSameSentenceLengthIterator(SameSentenceLengthBatchIterator):
    """Generate mini-batches with the same sentence length about NAIST Text Corpus."""

    def __init__(self, dataset, batch_size=128):
        super(NtcSameSentenceLengthIterator, self).__init__(
            self.get_dataset(dataset), batch_size, sort_key=lambda x: len(x[0][0]))

    def _batch_processor(self, batch):
        xss, yss = zip(*batch)
        return [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]

    @classmethod
    def get_dataset(cls, dataset):
        loaded_dataset = []
        for sentence in dataset:
            if sentence["pas"]:
                pas_seq = sentence["pas"]
                tokens = sentence["tokens"]
                for pas in pas_seq:
                    p_id = int(pas["p_id"])
                    ys = torch.LongTensor([int(a) for a in pas["args"]])
                    ts = torch.LongTensor([int(t) for t in tokens])
                    ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
                    loaded_dataset.append([[ts, ps], ys])

        return loaded_dataset


class PaddingBucketIterator(object):
    """Generate padded mini-batches to minimize padding as much as possible."""

    def __init__(self, dataset: List, sort_key=None, batch_size: int = 128, shuffle=False, padding_value: int = 0):
        self.dataset = dataset
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding_value = padding_value
        self.iterator = torchtext.data.BucketIterator(dataset,
                                                      batch_size=batch_size,
                                                      sort_key=sort_key,
                                                      shuffle=shuffle,
                                                      sort_within_batch=True)
        self.iterator.create_batches()

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.padding(next(self.iterator.batches))

    def create_batches(self):
        self.iterator.create_batches()

    def padding(self, batch):
        """Return a padded mini-batch"""
        raise NotImplementedError


class NtcBucketIterator(PaddingBucketIterator):
    """Generate padded mini-batches to minimize padding as much as possible about NAIST Text Corpus."""

    def __init__(self, dataset: List, batch_size: int = 128, shuffle: bool = False, multi_predicate: bool = False,
                 zero_drop: bool = False, mapping_pseudo_train: str = None, bert: bool = False,
                 loss_stop: bool = False, decode: bool = False, load_cpu: bool = False, full_ys: bool = False,
                 bert_embed_file: str = None, pseudo_bert_embed_file: str = None, joint_softmax: bool = False):
        self.bert = bert
        self.loss_stop = loss_stop
        self.decode = decode
        self.multi_predicate = multi_predicate
        self.zero_drop = zero_drop
        if mapping_pseudo_train:
            if not self.zero_drop:
                raise RuntimeError("If use the train vector instead of the pseudo vector, set zero_drop to True.")
            with open(mapping_pseudo_train) as fi:
                self.mapping_pseudo_train = json.load(fi)
        else:
            self.mapping_pseudo_train = None
        self.load_cpu = load_cpu
        self.full_ys = full_ys
        self.joint_softmax = joint_softmax
        self.bert_vec = h5py.File(bert_embed_file, "r") if bert_embed_file else None
        self.pseudo_bert_vec = h5py.File(pseudo_bert_embed_file, "r") if pseudo_bert_embed_file else None
        super(NtcBucketIterator, self).__init__(
            self.get_dataset(dataset), self.ntc_sort_key, batch_size, shuffle)

    @staticmethod
    def ntc_sort_key(instance):
        return len(instance[0][0])

    def get_dataset(self, dataset):
        if self.zero_drop and not self.load_cpu:
            raise RuntimeError("If 'zero_drop' is True, 'load_cpu' must be set to True.")
        loaded_dataset = []
        for sent in tqdm(dataset):
            for pas in sent["pas"]:
                p_id = int(pas["p_id"])
                if self.joint_softmax:
                    gold = self.create_joint_gold(pas["args"])
                else:
                    gold = torch.LongTensor([int(a) for a in pas["args"]])
                if self.full_ys:
                    ys = [p_id, sent["sentence id"], sent["file name"], gold]
                elif self.decode:
                    ys = [p_id, sent["sentence id"], sent["file name"]]
                else:
                    ys = gold
                ts = torch.LongTensor([int(t) for t in sent["tokens"]])
                ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(sent["tokens"]))])
                xs = [ts, ps]
                if self.loss_stop:
                    ys[sent["insert_ids"]] = 4
                if self.multi_predicate:
                    multi_predicate_positions = [pas["p_id"] for pas in sent["pas"]]
                    multi_predicate_indicator = torch.Tensor([[1.0] if i in multi_predicate_positions else [0.0]
                                                              for i in range(len(sent["tokens"]))])
                    xs.append(multi_predicate_indicator)
                if self.bert:
                    if self.load_cpu:
                        unique_id = sent["unique_id"]
                        if self.mapping_pseudo_train:
                            bert_vecs = torch.Tensor(self.bert_vec.get(str(unique_id))[()]) \
                                if unique_id < OFFSETS else \
                                torch.Tensor(self.bert_vec.get(self.mapping_pseudo_train[str(unique_id)])[()])
                        else:
                            bert_vecs = torch.Tensor(self.bert_vec.get(str(unique_id))[()]) \
                                if unique_id < OFFSETS else \
                                torch.Tensor(self.pseudo_bert_vec.get(str(unique_id))[()])
                        if self.zero_drop:
                            mask_ids = [idx for idx, token in enumerate(sent["surfaces"]) if token == MASK]
                            bert_vecs[mask_ids, :] = 0
                        xs.append(bert_vecs)
                        loaded_dataset.append([xs, ys])
                    else:
                        loaded_dataset.append([xs, ys, sent["unique_id"]])
                else:
                    loaded_dataset.append([xs, ys])

        return loaded_dataset

    def padding(self, batch):
        if self.bert and not self.load_cpu:
            xss, yss, uniques = zip(*batch)
        else:
            xss, yss = zip(*batch)
        xs_len = torch.Tensor([xs[0].size(0) for xs in xss])
        padded_xs = list(map(lambda e: pad_sequence(e, batch_first=True, padding_value=0), zip(*xss)))
        padded_xs.append(xs_len)
        if self.bert and not self.load_cpu:
            padded_xs.append(uniques)

        return [padded_xs, yss]

    def reset_dataset_with_pseudo(self, dataset, pseudo_bert_embed_file: str = None):
        self.dataset = None
        self.iterator = None
        if pseudo_bert_embed_file:
            print("# Load Embedding: {}".format(pseudo_bert_embed_file))
            self.pseudo_bert_vec = h5py.File(pseudo_bert_embed_file, "r")
        super(NtcBucketIterator, self).__init__(
            self.get_dataset(dataset), self.ntc_sort_key, self.batch_size, self.shuffle)

    @staticmethod
    def create_joint_gold(args):
        gold_each_word = torch.LongTensor([4] + [int(a) for a in args])
        gold_all_words = torch.LongTensor([0, 0, 0])
        for case in [0, 1, 2]:
            if case in args:
                gold_all_words[case] = args.index(case) + 1

        return gold_each_word, gold_all_words


# Train
def end2end_single_seq_instance(data, batch_generator):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]


# Test
def test_batch_generator(instance):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    sent_id = instance["sentence id"]
    doc_name = instance["file name"]
    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ts = torch.LongTensor([int(t) for t in tokens])
        ps = torch.Tensor([[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        yield [[ts, ps], [p_id, sent_id, doc_name]]


def test_end2end_single_seq_instance(data, batch_generator):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), yss]


# Multi sentence
def multi_sent_batch_generator(instance, prev_num):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
    prev_sents = instance["prev sentences"][-prev_num:]

    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])

        prev_ws = [int(t) for sent in prev_sents for t in sent]
        prev_ps = [[0.0] for sent in prev_sents for _ in sent]
        prev_ts = [[0.0] for sent in prev_sents for _ in sent]
        prev_sent_distance = [[prev_num * 1.0 - i] for i, sent in enumerate(prev_sents) for _ in sent]

        ws = torch.LongTensor(prev_ws + [int(t) for t in tokens])
        ps = torch.Tensor(prev_ps + [[1.0] if i in predicates else [0.0] for i in range(len(tokens))])
        ts = torch.Tensor(prev_ts + [[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        sent_distance = torch.Tensor(prev_sent_distance + [[0.0] for _ in range(len(tokens))])

        yield [[ws, ps, ts, sent_distance], ys]


def multi_sent_end2end_single_seq_instance(data, batch_generator, prev_num):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence, prev_num))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), torch.stack(yss)]


def multi_sent_test_batch_generator(instance, prev_num):
    pas_seq = instance["pas"]
    tokens = instance["tokens"]
    predicates = list(map(lambda pas: int(pas["p_id"]), pas_seq))
    prev_sents = instance["prev sentences"][-prev_num:]
    sent_id = instance["sentence id"]
    doc_name = instance["file name"]

    for pas in pas_seq:
        p_id = int(pas["p_id"])
        ys = torch.LongTensor([int(a) for a in pas["args"]])

        prev_ws = [int(t) for sent in prev_sents for t in sent]
        prev_ps = [[0.0] for sent in prev_sents for _ in sent]
        prev_ts = [[0.0] for sent in prev_sents for _ in sent]
        prev_sent_distance = [[prev_num * 1.0 - i] for i, sent in enumerate(prev_sents) for _ in sent]

        ws = torch.LongTensor(prev_ws + [int(t) for t in tokens])
        ps = torch.Tensor(prev_ps + [[1.0] if i in predicates else [0.0] for i in range(len(tokens))])
        ts = torch.Tensor(prev_ts + [[1.0] if i == p_id else [0.0] for i in range(len(tokens))])
        sent_distance = torch.Tensor(prev_sent_distance + [[0.0] for _ in range(len(tokens))])

        yield [[ws, ps, ts, sent_distance], [p_id, sent_id, doc_name]]


def multi_sent_test_end2end_single_seq_instance(data, batch_generator, prev_num):
    for sentence in data:
        if sentence["pas"]:
            instances = list(batch_generator(sentence, prev_num))
            xss, yss = zip(*instances)
            yield [list(map(lambda e: torch.stack(e), zip(*xss))), yss]
