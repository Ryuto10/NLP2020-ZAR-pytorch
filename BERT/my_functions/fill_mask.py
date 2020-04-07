import copy
import json
import os
from typing import List
import random

import torch
import torch.nn.functional as F
from pyknp import Juman


class FillerConfig(object):
    def __init__(self, subword_vocab, pre_trained_vocab_file, json_file, sent_options, instance_options, all_subwords):
        self.subword_vocab = subword_vocab
        self.pre_trained_vocab_file = pre_trained_vocab_file
        self.json_file = json_file
        self.sent_options = sent_options
        self.instance_options = instance_options
        self.all_subwords = all_subwords


class MaskFiller(object):
    def __init__(self, config: FillerConfig):
        self.subword_index2vocab = {v: k for k, v in config.subword_vocab.items()}
        self.pre_trained_vocab = self._set_vocab(config.pre_trained_vocab_file)
        self.json_file = open(config.json_file)
        self.sent_options = config.sent_options
        self.instance_options = config.instance_options
        self.all_subwords = config.all_subwords

        self.tokenizer = Juman(command="juman")
        self.filtered_ids = [v for k, v in config.subword_vocab.items() if not self._filtering_subword(k)]
        self.json_line = next(self.json_file).rstrip("\n")
        self.sent_id = 0
        self.n_instance = 0
        self.score_in_same_sentence = []
        self.predict_token_distribution = {}
        for k, v in config.subword_vocab.items():
            if v not in self.filtered_ids:
                self.predict_token_distribution[k] = 0

    def __call__(self, scores) -> List[str]:
        instances = None

        # If length of mask ids is 0, the sentence is skipped.
        while len(self.sent_options[self.sent_id][0]) == 0:
            self.json_line = next(self.json_file).rstrip("\n")
            self.sent_id += 1

        sent_id, mask_id = self.instance_options[self.n_instance]

        if self.sent_id != sent_id:
            mask_ids = self.sent_options[self.sent_id][0]
            assert len(self.score_in_same_sentence) == len(mask_ids)
            assert all(self.score_in_same_sentence[i][0] == mask_id for i, mask_id in enumerate(mask_ids))
            instances = self.get_instances()
            self.score_in_same_sentence = []
            self.json_line = next(self.json_file).rstrip("\n")
            self.sent_id += 1

        self.score_in_same_sentence.append((mask_id, self._get_mask_score(scores)))
        self.n_instance += 1

        return instances

    def get_instances(self) -> List[str]:
        raise NotImplementedError

    def pop(self):
        mask_ids = self.sent_options[self.sent_id][0]
        assert len(self.score_in_same_sentence) == len(mask_ids)
        assert all(self.score_in_same_sentence[i][0] == mask_id for i, mask_id in enumerate(mask_ids))
        instances = self.get_instances()
        self.score_in_same_sentence = []
        self.sent_id += 1

        # If length of mask ids is 0, the sentence is skipped.
        if len(self.sent_options) != self.sent_id:
            while self.sent_id < len(self.sent_options):
                assert len(self.sent_options[self.sent_id][0]) == 0
                self.sent_id += 1

        assert len(self.sent_options) == self.sent_id
        self.json_file.close()
        
        return instances

    @staticmethod
    def _set_vocab(file):
        """Open file of pre-trained vocab and convert to dict format."""
        if not file:
            raise ValueError("Please input pre-trained word indices.")
        if not os.path.exists(file):
            raise FileNotFoundError("{} doesn't exist.".format(file))

        print("\n# Load :'{}'".format(file))
        vocab = {}
        with open(file) as fi:
            for line in fi:
                word, idx = line.rstrip().split("\t")
                vocab[word] = idx

        return vocab

    def _filtering_subword(self, token) -> bool:
        """True if the condition is met, False otherwise."""
        if token not in self.pre_trained_vocab:
            return False
        morphs = self.tokenizer.analysis(token)
        if len(morphs) > 1 or morphs[0].hinsi != "名詞":
            return False
        return True

    def _get_mask_score(self, scores):
        for idx, token in enumerate(self.all_subwords[self.n_instance]):
            if token == "[MASK]":
                return scores[idx]

        raise RuntimeError("Can't find [MASK] token.")

    def _get_instance_with_replace_tokens(self, idx_token_pairs):
        instance = json.loads(self.json_line)
        sent_len = self.sent_options[self.sent_id][2]
        if sent_len != len(instance["tokens"]):
            raise RuntimeError("Sentence length mismatch.")

        for idx, token in idx_token_pairs:
            instance["tokens"][idx] = self.pre_trained_vocab[token]

        return instance


class BestNTokenFiller(MaskFiller):
    def __init__(self, config: FillerConfig, n_best=5, mode="json"):
        super(BestNTokenFiller, self).__init__(config)
        self.n_best = n_best
        self.mode = mode
        self.check_span = round(len(self.sent_options) / 15)

    def get_instances(self) -> List[str]:
        instances = []
        if self.mode == "json":
            for mask_id, score in self.score_in_same_sentence:
                for predict_token in self.get_best_n_tokens(score, mask_id):
                    instance = self._get_instance_with_replace_tokens([(mask_id, predict_token)])
                    instances.append(json.dumps(instance))
                    self.predict_token_distribution[predict_token] += 1
        elif self.mode == "surface":
            # sentence = self.get_sentence()
            # instances.append(sentence)

            # for ACL
            for mask_id, score in self.score_in_same_sentence:
                surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
                predict_tokens = self.get_best_n_tokens(score, mask_id)
                for predict_token in predict_tokens:
                    surfaces[mask_id] = predict_token
                    sentence = " ".join(surfaces)
                    instances.append(sentence)
        else:
            raise ValueError("Unsupported mode: {}".format(self.mode))

        if self.sent_id % self.check_span == 0:
            sentence = self.get_sentence()
            print(sentence)

        return instances

    def get_sentence(self):
        surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
        for mask_id, score in self.score_in_same_sentence:
            predict_tokens = self.get_best_n_tokens(score, mask_id)
            surfaces[mask_id] += " (" + " | ".join(predict_tokens) + ")"
        sentence = " ".join(surfaces)

        return sentence

    def get_best_n_tokens(self, score, mask_id):
        score[self.filtered_ids] = -10**3
        best_ids = torch.argsort(score, descending=True)
        gold_token = self.sent_options[self.sent_id][1][mask_id]
        predict_tokens = []
        for rank in range(score.size(0)):
            predict_token = self.subword_index2vocab[int(best_ids[rank])]
            if predict_token != gold_token:
                predict_tokens.append(predict_token)
            if len(predict_tokens) == self.n_best:
                break

        return predict_tokens


class MultiTokenFiller(MaskFiller):
    def __init__(self, config: FillerConfig, n_sample=10, prob=0.5, mode="json"):
        super(MultiTokenFiller, self).__init__(config)
        self.n_sample = n_sample
        self.prob = prob
        self.mode = mode
        self.check_span = round(len(self.sent_options) / 15)

    def get_instances(self) -> List[str]:
        instances = []
        for pairs in self.get_n_sampling_tokens():
            if self.mode == "json":
                instance = self._get_instance_with_replace_tokens(pairs)
                instance = json.dumps(instance)
            elif self.mode == "surface":
                instance = self.get_sentence(pairs)
            else:
                raise ValueError("Unsupported mode: {}".format(self.mode))

            if self.sent_id % self.check_span == 0:
                sentence = self.get_sentence(pairs)
                print(sentence)

            instances.append(instance)

        return instances

    def get_sentence(self, pairs):
        surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
        for idx, token in pairs:
            surfaces[idx] += " ({})".format(token)
        sentence = " ".join(surfaces)

        return sentence

    def get_n_sampling_tokens(self):
        scores = torch.stack([score for _, score in self.score_in_same_sentence])
        scores = F.softmax(scores, 1)
        scores[:, self.filtered_ids] = 0
        sampling_matrix = torch.multinomial(scores, self.n_sample * 10).tolist()

        gold_tokens = [(mask_id, self.sent_options[self.sent_id][1][mask_id])
                       for mask_id in self.sent_options[self.sent_id][0]]
        results = []
        for (mask_id, gold_token), sampling_ids in zip(gold_tokens, sampling_matrix):
            predict_tokens = []
            for idx in sampling_ids:
                predict_token = self.subword_index2vocab[idx]
                if idx not in self.filtered_ids and predict_token != gold_token:
                    predict_tokens.append(predict_token)
                if len(predict_tokens) == self.n_sample:
                    break
            if len(predict_tokens) != self.n_sample:
                raise RuntimeError("Insufficient number of prediction tokens")
            if gold_token in self.pre_trained_vocab:
                g_or_p = torch.bernoulli(torch.Tensor([self.prob] * self.n_sample))
            else:
                g_or_p = [0] * self.n_sample
            pairs = []
            for n, p_token in zip(g_or_p, predict_tokens):
                if n == 1:
                    pairs.append((mask_id, gold_token))
                else:
                    pairs.append((mask_id, p_token))
                    self.predict_token_distribution[p_token] += 1
            results.append(pairs)

        results = list(zip(*results))

        return results


class BestNTokenPredicateFiller(MaskFiller):
    def __init__(self, config: FillerConfig, n_best=5, mode="json"):
        super(BestNTokenPredicateFiller, self).__init__(config)
        self.n_best = n_best
        self.mode = mode
        self.check_span = round(len(self.sent_options) / 15)
        self.filtered_ids = [v for k, v in config.subword_vocab.items() if not self.filtering_subword(k)]

    def get_instances(self) -> List[str]:
        instances = []
        if self.mode == "json":
            for mask_id, score in self.score_in_same_sentence:
                for predict_token in self.get_best_n_tokens(score, mask_id):
                    instance = self.get_instance_with_replace_tokens([(mask_id, predict_token)])
                    instances.append(json.dumps(instance))
                    self.predict_token_distribution[predict_token] += 1
        elif self.mode == "surface":
            sentence = self.get_sentence()
            instances.append(sentence)
        else:
            raise ValueError("Unsupported mode: {}".format(self.mode))

        if self.sent_id % self.check_span == 0:
            sentence = self.get_sentence()
            print(sentence)

        return instances

    def get_instance_with_replace_tokens(self, idx_token_pairs):
        instance = json.loads(self.json_line)
        sent_len = self.sent_options[self.sent_id][2]
        if sent_len != len(instance["tokens"]):
            raise RuntimeError("Sentence length mismatch.")

        for idx, token in idx_token_pairs:
            instance["tokens"][idx] = self.pre_trained_vocab[token]
            instance["pas"] = [dic for dic in instance["pas"] if dic["p_id"] != idx]

        return instance

    def filtering_subword(self, token) -> bool:
        """True if the condition is met, False otherwise."""
        if token not in self.pre_trained_vocab:
            return False
        morphs = self.tokenizer.analysis(token)
        if len(morphs) > 1:
            return False
        return True

    def get_sentence(self):
        surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
        for mask_id, score in self.score_in_same_sentence:
            predict_tokens = self.get_best_n_tokens(score, mask_id)
            surfaces[mask_id] += " (" + " | ".join(predict_tokens) + ")"
        sentence = " ".join(surfaces)

        return sentence

    def get_best_n_tokens(self, score, mask_id):
        score[self.filtered_ids] = -10**3
        best_ids = torch.argsort(score, descending=True)
        gold_token = self.sent_options[self.sent_id][1][mask_id]
        gold_token_hinsi = self.tokenizer.analysis(gold_token)[0].hinsi
        predict_tokens = []
        for rank in range(score.size(0)):
            predict_token = self.subword_index2vocab[int(best_ids[rank])]
            if predict_token != gold_token and self.tokenizer.analysis(predict_token)[0].hinsi == gold_token_hinsi:
                predict_tokens.append(predict_token)
            if len(predict_tokens) == self.n_best:
                break

        return predict_tokens


class RandomNTokenFiller(MaskFiller):
    def __init__(self, config: FillerConfig, n_sample=5, mode="json"):
        super(RandomNTokenFiller, self).__init__(config)
        self.n_sample = n_sample
        self.mode = mode
        self.check_span = round(len(self.sent_options) / 15)
        self.filtered_ids = [v for k, v in config.subword_vocab.items() if not self.filtering_subword(k)]
        self.subword_ids = [idx for idx in self.subword_index2vocab if idx not in self.filtered_ids]
        self.predict_token_distribution = {}
        for k, v in config.subword_vocab.items():
            if v not in self.filtered_ids:
                self.predict_token_distribution[k] = 0

    def filtering_subword(self, token) -> bool:
        """True if the condition is met, False otherwise."""
        if token not in self.pre_trained_vocab:
            return False
        morphs = self.tokenizer.analysis(token)
        if len(morphs) > 1 and morphs[0].hinsi != "助詞":
            return False
        return True

    def get_instances(self) -> List[str]:
        instances = []
        if self.mode == "json":
            for mask_id, score in self.score_in_same_sentence:
                for predict_token in self.get_random_n_tokens(score, mask_id):
                    instance = self._get_instance_with_replace_tokens([(mask_id, predict_token)])
                    instances.append(json.dumps(instance))
                    self.predict_token_distribution[predict_token] += 1
        elif self.mode == "surface":
            sentence = self.get_sentence()
            instances.append(sentence)
        else:
            raise ValueError("Unsupported mode: {}".format(self.mode))

        if self.sent_id % self.check_span == 0:
            sentence = self.get_sentence()
            print(sentence)

        return instances

    def get_sentence(self):
        surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
        for mask_id, score in self.score_in_same_sentence:
            predict_tokens = self.get_random_n_tokens(score, mask_id)
            surfaces[mask_id] += " (" + " | ".join(predict_tokens) + ")"
        sentence = " ".join(surfaces)

        return sentence

    def get_random_n_tokens(self, score, mask_id):
        predict_tokens = []
        gold_token = self.sent_options[self.sent_id][1][mask_id]
        for rank in range(len(self.subword_index2vocab)):
            random_token = self.subword_index2vocab[random.choice(self.subword_ids)]
            if random_token != gold_token:
                predict_tokens.append(random_token)
            if len(predict_tokens) == self.n_sample:
                break

        return predict_tokens


class SamplingTokenFiller(MaskFiller):
    def __init__(self, config: FillerConfig, n_sample=5, mode="json"):
        super(SamplingTokenFiller, self).__init__(config)
        self.n_sample = n_sample
        self.mode = mode
        self.check_span = round(len(self.sent_options) / 15)

    def get_instances(self) -> List[str]:
        instances = []
        for pairs in self.get_n_sampling_tokens():
            if self.mode == "json":
                instance = self._get_instance_with_replace_tokens([pairs])
                instance = json.dumps(instance)
            elif self.mode == "surface":
                instance = self.get_sentence([pairs])
            else:
                raise ValueError("Unsupported mode: {}".format(self.mode))

            if self.sent_id % self.check_span == 0:
                sentence = self.get_sentence([pairs])
                print(sentence)

            instances.append(instance)

        return instances

    def get_sentence(self, pairs):
        surfaces = copy.deepcopy(self.sent_options[self.sent_id][1])
        for idx, token in pairs:
            surfaces[idx] += " ({})".format(token)
        sentence = " ".join(surfaces)

        return sentence

    def get_n_sampling_tokens(self):
        scores = torch.stack([score for _, score in self.score_in_same_sentence])
        scores = F.softmax(scores, 1)
        scores[:, self.filtered_ids] = 0
        sampling_matrix = torch.multinomial(scores, self.n_sample * 10).tolist()

        gold_tokens = [(mask_id, self.sent_options[self.sent_id][1][mask_id])
                       for mask_id in self.sent_options[self.sent_id][0]]
        results = []
        for (mask_id, gold_token), sampling_ids in zip(gold_tokens, sampling_matrix):
            pairs = []
            for idx in sampling_ids:
                predict_token = self.subword_index2vocab[idx]
                if idx not in self.filtered_ids and predict_token != gold_token:
                    pairs.append((mask_id, predict_token))
                    self.predict_token_distribution[predict_token] += 1
                if len(pairs) == self.n_sample:
                    break
            if len(pairs) != self.n_sample:
                raise RuntimeError("Insufficient number of prediction tokens")
            results += pairs

        return results
