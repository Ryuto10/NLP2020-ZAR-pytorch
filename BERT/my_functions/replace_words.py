# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import json
import logging
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pyknp import Juman
from torch.distributions import Categorical
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "_padding_"
BLACK_LIST = [UNK]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

juman = Juman()


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, instance, tokens, token_mask_ids, input_ids, input_mask):
        self.unique_id = unique_id
        self.instance = instance
        self.tokens = tokens
        self.token_mask_ids = token_mask_ids
        self.input_ids = input_ids
        self.input_mask = input_mask

    def send_to_device(self, device):
        self.input_ids = torch.LongTensor([self.input_ids]).to(device)
        self.input_mask = torch.LongTensor([self.input_mask]).to(device)


def prediction(model, feature, tokenizer, how_select):
    """how select"""
    all_predict_ids = []
    input_ids = feature.input_ids
    # Repeat updating 'input_ids'
    for token_mask_id in feature.token_mask_ids:
        predict = model(input_ids, token_type_ids=None, attention_mask=feature.input_mask)
        # delete black list tokens probability
        predict[:, token_mask_id, tokenizer.convert_tokens_to_ids(BLACK_LIST)] = -np.inf

        if how_select == "sample":
            dist = Categorical(logits=F.log_softmax(predict, dim=-1))
            pred_ids = dist.sample()
        elif how_select == "argmax":
            pred_ids = predict.argmax(dim=-1)
        else:
            raise ValueError("Selection mechanism %s not found!" % how_select)
        all_predict_ids.append(int(pred_ids[0][token_mask_id]))
        input_ids[0][token_mask_id] = pred_ids[0][token_mask_id]

    # Add 'bert_predicts' to instance
    predict_tokens = tokenizer.convert_ids_to_tokens(all_predict_ids)
    feature.instance["bert_predicts"] = copy.deepcopy(feature.instance["text_a"])
    for idx, token in zip(feature.instance["mask_ids"], predict_tokens):
        feature.instance["bert_predicts"][idx] = token

    return feature.instance


def convert_instances_to_features(instances, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for unique_id, instance in enumerate(instances):
        # Processing so that 'tokens_a' does not exceed 'seq_length'
        tokens_a = tokenizer.tokenize(" ".join(instance["text_a"]))
        if len(tokens_a) > seq_length - 2:
            mask_ids = [idx for idx, token in enumerate(tokens_a) if token == MASK]
            if len(mask_ids) > 1 and mask_ids[-1] - mask_ids[0] > seq_length - 3:
                raise ValueError("Masking range is over 128.")
            if mask_ids[-1] > seq_length - 3:
                end_idx = mask_ids[-1] + 1
                start_idx = end_idx - seq_length + 2
                tokens_a = tokens_a[start_idx:end_idx]
            else:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = [CLS] + tokens_a + [SEP]
        token_mask_ids = [idx for idx, token in enumerate(tokens) if token == MASK]
        input_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * (seq_length - len(tokens))
        input_mask = [1] * len(tokens) + [0] * (seq_length - len(tokens))

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length

        if unique_id < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % unique_id)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

        features.append(InputFeatures(unique_id=unique_id,
                                      instance=instance,
                                      tokens=tokens,
                                      token_mask_ids=token_mask_ids,
                                      input_ids=input_ids,
                                      input_mask=input_mask))

    return features


def create_masked_instances(args):
    """how mask"""
    # Load input instances
    with open(args.input_file) as fi:
        input_instances = [json.loads(line) for line in fi]

    print("# Create Masked Instances")
    print("## Number of replace: {} ~ {}".format(args.replace_min, args.replace_max))
    print("## Number of sample: {}".format(args.n_sample))
    instances = []
    data_size = int(len(input_instances) * args.data_ratio / 100)
    for in_instance in tqdm(input_instances[:data_size]):
        # Create instance
        positions = create_replace_positions(in_instance, args)
        for position in positions:
            instance = copy.deepcopy(in_instance)
            instance["text_a"] = copy.deepcopy(instance["surfaces"])
            instance["mask_ids"] = position
            instance["insert_ids"] = position
            for p in position:
                instance["text_a"][p] = MASK
                instances.append(instance)
    print("# Number of instances: {} -> {}".format(data_size, len(instances)))

    return instances


def create_replace_positions(instance, args) -> List[List[int]]:
    p_ids = [pas["p_id"] for pas in instance["pas"]]
    if args.predicate:
        candidate_positions = [i for i in range(len(instance["tokens"]))]
    else:
        candidate_positions = [i for i in range(len(instance["tokens"])) if i not in p_ids]

    positions = []
    for n in range(args.n_sample):
        n_words = min(random.randint(args.replace_min, args.replace_max), len(candidate_positions))
        positions.append(sorted(random.sample(candidate_positions, k=n_words)))

    return positions


def convert_bert_predicts_to_ids(instance, vocab):
    for idx in instance["mask_ids"]:
        token = instance["bert_predicts"][idx].strip("#")
        instance["bert_predicts"][idx] = token
        morphs = juman.analysis(token)
        morph = morphs[0]
        base = morph.genkei
        pos = morph.hinsi + "-" + morph.bunrui + "-" + morph.katuyou1
        if base in vocab:
            convert_id = vocab[base]
        elif pos in vocab:
            convert_id = vocab[pos]
        else:
            convert_id = vocab[PAD]

        instance["tokens"][idx] = convert_id

    return instance


def set_vocab(vocab_file):
    """Open file of pre-trained vocab and convert to dict format."""
    print("\n# Load '{}'".format(vocab_file))
    vocab = {}
    f = open(vocab_file)
    for line in f:
        split_line = line.rstrip().split("\t")
        word, idx = split_line[0], split_line[1]
        vocab[word] = idx
    f.close()

    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model",  default="/home/ryuto/data/jap_BERT/", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--vocab", default="/home/ryuto/data/NTC_Matsu_original/wordIndex.txt", type=str)

    # model parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model. (If Japanese model, set false)")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    # Data Augmentation Option
    parser.add_argument('--data_ratio', type=float, default=100,
                        help="full size = 100 (default=100)")
    parser.add_argument("--token_strategy", dest='how_select', default="argmax", type=str,
                        help="Choose from 'argmax' or 'sample'")
    parser.add_argument('--predicate', action='store_true',
                        help="If True, target word is replaced even if it is predicate.")

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--replace_max', type=int, default=5)
    parser.add_argument('--replace_min', type=int, default=3)
    parser.add_argument('--n_sample', type=int, default=3)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # vocab & tokenizer
    vocab = set_vocab(args.vocab)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Create MASK instances
    instances = create_masked_instances(args)

    # Create dataset
    features = convert_instances_to_features(instances=instances,
                                             seq_length=args.max_seq_length,
                                             tokenizer=tokenizer)
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()

    with open(args.output_file, "w", encoding='utf-8') as writer:
        for feature in tqdm(features):
            feature.send_to_device(device)
            instance = prediction(model=model,  feature=feature, tokenizer=tokenizer, how_select=args.how_select)
            instance = convert_bert_predicts_to_ids(instance=instance, vocab=vocab)
            print(json.dumps(instance), file=writer)


if __name__ == "__main__":
    main()
