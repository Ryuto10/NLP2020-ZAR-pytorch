# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
from typing import List
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
BLACK_LIST = [UNK]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, text, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length if max_seq_length else 0
        tokens = [CLS] + tokenizer.tokenize(text) + [SEP]
        self.len = len(tokens)

        if self.max_seq_length and self.len > self.max_seq_length:
            raise RuntimeError("'tokens_a' is over {}: {}".format(max_seq_length, self.len))

        self.tokens = tokens
        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens) + [0] * max(self.max_seq_length - self.len, 0)
        self.attention_mask = [1] * self.len + [0] * max(self.max_seq_length - self.len, 0)

    def send_to_device(self, device):
        self.input_ids = torch.LongTensor([self.input_ids]).to(device)
        self.attention_mask = torch.LongTensor([self.attention_mask]).to(device)


# ----- General Functions -----

def prediction_with_beam_search(device, model, feature, tokenizer, black_list, k=5):
    """how select"""
    feature.send_to_device(device)
    input_ids = feature.input_ids
    token_mask_ids = [idx for idx, token in enumerate(feature.tokens) if token == MASK]
    black_list_ids = tokenizer.convert_tokens_to_ids([t for t in black_list if t in tokenizer.vocab] + BLACK_LIST)

    topk_dict = {0: (1, [])}
    for n in range(len(token_mask_ids)):
        topk_buffer = []
        for prob, predicted_ids in topk_dict.values():
            if n == 0:
                predict = model(input_ids, token_type_ids=None, attention_mask=feature.attention_mask)
            else:
                filled_ids = copy.deepcopy(input_ids)
                filled_ids[0][token_mask_ids[:n]] = torch.LongTensor(predicted_ids).to(device)
                predict = model(filled_ids, token_type_ids=None, attention_mask=feature.attention_mask)
            predict[:, token_mask_ids[n], black_list_ids] = -np.inf
            topk_prob, topk_indices = predict.topk(k)
            topk_buffer += [(prob * float(p),  predicted_ids + [int(idx)])
                            for p, idx in zip(topk_prob[0][token_mask_ids[n]], topk_indices[0][token_mask_ids[n]])]
        topk_dict = {i: p_ids for i, p_ids in enumerate(sorted(topk_buffer, key=lambda x: -x[0])[:k])}

    output_sents = []
    output_tokens = []
    for i in range(k):
        output_ids = input_ids[0].tolist()
        for idx, token in zip(token_mask_ids, topk_dict[i][1]):
            output_ids[idx] = token
        output_sents.append(tokenizer.convert_ids_to_tokens(output_ids))
        output_tokens.append(tokenizer.convert_ids_to_tokens(topk_dict[i][1]))

    return output_sents, output_tokens


def prediction(model, feature, tokenizer, how_select, black_list) -> (List[str], List[str]):
    """how select"""
    all_predict_ids = []
    import ipdb; ipdb.set_trace()

    # text_a
    input_ids = feature.input_ids
    token_mask_ids = [idx for idx, token in enumerate(feature.tokens) if token == MASK]
    black_list_ids = tokenizer.convert_tokens_to_ids([t for t in black_list if t in tokenizer.vocab] + BLACK_LIST)
    for token_mask_id in token_mask_ids:
        predict = model(input_ids, token_type_ids=None, attention_mask=feature.attention_mask)
        predict[:, token_mask_id, black_list_ids] = -np.inf
        if how_select == "sample":
            dist = Categorical(logits=F.log_softmax(predict, dim=-1))
            pred_ids = dist.sample()
        elif how_select == "argmax":
            pred_ids = predict.argmax(dim=-1)
        else:
            raise ValueError("Selection mechanism %s not found!" % how_select)
        all_predict_ids.append(int(pred_ids[0][token_mask_id]))
        input_ids[0][token_mask_id] = pred_ids[0][token_mask_id]

    predict_sentence = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    predict_tokens = tokenizer.convert_ids_to_tokens(all_predict_ids)

    return predict_sentence, predict_tokens


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", default=None, type=str, required=True)
    # parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_version", default="bert-base-cased",
                        choices=["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-uncased"])

    # model parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model. (If Japanese model, set false)")
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    # Data Augmentation Option
    parser.add_argument('--data_ratio', type=float, default=100,
                        help="full size = 100 (default=100)")
    parser.add_argument("--token_strategy", dest='how_select', default="argmax", type=str,
                        help="Choose from 'argmax' or 'sample'")
    parser.add_argument('--topk', type=int, default=5)

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # vocab & tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_version, do_lower_case=args.do_lower_case)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.bert_version)
    model.to(device)
    model.eval()

    while True:
        black_list = input("BLACK LIST > ").split(" ")
        text = input("> ")
        if text == "q":
            break
        feature = InputFeatures(text=text, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
        output_sents, output_tokens = prediction_with_beam_search(device=device,
                                                                  model=model,
                                                                  feature=feature,
                                                                  tokenizer=tokenizer,
                                                                  black_list=black_list,
                                                                  k=args.topk)
        for sent in output_sents:
            print(" ".join(sent[1:-1]))


if __name__ == "__main__":
    main()
