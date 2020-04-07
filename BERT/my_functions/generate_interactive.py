# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import random
from itertools import groupby

import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from pyknp import Juman
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.distributions import Categorical

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "_padding_"
BLACK_LIST = [UNK]

juman = Juman()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, text: str, tokenizer: BertTokenizer, max_seq_length: int, language: str):
        self.max_seq_length = max_seq_length if max_seq_length else 0
        self.lang = language
        if self.lang == "ja":
            tokens = []
            for v, group in groupby(text, key=lambda x: x == "M"):
                if v:
                    tokens += [MASK for _ in group]
                else:
                    tokens += [morph.midasi for morph in juman.analysis("".join(group))]
        elif self.lang == "en":
            tokens = [MASK if token == "M" else token for token in text.split(" ")]
        else:
            raise ValueError("Unsupported value: {}".format(self.lang))

        self.original_tokens = tokens
        self.original_token_mask_ids = [idx for idx, token in enumerate(self.original_tokens) if token == MASK]
        self.tokens = [CLS] + tokenizer.tokenize(" ".join(tokens)) + [SEP]
        self.token_mask_ids = [idx for idx, token in enumerate(self.tokens) if token == MASK]
        self.len = len(self.tokens)

        if self.max_seq_length and self.len > self.max_seq_length:
            raise RuntimeError("'tokens_a' is over {}: {}".format(max_seq_length, self.len))

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens) + [0] * max(self.max_seq_length - self.len, 0)
        self.attention_mask = [1] * self.len + [0] * max(self.max_seq_length - self.len, 0)

    def send_to_device(self, device):
        self.input_ids = torch.LongTensor([self.input_ids]).to(device)
        self.attention_mask = torch.LongTensor([self.attention_mask]).to(device)


def prediction_with_beam_search(device, model, feature, tokenizer, black_list, k=5):
    """how select"""
    feature.send_to_device(device)
    input_ids = feature.input_ids
    black_list_ids = tokenizer.convert_tokens_to_ids([t for t in black_list if t in tokenizer.vocab] + BLACK_LIST)

    topk_dict = {0: (1, [])}
    for n in range(len(feature.token_mask_ids)):
        topk_buffer = []
        for prob, predicted_ids in topk_dict.values():
            if n == 0:
                predict = model(input_ids, token_type_ids=None, attention_mask=feature.attention_mask)
            else:
                filled_ids = copy.deepcopy(input_ids)
                filled_ids[0][feature.token_mask_ids[:n]] = torch.LongTensor(predicted_ids).to(device)
                predict = model(filled_ids, token_type_ids=None, attention_mask=feature.attention_mask)
            predict[:, feature.token_mask_ids[n], black_list_ids] = -np.inf
            topk_prob, topk_indices = predict.topk(k)
            topk_buffer += [(prob * float(p), predicted_ids + [int(idx)])
                            for p, idx in zip(topk_prob[0][feature.token_mask_ids[n]],
                                              topk_indices[0][feature.token_mask_ids[n]])]
        topk_dict = {i: p_ids for i, p_ids in enumerate(sorted(topk_buffer, key=lambda x: -x[0])[:k])}

    output_sents = []
    output_tokens = []
    for i in range(k):
        output_ids = input_ids[0].tolist()
        for idx, token in zip(feature.token_mask_ids, topk_dict[i][1]):
            output_ids[idx] = token
        output_sents.append(tokenizer.convert_ids_to_tokens(output_ids))
        output_tokens.append(tokenizer.convert_ids_to_tokens(topk_dict[i][1]))

    return output_sents, output_tokens


def prediction_single(device, model, feature, tokenizer, how_select, black_list):
    """how select"""
    all_predict_ids = []
    feature.send_to_device(device)
    input_ids = feature.input_ids
    # Repeat updating 'input_ids'
    for n, token_mask_id in enumerate(feature.token_mask_ids):
        predict = model(input_ids, token_type_ids=None, attention_mask=feature.attention_mask)
        predict[:, token_mask_id, tokenizer.convert_tokens_to_ids(
            BLACK_LIST + [t for t in black_list if t in tokenizer.vocab]
        )] = -np.inf

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

    return predict_tokens


def prediction_multi(device, model, feature, tokenizer, how_select, black_list):
    """how select"""
    all_predict_ids = []
    feature.send_to_device(device)
    input_ids = feature.input_ids
    # Repeat updating 'input_ids'
    for n, token_mask_id in enumerate(feature.token_mask_ids):
        predict = model(input_ids, token_type_ids=None, attention_mask=feature.attention_mask)
        predict[:, token_mask_id, tokenizer.convert_tokens_to_ids(
            BLACK_LIST + [t for t in black_list if t in tokenizer.vocab]
        )] = -np.inf

        if how_select == "sample":
            dist = Categorical(logits=F.log_softmax(predict, dim=-1))
            pred_ids = dist.sample()
        elif how_select == "argmax":
            pred_ids = predict.argmax(dim=-1)
        else:
            raise ValueError("Selection mechanism %s not found!" % how_select)
        all_predict_ids.append(int(pred_ids[0][token_mask_id]))

    # Add 'bert_predicts' to instance
    predict_tokens = tokenizer.convert_ids_to_tokens(all_predict_ids)

    return predict_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="/home/ryuto/data/jawiki-kurohashi-bert", type=str,
                        help="Please fill the path to directory of BERT model, or the name of BERT model."
                             "Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    # BERT model parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model. (If Japanese model, set false)")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--language", type=str, default="ja", help="Choose from 'ja' or 'en' (default='ja').")

    # Data Augmentation Option
    parser.add_argument("--how_select", dest='how_select', default="argmax", type=str,
                        help="Choose from 'argmax' or 'sample' or 'beam'. (default='argmax')")
    parser.add_argument("--how_many", default='multi', type=str,
                        help="Choose from 'single' or 'multi'. (default='multi')")
    parser.add_argument('--topk', type=int, default=5, help="for beam search")

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()
    logger.info(args)

    # Seed
    random.seed(args.seed)
    logger.info("Seed: {}".format(args.seed))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    logger.info("language: {}".format(args.language))
    logger.info("BERT model: {}".format(args.bert_model))
    logger.debug("Loading BERT model...")
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    logger.debug("Sending BERT model to device...")
    model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Input the sentence
    logger.info("How select tokens: {}".format(args.how_select))
    logger.info("How many tokens to predict at once: {}".format(args.how_many))
    print("Mask token is 'M'.")
    while True:
        text = input("Sentence: ")
        if text == "q":
            break
        black_list = input("Black list of tokens (separator is ','): ").replace(" ", "").replace("　", "").split(",")

        # Input feature
        feature = InputFeatures(text=text,
                                tokenizer=tokenizer,
                                max_seq_length=args.max_seq_length,
                                language=args.language)
        logger.debug(feature.tokens)
        if len(feature.token_mask_ids) == 0:
            print("Not found mask token (mask token is 'M').")
            continue

        if args.how_select == "beam":
            output_sents, output_tokens = prediction_with_beam_search(device=device,
                                                                      model=model,
                                                                      feature=feature,
                                                                      tokenizer=tokenizer,
                                                                      black_list=black_list,
                                                                      k=args.topk)
            for sent in output_sents:
                print(" ".join(sent[1:feature.len - 1]))

        else:
            if args.how_many == "single":
                predict_tokens = prediction_single(device=device,
                                                   model=model,
                                                   feature=feature,
                                                   tokenizer=tokenizer,
                                                   how_select=args.how_select,
                                                   black_list=black_list)
            elif args.how_many == "multi":
                predict_tokens = prediction_multi(device=device,
                                                  model=model,
                                                  feature=feature,
                                                  tokenizer=tokenizer,
                                                  how_select=args.how_select,
                                                  black_list=black_list)
            else:
                raise ValueError("Unsupported value: {}".format(args.how_many))

            assert len(predict_tokens) == len(feature.token_mask_ids)
            # tokens = feature.tokens
            # for idx, p_token in zip(feature.token_mask_ids, predict_tokens):
            #     tokens[idx] = p_token
            # print(" ".join(tokens[1:feature.len - 1]))

            filled_tokens = copy.deepcopy(feature.original_tokens)
            for idx, p_token in zip(feature.original_token_mask_ids, predict_tokens):
                filled_tokens[idx] = p_token
            print(" ".join(filled_tokens))


if __name__ == "__main__":
    main()
