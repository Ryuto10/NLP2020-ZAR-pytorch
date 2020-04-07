# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
from typing import List
from os import path

import numpy as np
import torch
from tqdm import tqdm
import h5py

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"
PAD = 0

SELECT_VECTOR_MODES = ["start", "end", "sum", "ave"]


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=path.abspath, required=True)
    parser.add_argument("--output_file", type=path.abspath, required=True)

    # BERT option
    parser.add_argument("--bert_model", default="/home/ryuto/data/jap_BERT", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layer", default=-1, type=int, help="If -1, use last layer.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    # Other option
    parser.add_argument('--ntc', action='store_true')
    parser.add_argument('--how_select_vec', type=str, default="start",
                        help="Choose from {}".format(", ".join(SELECT_VECTOR_MODES)))

    return parser


class SubWordAlignment(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        super(SubWordAlignment, self).__init__(vocab_file, do_lower_case, max_len, never_split)

    def alignment(self, tokens: List[str], subwords: List[str]):
        tokenized_tokens = [[tokenized_tokens for tokenized_tokens in self.basic_tokenizer.tokenize(token)]
                            for token in tokens]
        assert len(tokenized_tokens) == len(tokens)

        idx = 0
        indices = []
        for chunk in tokenized_tokens:
            buffer = []
            for token in chunk:
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    buffer.append(idx)
                    assert sub_token == subwords[idx]
                    idx += 1
            indices.append(buffer)

        assert len(subwords) == idx
        assert len(tokens) == len(indices)

        return indices


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, tokenizer, alignment, max_seq_length):
        self.tokens = tokens
        self.max_seq_length = max_seq_length
        self.tokenized_tokens = tokenizer.tokenize((" ".join(tokens)))
        self.merged_indices = alignment.alignment(tokens=tokens, subwords=self.tokenized_tokens)
        self.len = len(self.tokenized_tokens)
        self.inputs = []
        for n in range(int(self.len / (self.max_seq_length - 2)) + 1):
            start = n * (self.max_seq_length - 2)
            end = (n + 1) * (self.max_seq_length - 2)
            in_tokens = [CLS] + self.tokenized_tokens[start:end] + [SEP]
            ids = tokenizer.convert_tokens_to_ids(in_tokens) + [0] * (self.max_seq_length - len(in_tokens))
            att_mask = [1] * len(in_tokens) + [0] * (self.max_seq_length - len(in_tokens))
            self.inputs.append((ids, att_mask))


def send_to_device(device, ids):
    ids = torch.LongTensor([ids]).to(device)
    return ids


def prediction(device, model, feature, layer, how_select_vec) -> np.array:
    vectors = []
    for input_ids, attention_mask in feature.inputs:
        input_ids = send_to_device(device, input_ids)
        attention_mask = send_to_device(device, attention_mask)
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        # 1が立っているところを抽出
        end_idx = int(sum(attention_mask[0])) - 1
        vectors.append([layer.detach().cpu().numpy()[0][1:end_idx] for layer in all_encoder_layers])
    # 連結
    layer_vectors = np.concatenate([chunk[layer] for chunk in vectors])
    assert len(layer_vectors) == feature.len

    if how_select_vec == "start":
        indices = [ids[0] for ids in feature.merged_indices]
        output = layer_vectors[indices]
    elif how_select_vec == "end":
        indices = [ids[-1] for ids in feature.merged_indices]
        output = layer_vectors[indices]
    elif how_select_vec == "sum":
        output = np.array([np.sum(layer_vectors[ids], axis=0) for ids in feature.merged_indices])
    elif how_select_vec == "ave":
        output = np.array([np.mean(layer_vectors[ids], axis=0) for ids in feature.merged_indices])
    else:
        raise ValueError("Unsupported value: '{}'".format(how_select_vec))
    assert len(output) == len(feature.tokens)

    return output


def read_file(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    alignment = SubWordAlignment.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()

    if args.ntc:
        fo = h5py.File(args.output_file, 'w')
    else:
        fo = open(args.output_file, "w")
    with open(args.input_file) as fi:
        for line in tqdm(fi):
            if args.ntc:
                instance = json.loads(line)
                unique_id = instance["unique_id"]
                # tokens = [UNK if idx == PAD else surf for surf, idx in zip(instance["surfaces"], instance["tokens"])
                tokens = instance["surfaces"]
            else:
                tokens = line.rstrip("\n").split()
            feature = InputFeatures(tokens=tokens,
                                    tokenizer=tokenizer,
                                    alignment=alignment,
                                    max_seq_length=args.max_seq_length)
            layer_token_vectors = prediction(device=device,
                                             model=model,
                                             feature=feature,
                                             layer=args.layer,
                                             how_select_vec=args.how_select_vec)
            if args.ntc:
                fo.create_dataset(str(unique_id), data=layer_token_vectors)
            else:
                print(json.dumps(instance), file=fo)
    fo.close()


if __name__ == "__main__":
    main()
