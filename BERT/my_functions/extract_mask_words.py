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
import copy
import json
from os import path, remove

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from logzero import logger
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm
import h5py

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"
MASK = "[MASK]"
PAD = 0
NTC_UNK = "_padding_"

BLACK_LIST = [UNK]


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=path.abspath, required=True,
                        help="masked jsonl")
    parser.add_argument("--output_file", type=path.abspath, required=True,
                        help="hdf5 or jsonl")
    parser.add_argument("--what_replace", type=str,  required=True,
                        help="Choose from 'word' or 'vec'",)

    # Option
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--how_predict", default="single",
                        help="Choose from 'single', 'multi' (default is single).")
    parser.add_argument('--data_limit', type=int, default=None,
                        help="Maximum line number of the file to be extracted.")

    # Vector
    parser.add_argument("--train_matrix", type=path.abspath,
                        help="Path to train.hdf5. Add option if 'what_replace' is 'vec'.")
    parser.add_argument('--map', type=path.abspath,
                        help="Path to file of unique_id conversion from pseudo to train. "
                             "Add option if 'what_replace' is 'vec'.")

    # Word
    parser.add_argument("--how_select", default="argmax",
                        help="Choose from 'argmax', 'sample'  (default is 'argmax'). "
                             "Add option if 'what_replace' is 'word'.")
    parser.add_argument("--juman", action="store_true",
                        help="Add option if 'what_replace' is 'word'.")
    parser.add_argument("--vocab", default="/home/ryuto/data/NTC_Matsu_original/wordIndex.txt", type=str,
                        help="Path to pre-trained vocab file."
                             "Add option if 'what_replace' is 'word' and 'juman' is set.")

    # BERT option
    parser.add_argument("--bert_model", default="/home/ryuto/data/jap_BERT", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    return parser


def create_tmp_file():
    fn = ".tmp"
    fn_end = ".jsonl"
    fn_n = ""
    n = 1
    while path.exists(fn + fn_n + fn_end):
        n += 1
        fn_n = str(n)

    return fn + fn_n + fn_end


def read_instance(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def length_file(file):
    with open(file) as fi:
        for idx, _ in enumerate(fi):
            pass
    return idx


def batch_generator_with_single(file_path: str, tokenizer: BertTokenizer, max_seq_length: int,
                                batch_size: int, device, data_limit: int):
    batch_inputs = []
    batch_att_mask = []
    batch_target_ids = []
    batch_position = []

    for n, instance in enumerate(read_instance(file_path)):
        if data_limit == n:
            logger.info("The maximum number of rows has been reached: {}".format(data_limit))
            batch_inputs = []
            break

        if n % 1000 == 0:
            logger.info("Unique ID: {}".format(instance["unique_id"]))

        tokens_with_mask = instance["surfaces"]
        mask_ids = [idx for idx, token in enumerate(tokens_with_mask) if token == MASK]
        for mask_idx in mask_ids:
            original_tokens = copy.deepcopy(instance["original_surfaces"])
            original_tokens[mask_idx] = MASK
            tokenized_tokens = tokenizer.tokenize(" ".join(original_tokens))

            if n < 3:
                logger.debug(original_tokens)
                logger.debug(tokenized_tokens)
                logger.debug(mask_idx)

            if tokenized_tokens.index(MASK) < max_seq_length - 2:
                in_tokens = [CLS] + tokenized_tokens[0:max_seq_length - 2] + [SEP]
            elif len(tokenized_tokens) < (max_seq_length - 2) * 2:
                logger.debug("exceed {}".format(max_seq_length))
                in_tokens = [CLS] + tokenized_tokens[len(tokenized_tokens) - max_seq_length + 2:] + [SEP]
                assert len(in_tokens) == max_seq_length
            else:
                raise RuntimeError("Sentence is too long.")

            assert MASK in in_tokens
            input_ids = tokenizer.convert_tokens_to_ids(in_tokens) + [0] * (max_seq_length - len(in_tokens))
            att_mask = [1] * len(in_tokens) + [0] * (max_seq_length - len(in_tokens))

            batch_inputs.append(input_ids)
            batch_att_mask.append(att_mask)
            batch_target_ids.append(in_tokens.index(MASK))
            batch_position.append((instance["unique_id"], instance["sentence id"], instance["file name"], mask_idx))
            if len(batch_inputs) == batch_size:
                assert len(batch_inputs) == len(batch_att_mask) == len(batch_target_ids) == len(batch_position)
                batch_inputs = torch.LongTensor(batch_inputs).to(device)
                batch_att_mask = torch.LongTensor(batch_att_mask).to(device)
                batch_target_ids = [[i for i in range(batch_size)], batch_target_ids]

                yield batch_inputs, batch_att_mask, batch_target_ids, batch_position

                batch_inputs = []
                batch_att_mask = []
                batch_target_ids = []
                batch_position = []
    if batch_inputs:
        batch_target_ids = [[i for i in range(len(batch_inputs))], batch_target_ids]
        batch_inputs = torch.LongTensor(batch_inputs).to(device)
        batch_att_mask = torch.LongTensor(batch_att_mask).to(device)

        yield batch_inputs, batch_att_mask, batch_target_ids, batch_position


def batch_generator_with_multi(file_path: str, tokenizer: BertTokenizer, max_seq_length: int,
                               batch_size: int, device, data_limit: int):
    batch_inputs = []
    batch_att_mask = []
    batch_target_ids = []
    batch_position = []

    for n, instance in enumerate(read_instance(file_path)):
        if data_limit == n:
            logger.info("The maximum number of rows has been reached: {}".format(data_limit))
            batch_inputs = []
            break

        tokens_with_mask = instance["surfaces"]
        mask_ids = [idx for idx, token in enumerate(tokens_with_mask) if token == MASK]

        tokenized_tokens = tokenizer.tokenize(" ".join(tokens_with_mask))

        if n < 3:
            logger.debug(tokens_with_mask)
            logger.debug(tokenized_tokens)
            logger.debug(mask_ids)

        subword_mask_ids = [idx for idx, subword in enumerate(tokenized_tokens) if subword == MASK]
        within_mask_ids = [idx for idx in subword_mask_ids if idx < max_seq_length - 2]
        out_mask_ids = [idx for idx in subword_mask_ids if idx >= max_seq_length - 2]

        buffer = []
        if within_mask_ids:
            in_tokens = [CLS] + tokenized_tokens[0:max_seq_length - 2] + [SEP]
            target_ids = [idx for idx, token in enumerate(in_tokens) if token == MASK]
            buffer.append((in_tokens, target_ids))
        if out_mask_ids:
            logger.debug("exceed {}".format(max_seq_length))
            in_tokens = [CLS] + tokenized_tokens[len(tokenized_tokens) - max_seq_length + 2:] + [SEP]
            target_ids = [idx for idx, token in enumerate(in_tokens) if token == MASK][-len(out_mask_ids):]
            assert len(in_tokens) == max_seq_length
            logger.debug(in_tokens)
            logger.debug(target_ids)
            buffer.append((in_tokens, target_ids))
            if out_mask_ids[-1] >= (max_seq_length - 2) * 2:
                raise RuntimeError("Sentence is too long.")

        len_batch = len(batch_inputs)
        for in_tokens, target_ids in buffer:
            input_ids = tokenizer.convert_tokens_to_ids(in_tokens) + [0] * (max_seq_length - len(in_tokens))
            att_mask = [1] * len(in_tokens) + [0] * (max_seq_length - len(in_tokens))
            batch_inputs.append(input_ids)
            batch_att_mask.append(att_mask)
            batch_target_ids += [(len_batch, i) for i in target_ids]
        batch_position += [(instance["unique_id"], instance["sentence id"], instance["file name"], mask_idx)
                           for mask_idx in mask_ids]

        if len(batch_inputs) >= batch_size:
            assert len(batch_inputs) == len(batch_att_mask)
            batch_inputs = torch.LongTensor(batch_inputs).to(device)
            batch_att_mask = torch.LongTensor(batch_att_mask).to(device)
            batch_target_ids = [[i[0] for i in batch_target_ids], [i[1] for i in batch_target_ids]]

            yield batch_inputs, batch_att_mask, batch_target_ids, batch_position

            batch_inputs = []
            batch_att_mask = []
            batch_target_ids = []
            batch_position = []

    if batch_inputs:
        assert len(batch_inputs) == len(batch_att_mask)
        batch_inputs = torch.LongTensor(batch_inputs).to(device)
        batch_att_mask = torch.LongTensor(batch_att_mask).to(device)
        batch_target_ids = [[i[0] for i in batch_target_ids], [i[1] for i in batch_target_ids]]

        yield batch_inputs, batch_att_mask, batch_target_ids, batch_position


def predict_to_words(batch_predict, batch_position, juman, vocab, tokenizer, how_select):
    if how_select == "sample":
        dist = Categorical(logits=F.log_softmax(batch_predict, dim=-1))
        pred_ids = dist.sample()
    elif how_select == "argmax":
        pred_ids = batch_predict.argmax(dim=-1)
    else:
        raise ValueError("Selection mechanism %s not found!" % how_select)

    # 45はなぜかKeyErrorが出るというクソ仕様 & 0~4は特殊トークンなので,一括して'[UNK]'(index=1)にしたあと'_padding_'にする
    pred_ids = [1 if i == 45 or i <= 4 else i for i in pred_ids.tolist()]
    predicted_tokens = [token for token in tokenizer.convert_ids_to_tokens(pred_ids)]
    convert_ids = []
    if juman is not None:
        # Convert tokens to ids
        for token in predicted_tokens:
            token = token.strip("#")
            if token == UNK:
                convert_ids.append(vocab[NTC_UNK])
                continue
            morphs = [m for m in juman.analysis(token)]
            if len(morphs) == 0:
                base = NTC_UNK
            else:
                base = "".join(m.genkei for m in morphs)
                pos = morphs[-1].hinsi + "-" + morphs[-1].bunrui + "-" + morphs[-1].katuyou1
            if base in vocab:
                convert_ids.append(vocab[base])
            elif pos in vocab:
                convert_ids.append(vocab[pos])
            else:
                convert_ids.append(vocab[NTC_UNK])
    else:
        convert_ids = [0] * len(predicted_tokens)

    for token_idx, surf, position in zip(convert_ids, predicted_tokens, batch_position):
        yield {"unique_id": position[0],
               "sentence id": position[1],
               "file name": position[2],
               "mask idx": position[3],
               "predict": (token_idx, surf)}


def predict_to_vectors(batch_predict, batch_position):
    for vector, position in zip(batch_predict, batch_position):
        assert vector.shape[0] == 768
        yield {"unique_id": position[0],
               "sentence id": position[1],
               "file name": position[2],
               "mask idx": position[3],
               "predict": vector.tolist()}


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
    parser = create_parser()
    args = parser.parse_args()

    if path.exists(args.output_file):
        raise FileExistsError("'{}' already exists.".format(args.output_file))

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.what_replace == "word":
        with open(args.input_file) as fi:
            instance = json.loads(next(fi))
            keys = instance.keys()
            logger.info(keys)
            assert "original_surfaces" in keys

        if args.juman:
            from pyknp import Juman
            juman = Juman()
            vocab = set_vocab(args.vocab)
        else:
            juman = None
            vocab = None
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    elif args.what_replace == "vec":
        if not args.train_matrix:
            raise ValueError("Please enter 'train_matrix'")
        if not args.map:
            raise ValueError("Please enter 'map'")
        model = BertModel.from_pretrained(args.bert_model)
    else:
        raise ValueError("Unsupported value: '{}'".format(args.what_replace))
    model.to(device)
    model.eval()

    logger.info(args)

    # batch generator
    if args.how_predict == "single":
        batch_generator = batch_generator_with_single(args.input_file, tokenizer, args.max_seq_length,
                                                      args.batch_size, device, args.data_limit)
    elif args.how_predict == "multi":
        batch_generator = batch_generator_with_multi(args.input_file, tokenizer, args.max_seq_length,
                                                     args.batch_size, device, args.data_limit)
    else:
        raise ValueError("Unsupported value: '{}'".format(args.how_predict))

    # tmp_file = create_tmp_file()
    tmp_file = ".tmp." + path.basename(args.input_file) + "." + path.basename(args.output_file)
    logger.info("Tmp file: '{}'".format(tmp_file))
    logger.info("Start BERT prediction")

    len_file = length_file(args.input_file)
    total = (1 + len_file // args.batch_size) * (1 if args.how_predict == "multi" else 20)
    logger.info("wc -l '{}' = {}".format(args.input_file, len_file))

    # predict
    fo = open(tmp_file, "w")
    for batch_inputs, batch_att_mask, batch_target_ids, batch_position in tqdm(batch_generator, total=total):
        predict = model(batch_inputs, token_type_ids=None, attention_mask=batch_att_mask)
        if args.what_replace == "vec":
            predict = predict[0][-1]  # encoded_layers の最終層を使う
        target_predict = predict[batch_target_ids].cpu()

        assert len(target_predict) == len(batch_position)

        if args.what_replace == "word":
            converter = predict_to_words(target_predict, batch_position, juman, vocab, tokenizer, args.how_select)
        elif args.what_replace == "vec":
            converter = predict_to_vectors(target_predict, batch_position)
        else:
            raise ValueError("Unsupported value: '{}'".format(args.what_replace))

        for line in converter:
            print(json.dumps(line), file=fo)
    fo.close()

    logger.info("Start replace")
    tmp_f = open(tmp_file)

    # replace
    if args.what_replace == "vec":
        train_vec_file = h5py.File(args.train_matrix, "r")
        fo = h5py.File(args.output_file, 'w')
        with open(args.map) as fi:
            mapping_pseudo_to_train: dict = json.load(fi)

        for instance in tqdm(read_instance(args.input_file)):
            unique_id = str(instance["unique_id"])
            mask_ids = [idx for idx, token in enumerate(instance["surfaces"]) if token == MASK]
            train_vec = train_vec_file.get(mapping_pseudo_to_train[str(unique_id)])[()]
            assert len(train_vec) == len(instance["tokens"])

            for mask_idx in mask_ids:
                tmp_line = json.loads(next(tmp_f))
                assert unique_id == str(tmp_line["unique_id"])
                assert instance["sentence id"] == tmp_line["sentence id"]
                assert instance["file name"] == tmp_line["file name"]
                assert mask_idx == tmp_line["mask idx"]
                train_vec[mask_idx] = np.array(tmp_line["predict"])

            fo.create_dataset(unique_id, data=train_vec)

        train_vec_file.close()
        fo.close()

    elif args.what_replace == "word":
        fo = open(args.output_file, 'w')
        for n, instance in tqdm(enumerate(read_instance(args.input_file))):
            unique_id = str(instance["unique_id"])
            mask_ids = [idx for idx, token in enumerate(instance["surfaces"]) if token == MASK]
            new_instance = copy.deepcopy(instance)
            tokens = new_instance["tokens"]
            surfaces = new_instance["surfaces"]

            for mask_idx in mask_ids:
                tmp_line = json.loads(next(tmp_f))
                assert unique_id == str(tmp_line["unique_id"])
                assert instance["sentence id"] == tmp_line["sentence id"]
                assert instance["file name"] == tmp_line["file name"]
                assert mask_idx == tmp_line["mask idx"]
                token_idx, surf = tmp_line["predict"]
                tokens[mask_idx] = token_idx
                surfaces[mask_idx] = surf

            new_instance["tokens"] = tokens
            new_instance["surfaces"] = surfaces
            new_instance["mask_ids"] = mask_ids
            assert len([idx for idx, token in enumerate(new_instance["surfaces"]) if token == MASK]) == 0

            print(json.dumps(new_instance), file=fo)
            if n < 10:
                logger.debug("".join(instance["surfaces"]))
                logger.debug("".join(new_instance["original_surfaces"]))
                logger.debug("".join(new_instance["surfaces"]))
        fo.close()
    else:
        raise ValueError("Unsupported value: '{}'".format(args.what_replace))

    tmp_f.close()
    logger.info("delete: {}".format(tmp_file))
    remove(tmp_file)
    logger.info("done")


if __name__ == "__main__":
    main()
