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
import logging
import re
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from my_functions.fill_mask import FillerConfig, BestNTokenFiller, MultiTokenFiller, BestNTokenPredicateFiller, RandomNTokenFiller, SamplingTokenFiller
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                # tokens_a = tokens_a[0:(seq_length - 2)]

                # add
                mask_ids = []
                for idx, token in enumerate(tokens_a):
                    if token == "[MASK]":
                        mask_ids.append(idx)
                if len(mask_ids) > 1 and mask_ids[-1] - mask_ids[0] > seq_length - 3:
                    raise ValueError("Masking range is over 128.")
                if mask_ids[-1] > seq_length - 3:
                    end_idx = mask_ids[-1] + 1
                    start_idx = end_idx - seq_length + 2
                    tokens_a = tokens_a[start_idx:end_idx]
                else:
                    tokens_a = tokens_a[0:(seq_length - 2)]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            # line = line.strip()
            line = line.rstrip("\n")
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


# add
def read_examples_and_mask(input_file, case_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0

    case_file_object = open(case_file)
    sentence_index = 0
    sent_options = []
    instance_options = []

    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            case_types = case_file_object.readline()
            if not line:
                break

            line = line.rstrip("\n")
            case_types = case_types.rstrip("\n").split(" ")
            len_sent = len(line.split(" "))
            assert len(case_types) == len_sent

            mask_ids = [idx for idx, case_type in enumerate(case_types) if case_type != "null"]
            sent_options.append((mask_ids, line.split(" "), len_sent))

            for mask_id in mask_ids:
                surfaces = line.split(" ")
                surfaces[mask_id] = "[MASK]"
                text_a = " ".join(surfaces)
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=None))
                unique_id += 1
                instance_options.append((sentence_index, mask_id))

            sentence_index += 1

    print("# Size of original sentence: {}\n# Size of masking sentence: {}".format(sentence_index, unique_id))

    return examples, sent_options, instance_options


# add
def read_examples_and_mask_pred(input_file, json_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0

    json_file_object = open(json_file)
    sentence_index = 0
    sent_options = []
    instance_options = []

    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            json_line = json_file_object.readline()
            if json_line:
                json_line = json.loads(json_line)
                tokens = json_line["tokens"]
                p_ids = [dic["p_id"] for dic in json_line["pas"]]
            else:
                raise RuntimeError("Can not align surface file with json file.")

            line = line.rstrip("\n")
            len_sent = len(line.split(" "))
            assert len(tokens) == len_sent

            mask_ids = p_ids
            sent_options.append((mask_ids, line.split(" "), len_sent))

            for mask_id in mask_ids:
                surfaces = line.split(" ")
                surfaces[mask_id] = "[MASK]"
                text_a = " ".join(surfaces)
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=None))
                unique_id += 1
                instance_options.append((sentence_index, mask_id))

            sentence_index += 1

    print("# Size of original sentence: {}\n# Size of masking sentence: {}".format(sentence_index, unique_id))

    return examples, sent_options, instance_options


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--case_file", type=str)
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--base2index", type=str)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--sampling_prob", type=float, default=0.5)
    parser.add_argument("--fill_mode", type=str, default=None,
                        help="Choose from 'best_n', 'best_n_surface', ...")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.fill_mode.startswith("predicate"):
        examples, sent_options, instance_options = read_examples_and_mask_pred(args.input_file, args.json_file)
    else:
        examples, sent_options, instance_options = read_examples_and_mask(args.input_file, args.case_file)

    # add
    # split_sentence_dir = "/work01/ryuto/data/NTC_BERT_split"
    # split_sentence_file = os.path.join(split_sentence_dir, os.path.basename(args.output_file))
    # if os.path.exists(split_sentence_file):
    #     os.remove(split_sentence_file)
    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    # features = convert_examples_to_features(
    #     examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_subwords = [feature.tokens for feature in features]

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()

    config = FillerConfig(subword_vocab=tokenizer.vocab,
                          pre_trained_vocab_file=args.base2index,
                          json_file=args.json_file,
                          sent_options=sent_options,
                          instance_options=instance_options,
                          all_subwords=all_subwords)

    if args.fill_mode == "n_best":
        filler = BestNTokenFiller(config, n_best=args.n_best, mode="json")
    elif args.fill_mode == "n_best_surface":
        filler = BestNTokenFiller(config, n_best=args.n_best, mode="surface")
    elif args.fill_mode == "multi_sampling":
        filler = MultiTokenFiller(config, n_sample=args.n_sample, prob=args.sampling_prob, mode="json")
    elif args.fill_mode == "multi_sampling_surface":
        filler = MultiTokenFiller(config, n_sample=args.n_sample, prob=args.sampling_prob, mode="surface")
    elif args.fill_mode == "predicate":
        filler = BestNTokenPredicateFiller(config, n_best=args.n_best, mode="json")
    elif args.fill_mode == "predicate_surface":
        filler = BestNTokenPredicateFiller(config, n_best=args.n_best, mode="surface")
    elif args.fill_mode == "random":
        filler = RandomNTokenFiller(config, n_sample=args.n_best, mode="json")
    elif args.fill_mode == "random_surface":
        filler = RandomNTokenFiller(config, n_sample=args.n_best, mode="surface")
    elif args.fill_mode == "sampling":
        filler = SamplingTokenFiller(config, n_sample=args.n_best, mode="json")
    elif args.fill_mode == "sampling_surface":
        filler = SamplingTokenFiller(config, n_sample=args.n_best, mode="surface")
    else:
        raise ValueError("Unsupported Value: {}".format(args.fill_mode))

    with open(args.output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_indices in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            prediction = model(input_ids, token_type_ids=None, attention_mask=input_mask)

            for scores in prediction:
                instances = filler(scores)
                if instances is not None:
                    print("\n".join(instances), file=writer)

        instances = filler.pop()
        if instances:
            print("\n".join(instances), file=writer)
    with open(args.output_file + ".distribution", "w") as fo:
        json.dump(filler.predict_token_distribution, fo)


if __name__ == "__main__":
    main()
