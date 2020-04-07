import argparse
import copy
import json
import random
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from pyknp import Juman
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.distributions import Categorical
from tqdm import tqdm

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
BLACK_LIST = [UNK]

COLORS = {"black": '\033[30m',
          "red": '\033[31m',
          "green": '\033[32m',
          "yellow": '\033[33m',
          "blue": '\033[34m',
          "purple": '\033[35m',
          "cyan": '\033[36m',
          "white": '\033[37m',
          "end": '\033[0m',
          "bold": '\033[1m',
          "underline": '\033[4m',
          "invisible": '\033[08m',
          "reverse": '\033[07m'}
CASE_COLOR = {0: "yellow", 1: "blue", 2: "cyan"}

juman = Juman()

OFFSETS = 10 ** 5
NOUN = ["名詞", "接尾辞"]
VERB = ["動詞"]
PARTICLE = ["助詞"]
SYMBOL = ["特殊"]
CONTENT_POS = ["名詞", "接尾辞", "動詞", "形容詞", "副詞", "接頭辞"]
FUNCTION_POS = ["助詞", "助動詞", "特殊"]
MINUS_VERB_SYMBOL = ["動詞", "特殊"]
MINUS_VERB_SYMBOL_FUNCTION = ["動詞", "特殊", "助詞", "助動詞"]
ARGUMENT_RATE = path.join(path.dirname(__file__), "argument-rate.txt")

WHERE_MASK = {"all": None, "noun": NOUN, "verb": VERB, "particle": PARTICLE, "symbol": SYMBOL,
              "content": CONTENT_POS, "function": FUNCTION_POS, "argument": None,
              "minus_verb_symbol": MINUS_VERB_SYMBOL, "minus_verb_symbol_function": MINUS_VERB_SYMBOL_FUNCTION}
WHICH_ARG = ["false", "true", "free"]


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_file', type=path.abspath, required=True, help="Path to input file.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, required=True, help="Path to output file.")

    parser.add_argument("--bert_model", default="/home/ryuto/data/jawiki-kurohashi-bert", type=str,
                        help="Please fill the path to directory of BERT model, or the name of BERT model."
                             "Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument('--where_mask', type=str, default="all", help="Choose from {}".format(", ".join(WHERE_MASK)))
    parser.add_argument('--which_arg', type=str, default="free", help="Choose from {}".format(", ".join(WHICH_ARG)))
    parser.add_argument('--random_rate', type=float, default=1)
    parser.add_argument('--minus', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument("--how_select", dest='how_select', default="argmax", type=str,
                        help="Choose from 'argmax' or 'sample' or 'beam'. (default='argmax')")
    parser.add_argument("--how_many", default='multi', type=str,
                        help="Choose from 'single' or 'multi'. (default='multi')")
    parser.add_argument('--topk', type=int, default=5, help="for beam search")

    return parser


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens: [str, ...], tokenizer: BertTokenizer, max_seq_length: int):
        self.max_seq_length = max_seq_length if max_seq_length else 0
        self.tokens = [CLS] + tokenizer.tokenize(" ".join(tokens)) + [SEP]
        self.token_mask_ids = [idx for idx, token in enumerate(self.tokens) if token == MASK]
        self.len = len(self.tokens)

        if self.max_seq_length and self.len > self.max_seq_length:
            logger.warning("'tokens_a' is over {}: {}".format(max_seq_length, self.len))
            # raise RuntimeError("'tokens_a' is over {}: {}".format(max_seq_length, self.len))
        else:
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


def read_file(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def add_color(token, color=None, end=True):
    if color:
        token = COLORS[color] + token
    if end:
        token += COLORS["end"]
    return token


def create_mask_indices(instance: dict,
                        pas: dict,
                        where_mask: str,
                        which_arg: str,
                        random_rate: float,
                        minus: bool,
                        argument_w: dict):
    # maskする対象をrandom rateに従って選ぶ
    if where_mask == "all":
        mask_indices = {idx for idx, token in enumerate(instance["pos"])
                        if random.random() < random_rate}
    elif where_mask == "argument":
        mask_indices = {idx for idx, pos in enumerate(instance["pos"])
                        if pos in argument_w and random.random() < argument_w[pos] * random_rate}
    elif where_mask == "minus_verb_symbol":
        mask_indices = {idx for idx, pos in enumerate(instance["pos"])
                        if pos.split("-")[0] not in WHERE_MASK[where_mask] and random.random() < random_rate}
    elif where_mask == "minus_verb_symbol_function":
        mask_indices = {idx for idx, pos in enumerate(instance["pos"])
                        if pos.split("-")[0] not in WHERE_MASK[where_mask] and random.random() < random_rate}
    elif minus:
        mask_indices = {idx for idx, pos in enumerate(instance["pos"])
                        if pos.split("-")[0] not in WHERE_MASK[where_mask] and random.random() < random_rate}
    else:
        mask_indices = {idx for idx, pos in enumerate(instance["pos"])
                        if pos.split("-")[0] in WHERE_MASK[where_mask] and random.random() < random_rate}

    # 項をmaskするかどうか
    arg_indices = {idx for idx, case in enumerate(pas["args"]) if case != 3}
    if which_arg == "true":
        mask_indices = mask_indices | arg_indices
    elif which_arg == "false":
        mask_indices = mask_indices - arg_indices

    # 対象の述語はmaskしない
    mask_indices = mask_indices - {pas["p_id"]}

    return mask_indices


def test_create_mask_indices(argument_w):
    instance = {"pos": ["接尾辞-名詞性述語接尾辞-*", "名詞-形式名詞-*", "接尾辞-名詞性名詞接尾辞-*", "形容詞-*-イ形容詞アウオ段",
                        "動詞-*-サ変動詞", "助詞-副助詞-*", "特殊-読点-*", "助動詞-*-ナ形容詞", "接頭辞-名詞接頭辞-*",
                        "名詞-副詞的名詞-*", "副詞-*-*", "動詞-*-子音動詞ワ行", "助詞-接続助詞-*", "特殊-句点-*"]}
    pas = {"p_id": 4, "args": [3, 3, 0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3]}

    examples = [(("all", "free", 1), (1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
                (("all", "free", 0), (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                (("all", "true", 0), (-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1)),
                (("all", "false", 1), (1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1)),
                (("noun", "false", 1), (1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                (("noun", "free", 1), (1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1)),
                (("verb", "true", 1), (-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)),
                (("particle", "free", 1), (-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1)),
                (("symbol", "false", 1), (-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1)),
                (("minus_verb_symbol", "false", 1), (1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1)),
                (("minus_verb_symbol_function", "false", 1), (1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1)),
                (("content", "false", 1), (1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1)),
                (("function", "true", 1), (-1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1))]
    for x, y in examples:
        mask_indices = create_mask_indices(instance, pas, *x, False, argument_w)
        assert len(instance["pos"]) == len(y)
        for idx, p in enumerate(y):
            if p == 1:
                assert idx in mask_indices
            elif p == -1:
                assert idx not in mask_indices


def main():
    parser = create_parser()
    args = parser.parse_args()
    logger.info(args)

    random.seed(args.seed)

    with open(ARGUMENT_RATE) as fi:
        argument_w = {line.split()[0]: float(line.rstrip("\n").split()[-1]) for line in fi}
    test_create_mask_indices(argument_w)

    if path.exists(args.out_file):
        raise FileExistsError("Already exists: {}".format(args.out_file))
    if args.where_mask not in WHERE_MASK:
        raise ValueError("Unsupported mode = '{}'\nChoose from: {}".format(args.where_mask, WHERE_MASK))
    if args.which_arg not in WHICH_ARG:
        raise ValueError("Unsupported mode = '{}'\nChoose from: {}".format(args.which_arg, WHICH_ARG))

    logger.info("Where to mask: '{}'".format(args.where_mask))
    logger.info("Whether to mask the argument: '{}'".format(args.which_arg))
    logger.info("Random rate: {}".format(args.random_rate))
    logger.info("Minus: {}".format(args.minus))
    logger.info("How select tokens: {}".format(args.how_select))
    logger.info("How many tokens to predict at once: {}".format(args.how_many))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: {}".format(device))
    logger.info("BERT model: {}".format(args.bert_model))
    logger.debug("Loading BERT model...")
    max_seq_length = 128
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    black_list = []

    logger.debug("sort by length of tokens")
    instances = [instance for instance in tqdm(read_file(args.in_file))]
    sorted_instances = sorted(instances, key=lambda x: len(x["surfaces"]))
    logger.debug("sort is done")

    fo = open(args.out_file, "w")
    logger.debug("Start to fill the mask")
    for instance in tqdm(sorted_instances[10000:15000]):
        for pas in instance["pas"]:
            if len(set(pas["args"])) == 1:
                continue
            if "zero" not in pas["types"]:
                continue

            predict_sents = []
            mask_indices = create_mask_indices(instance=instance,
                                               pas=pas,
                                               where_mask=args.where_mask,
                                               which_arg=args.which_arg,
                                               random_rate=args.random_rate,
                                               minus=args.minus,
                                               argument_w=argument_w)
            if not mask_indices:
                continue

            original_tokens = copy.deepcopy(instance["surfaces"])
            masked_tokens = [MASK if idx in mask_indices else surf for idx, surf in enumerate(instance["surfaces"])]
            feature = InputFeatures(tokens=masked_tokens,
                                    tokenizer=tokenizer,
                                    max_seq_length=max_seq_length)

            if feature.len > max_seq_length:
                continue

            if args.how_select == "beam":
                output_sents, output_tokens = prediction_with_beam_search(device=device,
                                                                          model=model,
                                                                          feature=feature,
                                                                          tokenizer=tokenizer,
                                                                          black_list=black_list,
                                                                          k=args.topk)
                for sent in output_sents:
                    predict_sents.append(sent[1:feature.len - 1])

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

                filled_tokens = copy.deepcopy(masked_tokens)
                for idx, p_token in zip(sorted(list(mask_indices)), predict_tokens):
                    filled_tokens[idx] = p_token
                predict_sents.append(filled_tokens)

            print("{}: {}".format(instance["file name"], instance["sentence id"]), file=fo)
            for idx, tokens in enumerate([original_tokens, masked_tokens, *predict_sents]):
                case_ids = [(c_id, case) for c_id, case in enumerate(pas["args"]) if case != 3]
                tokens[pas["p_id"]] = add_color(tokens[pas["p_id"]], "underline")
                for c_id, case in case_ids:
                    tokens[c_id] = add_color(tokens[c_id], CASE_COLOR[case])
                print("{} :{}".format(idx, " ".join(tokens)), file=fo)
            print("\n", file=fo)
    fo.close()
    logger.info("done")


if __name__ == "__main__":
    main()
