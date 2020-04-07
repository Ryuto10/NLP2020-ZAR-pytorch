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
from toolz import sliding_window
from torch.distributions import Categorical
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "_padding_"
CASE_PARTICLES = ["が", "の", "を", "に", "へ", "と", "より", "から", "で"]
BLACK_LIST = [UNK, "。", "、", "—"]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

juman = Juman()


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, tokens, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length if max_seq_length else 0
        self.tokens = [CLS] + tokenizer.tokenize(" ".join(tokens)) + [SEP]
        self.len = len(self.tokens)

        if self.max_seq_length and self.len > self.max_seq_length:
            raise RuntimeError("'tokens_a' is over {}: {}".format(max_seq_length, self.len))

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

    output_ids = input_ids[0].tolist()
    for idx, token in zip(token_mask_ids, topk_dict[0][1]):
        output_ids[idx] = token
    output_sent = tokenizer.convert_ids_to_tokens(output_ids)[1:feature.len - 1]
    output_tokens = tokenizer.convert_ids_to_tokens(topk_dict[0][1])

    return output_sent, output_tokens


def prediction(model, feature, tokenizer, how_select) -> List[str]:
    """how select"""
    all_predict_ids = []

    # text_a
    input_ids = feature.ids_a
    token_mask_ids = [idx for idx, token in enumerate(feature.tokens_a) if token == MASK]
    for token_mask_id in token_mask_ids:
        predict = model(input_ids, token_type_ids=None, attention_mask=feature.mask_a)
        predict[:, token_mask_id, tokenizer.convert_tokens_to_ids(
            BLACK_LIST + [t for t in feature.instance["black_list"] if t in tokenizer.vocab]
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

    # text_b
    if feature.tokens_b:
        input_ids = torch.cat((input_ids[:, :feature.len_a - 1], feature.ids_b), dim=-1)
        token_mask_ids = [idx for idx, token in enumerate(feature.tokens_b, feature.len_a - 1) if token == MASK]
        for token_mask_id in token_mask_ids:
            predict = model(input_ids, token_type_ids=None, attention_mask=feature.mask_ab)
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

    predict_tokens = tokenizer.convert_ids_to_tokens(all_predict_ids)

    return predict_tokens


def create_feature(instance, tokenizer, max_seq_length):
    tokens_a = [CLS] + tokenizer.tokenize(" ".join(instance["text_a"])) + [SEP]
    if tokens_a > max_seq_length:
        raise RuntimeError("'tokens_a' is over {}: {}".format(max_seq_length, len(tokens_a)))

    tokens_b = [CLS] + tokenizer.tokenize(" ".join(instance["text_a"])) + [SEP] if "text_b" in instance else None
    if tokens_b and tokens_b > max_seq_length:
        raise RuntimeError("'tokens_b' is over {}: {}".format(max_seq_length, len(tokens_b)))

    input_ids = tokenizer.convert_tokens_to_ids(tokens_a) + [0] * (max_seq_length - len(tokens_a))
    input_mask = [1] * len(tokens_a) + [0] * (max_seq_length - len(tokens_a))
    feature = InputFeatures(input_ids, input_mask)

    return feature


def create_masked_instances(args, vocab):
    """how mask"""
    # Load input instances
    with open(args.input_file) as fi:
        input_instances = [json.loads(line) for line in fi]

    print("# Create Masked Instances")
    print("## Number of insert: {}".format(args.n_insert))
    print("## Type of MASK: {}".format(args.mask_strategy))
    instances = []
    data_size = int(len(input_instances) * args.data_ratio / 100)
    for in_instance in tqdm(input_instances[:data_size]):
        for pas in in_instance["pas"]:
            for paths, pair, triple, position_type in extract_paths(in_instance, pas):
                if position_type == "before":
                    if args.mask_strategy == "alpha-a" or args.mask_strategy == "all":
                        new_instance = alpha_a(in_instance, pas, paths, pair, triple, vocab, args.n_insert)
                        instances.append(new_instance)
                    if (args.mask_strategy == "alpha-b" or args.mask_strategy == "all") and triple[1] == "が":
                        new_instance = alpha_b(in_instance, pas, paths, pair, triple, vocab, args.n_insert)
                        instances.append(new_instance)
                if position_type == "after":
                    if args.mask_strategy == "beta-a" or args.mask_strategy == "all":
                        new_instance = alpha_b(in_instance, pas, paths, pair, triple, vocab, args.n_insert)
                        instances.append(new_instance)
                    if args.mask_strategy == "beta-b" or args.mask_strategy == "all":
                        new_instance = beta_b(in_instance, pas, paths, pair, triple, vocab, args.n_insert)
                        instances.append(new_instance)
    print("# Number of instances: {} -> {}".format(data_size, len(instances)))

    return instances


def fix_tokens(instance, predict_tokens, vocab):
    c_subword, c_juman = 0, 0
    instance["insert_ids"] = []
    for k in sorted(instance["insert_position"]):
        morphs = juman.analysis("".join(token.strip("#") for token in predict_tokens[c_subword:c_subword + instance["insert_position"][k]]))
        n_tokens = len(morphs)
        convert_ids = []
        for i, morph in enumerate(morphs):
            base = morphs[i - 1].genkei + morph.genkei if i != 0 and morph.hinsi == "動詞" and morphs[i - 1].bunrui == "サ変名詞" else morph.genkei
            pos = morph.hinsi + "-" + morph.bunrui + "-" + morph.katuyou1
            if base in vocab:
                convert_ids.append(vocab[base])
            elif pos in vocab:
                convert_ids.append(vocab[pos])
            else:
                convert_ids.append(vocab[PAD])
        instance["tokens"][k + c_juman:k + c_juman] = convert_ids
        instance["pas"][0]["args"][k + c_juman:k + c_juman] = [3] * n_tokens
        if k + c_juman < instance["pas"][0]["p_id"]:
            instance["pas"][0]["p_id"] += n_tokens
        instance["insert_ids"] += list(range(k + c_juman, k + c_juman + n_tokens))
        instance["bert_predicts"][k + c_juman:k + c_juman] = [m.midasi for m in morphs]
        c_subword += instance["insert_position"][k]
        c_juman += n_tokens

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


# ----- MASK Generation -----

def extract_paths(instance, pas):
    """
    Yields:
        Paths： (項の前，項の後ろ，述語の前，述語の後ろ)
        pair: (target_word_index, predicate_index)
        triple: (target_word, particle, predicate)
        position_type： 'before' or 'after'
    """
    case_converter = {0: "が", 1: "を", 2: "に"}
    phrase_ids = [idx for idx, v in enumerate(instance["bunsetsu"]) if v == 1] + [len(instance["tokens"])]
    phrase_range = [(sta, end) for sta, end in sliding_window(2, phrase_ids)]
    tree = [p for p in instance["tree"] if p]
    assert len(phrase_range) == len(tree)

    predicate = instance["surfaces"][pas["p_id"]]
    zero_ids = [idx for idx, t in enumerate(pas["types"]) if t == "zero"]
    for zero_idx in zero_ids:
        path_zero_before, path_zero_after, path_predicate_before, path_predicate_after,  = {}, {}, {}, {}
        predicate_idx, arg_idx = None, None

        # Create Path from zero (after)
        target = None
        for (sta, end), (index, head) in zip(phrase_range, tree):
            if target is None and sta <= zero_idx < end:
                path_zero_after[index] = list(range(sta, end))
                arg_idx = index
                target = head
            elif target is not None and target == index:
                path_zero_after[index] = list(range(sta, end))
                target = head

        # Create Path from zero (before)
        target = None
        for (sta, end), (index, head) in zip(phrase_range[::-1], tree[::-1]):
            if target is None and sta <= zero_idx < end:
                path_zero_before[index] = list(range(sta, end))
                target = index
            elif target is not None and target == head:
                path_zero_before[index] = list(range(sta, end))
                target = index

        # Create Path from predicate (after)
        target = None
        for (sta, end), (index, head) in zip(phrase_range, tree):
            if target is None and sta <= pas["p_id"] < end:
                path_predicate_after[index] = list(range(sta, end))
                predicate_idx = index
                target = head
            if target is not None and target == index:
                path_predicate_after[index] = list(range(sta, end))
                target = head

        # Create Path from predicate (before)
        target = None
        for (sta, end), (index, head) in zip(phrase_range[::-1], tree[::-1]):
            if target is None and sta <= pas["p_id"] < end:
                path_predicate_before[index] = list(range(sta, end))
                target = index
            if target is not None and target == head:
                path_predicate_before[index] = list(range(sta, end))
                target = index

        paths = (path_zero_before, path_zero_after, path_predicate_before, path_predicate_after)
        pair = (instance["tokens"][zero_idx], instance["tokens"][pas["p_id"]])
        triple = (instance["surfaces"][zero_idx], case_converter[pas["args"][zero_idx]], predicate)
        position_type = "before" if arg_idx < predicate_idx else "after"

        yield paths, pair, triple, position_type


def alpha_a(instance, pas, paths, pair, triple, vocab, n_insert=(3, 5)):
    """α-a：「項 < 述語」 -> 「項 < 述語」へ 変換"""
    target_word, particle, predicate = triple

    # Merge path
    path_zero_before, path_zero_after, path_predicate_before, path_predicate_after = paths
    path = copy.deepcopy(path_zero_before)
    path.update(path_predicate_after)
    path.update(path_predicate_before)
    path.update(path_predicate_after)

    # Create Instance
    fake_args_idx = min(path_predicate_before)
    before_idx = min(path)
    end_idx = min(path_zero_after.keys() & path_predicate_after.keys())

    text_a, indices, insert_position = [], [], {}
    tree = [t for t in instance["tree"] if t is not None]

    for k in sorted(path):
        # 省略されていた部分を挿入
        if k == fake_args_idx:
            n = random.randint(*n_insert)
            insert_position[len(indices)] = n
            text_a += [MASK] * n
            text_a += ["、", "、", target_word, particle]
            indices.append("、")
        # Insert MASK
        elif before_idx + 1 != k:
            n = random.randint(*n_insert)
            insert_position[len(indices)] = n
            text_a += [MASK] * n

        indices += path[k]
        text_a += [instance["surfaces"][idx] for idx in path[k]]
        if k == end_idx:
            break
        before_idx = k
    # Insert MASK
    if end_idx != tree[-1][0]:
        n = random.randint(*n_insert)
        insert_position[len(indices)] = n
        text_a += [MASK] * n + ["。"]
        indices.append("。")

    # Create new instance
    tokens = [vocab[i] if type(i) == str else instance["tokens"][i] for i in indices]
    p_ids = [1 if idx == pas["p_id"] else 0 for idx in range(len(instance["tokens"]))]
    p_id = [0 if type(i) == str else p_ids[i] for i in indices].index(1)
    args = [3 if type(i) == str else pas["args"][i] for i in indices]
    surfs = [i if type(i) == str else instance["surfaces"][i] for i in indices]

    new_instance = copy.deepcopy(instance)
    new_instance["tokens"] = tokens
    new_instance["pas"] = [{"p_id": p_id, "args": args}]
    new_instance["insert_position"] = insert_position
    new_instance["text_a"] = text_a
    new_instance["black_list"] = triple
    new_instance["bert_predicts"] = surfs

    return new_instance


def alpha_b(instance, pas, paths, pair, triple, vocab, n_insert=(3, 5)):
    """
    α-b：「項 < 述語」 -> 「項 > 述語」へ 変換 (が格のみ)
    β-a：「項 > 述語」 -> 「項 > 述語」へ 変換 (が格のみ)
    """

    target_word, particle, predicate = triple

    # Merge path
    _, _, path_predicate_before, _ = paths

    # Create Instance
    predicate_idx = max(path_predicate_before)
    text_a, text_b, surfs, indices, insert_position = [], [], [], [], {}

    for k in sorted(path_predicate_before):
        if k == predicate_idx:
            for idx in path_predicate_before[k]:
                if idx == pas["p_id"]:
                    text_a += [instance["bases"][idx], "、"]
                    # surfs += [instance["bases"][idx], "、"]
                    # indices += [idx, str(vocab["、"])]
                    surfs.append(instance["bases"][idx])
                    indices.append(idx)
                    break
                else:
                    text_a += [instance["surfaces"][idx]]
                    surfs += [instance["surfaces"][idx]]
                    indices.append(idx)
            break
        else:
            indices += path_predicate_before[k]
            text_a += [instance["surfaces"][idx] for idx in path_predicate_before[k]]
            surfs += [instance["surfaces"][idx] for idx in path_predicate_before[k]]
    # Insert MASK (text_a)
    n = random.randint(*n_insert)
    insert_position[len(indices)] = n
    text_a += [MASK] * n + [target_word]
    surfs.append(target_word)
    indices.append(str(pair[0]))
    # Insert MASK (text_b)
    insert_case_particle = random.choice(CASE_PARTICLES)
    surfs.append(insert_case_particle)
    indices.append(str(vocab[insert_case_particle]))

    n = random.randint(*n_insert)
    insert_position[len(indices)] = n
    text_b = [insert_case_particle] + [MASK] * n + ["。"]
    surfs.append("。")
    indices.append(str(vocab["。"]))

    # Create new instance
    tokens = [int(i) if type(i) == str else instance["tokens"][i] for i in indices]
    p_ids = [1 if idx == pas["p_id"] else 0 for idx in range(len(instance["tokens"]))]
    p_id = [0 if type(i) == str else p_ids[i] for i in indices].index(1)
    args = [3 if type(i) == str else pas["args"][i] for i in indices]
    case_converter = {"が": 0, "を": 1, "に": 2}
    args[-3] = case_converter[particle]

    new_instance = copy.deepcopy(instance)
    new_instance["tokens"] = tokens
    new_instance["pas"] = [{"p_id": p_id, "args": args}]
    new_instance["insert_position"] = insert_position
    new_instance["text_a"] = text_a
    new_instance["text_b"] = text_b
    new_instance["black_list"] = triple
    new_instance["bert_predicts"] = surfs

    return new_instance


def beta_b(instance, pas, paths, pair, triple, vocab, n_insert=(3, 5)):
    """β-b：「項 > 述語」 -> 「項 > 述語」へ 変換"""
    target_word, particle, predicate = triple

    # Merge path
    path_zero_before, _, path_predicate_before, _ = paths

    # Create Instance
    text_a, indices, insert_position, before_idx = [], [], {}, 0
    tree = [t for t in instance["tree"] if t is not None]

    for k in sorted(path_zero_before):
        if k <= max(path_predicate_before):
            continue
        # Insert MASK
        if before_idx + 1 != k:
            n = random.randint(*n_insert)
            insert_position[len(indices)] = n
            text_a += [MASK] * n
        indices += path_zero_before[k]
        text_a += [instance["surfaces"][idx] for idx in path_zero_before[k]]
        before_idx = k

    # Insert MASK
    n = random.randint(*n_insert)
    insert_position[len(indices)] = n
    text_a += [MASK] * n + ["、", "、", target_word, particle]
    indices.append("、")

    for k in sorted(path_predicate_before):
        indices += path_predicate_before[k]
        text_a += [instance["surfaces"][idx] for idx in path_predicate_before[k]]
    # Insert MASK
    if max(path_predicate_before) != tree[-1][0]:
        n = random.randint(*n_insert)
        insert_position[len(indices)] = n
        text_a += [MASK] * n + ["。"]
        indices.append("。")

    tokens = [vocab[i] if type(i) == str else instance["tokens"][i] for i in indices]
    p_ids = [1 if idx == pas["p_id"] else 0 for idx in range(len(instance["tokens"]))]
    p_id = [0 if type(i) == str else p_ids[i] for i in indices].index(1)
    args = [3 if type(i) == str else pas["args"][i] for i in indices]
    surfs = [i if type(i) == str else instance["surfaces"][i] for i in indices]

    new_instance = copy.deepcopy(instance)
    new_instance["tokens"] = tokens
    new_instance["pas"] = [{"p_id": p_id, "args": args}]
    new_instance["insert_position"] = insert_position
    new_instance["text_a"] = text_a
    new_instance["black_list"] = triple
    new_instance["bert_predicts"] = surfs

    return new_instance


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
    parser.add_argument('--mask_strategy', type=str, default="all",
                        help="Choose from 'alpha-a', 'alpha-b', 'beta-a', 'beta-b' or 'all'")
    parser.add_argument('--n_insert', type=int, nargs=2, default=(3, 5))
    parser.add_argument('--topk', type=int, default=5)

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # vocab & tokenizer
    vocab = set_vocab(args.vocab)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    instances = create_masked_instances(args, vocab)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()

    print("## Beam Search Top K: {}".format(args.topk))

    with open(args.output_file, "w", encoding='utf-8') as writer:
        for instance in tqdm(instances):
            feature = InputFeatures(tokens=instance["text_a"], tokenizer=tokenizer, max_seq_length=args.max_seq_length)
            output_sent, output_tokens = prediction_with_beam_search(device=device,
                                                                     model=model,
                                                                     feature=feature,
                                                                     tokenizer=tokenizer,
                                                                     black_list=instance["black_list"],
                                                                     k=args.topk)
            if "text_b" in instance:
                tokens = output_sent + instance["text_b"]
                feature = InputFeatures(tokens=tokens, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
                output_sent, output_tokens_b = prediction_with_beam_search(device=device,
                                                                           model=model,
                                                                           feature=feature,
                                                                           tokenizer=tokenizer,
                                                                           black_list=[],
                                                                           k=args.topk)
                output_tokens += output_tokens_b

            output = fix_tokens(instance, output_tokens, vocab)
            print(json.dumps(output), file=writer)


if __name__ == "__main__":
    main()
