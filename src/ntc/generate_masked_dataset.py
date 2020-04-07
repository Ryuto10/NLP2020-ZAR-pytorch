# coding=utf-8

import argparse
import copy
import json
import logging
import random
from collections import defaultdict
from typing import List

from toolz import sliding_window
from tqdm import tqdm

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "_padding_"
DONT_INSERT = ["する", "なる", "いる", "ある", "ない"]
BLACK_LIST = ["が", "は", "を", "に", "の", "も", "「", "」", "。", "『", "』",
              "：", "（", "）", UNK, "へ", "と", "こと", "し"]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_masked_instances(args):
    """how mask"""
    # Load input instances
    with open(args.input_file) as fi:
        input_instances = [json.loads(line) for line in fi]

    print("# Create Masked Instances")
    print("## Number of MASK positions: {} ~ {}".format(args.position_min, args.position_max))
    print("## Number of MASK inserts: {} ~ {}".format(args.insert_min, args.insert_max))
    instances = []
    data_size = int(len(input_instances) * args.data_ratio / 100)
    for in_instance in tqdm(input_instances[:data_size]):
        # Create instance
        positions: {int: List[str]} = create_insert_positions(in_instance)
        n_choice = random.randint(min(args.position_min, len(positions)),
                                  min(args.position_max, len(positions)))
        if args.how_mask == "insert":
            for position, black_list in random.sample(positions.items(), k=n_choice):
                n_insert = random.randint(args.insert_min, args.insert_max)
                instance = inserted_instance(instance=in_instance, position=position, n_insert=n_insert)
                instance["black_list"] = list(set(black_list))
                instances.append(instance)
        elif args.how_mask == "replace":
            positions[len(in_instance["surfaces"])] = []
            ranges = create_replace_ranges(positions)
            for (start, end), black_list in random.sample(ranges.items(), k=min(n_choice, len(ranges))):
                n_insert = random.randint(args.insert_min, args.insert_max)
                instance = replaced_instance(instance=in_instance, start=start, end=end, n_insert=n_insert)
                if instance:
                    instance["black_list"] = list(set(black_list))
                    instances.append(instance)
        else:
            raise ValueError("Mask mechanism %s not found!" % args.how_mask)
    print("# Number of instances: {} -> {}".format(data_size, len(instances)))

    return instances


def inserted_instance(instance, position, n_insert):
    """example:
    input:
        surfaces = 太郎 が 走った
        position = 2
        n_insert = 3
    change:
        text_a = 太郎 が [MASK] [MASK] [MASK] 走った
        mask_ids = [2, 3, 4]
        p_id: 2 -> 5
        args: [0, 3, 3] -> [0, 3, 3, 3, 3, 3]
        types: [dep, null, null] -> [dep, null, null, null, null]
    """
    # Copy
    new_instance = copy.deepcopy(instance)

    # Create text_a
    new_instance["text_a"] = instance["surfaces"][0:position] + [MASK] * n_insert + instance["surfaces"][position:]

    # Create mask_ids
    new_instance["mask_ids"] = [idx for idx, token in enumerate(new_instance["text_a"]) if token == MASK]

    # Shift tokens
    new_instance["tokens"] = instance["tokens"][0:position] + [0] * n_insert + instance["tokens"][position:]

    # Shift args
    for pas in new_instance["pas"]:
        pas["p_id"] = pas["p_id"] + n_insert if pas["p_id"] >= position else pas["p_id"]
        pas["args"] = pas["args"][0:position] + [3] * n_insert + pas["args"][position:]
        pas["types"] = pas["types"][0:position] + ["null"] * n_insert + pas["types"][position:]

    return new_instance


def replaced_instance(instance, start, end, n_insert):
    """example:
    input:
        surfaces = 太郎 が ボール を 投げた
        start = 2
        end = 4
        n_insert = 3
    change:
        text_a = 太郎 が [MASK] [MASK] [MASK] 投げた
        mask_ids = [2, 3, 4]
        p_id: 4 -> 5
        args: [0, 3, 1, 3, 3] -> [0, 3, 3, 3, 3, 3]
        types: [dep, null, dep, null, null] -> [dep, null, null, null, null, null]
    """
    # Copy
    new_instance = copy.deepcopy(instance)

    # Shift args
    for pas in new_instance["pas"]:
        if start <= pas["p_id"] < end:
            return None
        pas["p_id"] = pas["p_id"] + start - end + n_insert if pas["p_id"] >= end else pas["p_id"]
        pas["args"][start:end] = [3] * n_insert
        pas["types"][start:end] = ["null"] * n_insert

    # Shift tokens
    new_instance["tokens"][start:end] = [0] * n_insert

    # Create text_a
    new_instance["text_a"] = copy.deepcopy(instance["surfaces"])
    new_instance["text_a"][start:end] = [MASK] * n_insert

    # Create mask_ids
    new_instance["mask_ids"] = [idx for idx, token in enumerate(new_instance["text_a"]) if token == MASK]

    return new_instance


def create_insert_positions(instance) -> {int: List[str]}:
    """文節区切りの左に項，右に述語項構造解析がくる場所について，{position: black_list} となる dict を作成"""
    insert_positions = defaultdict(lambda: [])
    bunsetsu_ids = [idx for idx, binary in enumerate(instance["bunsetsu"]) if binary == 1]
    for b_id in bunsetsu_ids:

        # insertする場所の後ろの単語が「する」などの場合，insertしない
        if instance["bases"][b_id] in DONT_INSERT:
            continue

        for pas in instance["pas"]:
            for c_id, case in enumerate(pas["args"]):
                if case != 3 and c_id < b_id <= pas["p_id"]:
                    insert_positions[b_id].append(instance["surfaces"][pas["p_id"]])
                    insert_positions[b_id].append(instance["bases"][pas["p_id"]])
                    insert_positions[b_id].append(instance["surfaces"][c_id])
                    insert_positions[b_id].append(instance["bases"][c_id])

    return insert_positions


def create_replace_ranges(positions: {int: List[str]}) -> {(int, int): List[str]}:
    ranges = {(pair[0][0], pair[1][0]): pair[0][1] + pair[1][1] for pair in sliding_window(2, positions.items())}

    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
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
    parser.add_argument("--augment_strategy", dest='how_mask', default="insert", type=str,
                        help="Choose from 'insert' or 'replace'")
    parser.add_argument("--token_strategy", dest='how_select', default="argmax", type=str,
                        help="Choose from 'argmax' or 'sample'")

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--insert_max', type=int, default=7)
    parser.add_argument('--insert_min', type=int, default=3)
    parser.add_argument('--position_max', type=int, default=3)
    parser.add_argument('--position_min', type=int, default=1)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # Create MASK instances
    instances = create_masked_instances(args)

    with open(args.output_file, "w", encoding='utf-8') as writer:
        for instance in instances:
            print(json.dumps(instance), file=writer)


if __name__ == "__main__":
    main()
