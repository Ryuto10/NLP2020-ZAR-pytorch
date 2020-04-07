import argparse
from os import path
import json
from tqdm import tqdm
from itertools import islice
import random

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
PAD = 0
MASK = "[MASK]"

WHERE_MASK = {"all": None, "noun": NOUN, "verb": VERB, "particle": PARTICLE, "symbol": SYMBOL,
              "content": CONTENT_POS, "function": FUNCTION_POS, "argument": None,
              "minus_verb_symbol": MINUS_VERB_SYMBOL, "minus_verb_symbol_function": MINUS_VERB_SYMBOL_FUNCTION}
WHICH_ARG = ["false", "true", "free"]


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_file', type=path.abspath, required=True, help="Path to input file.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, required=True, help="Path to output file.")

    parser.add_argument('--where_mask', type=str, default="all", help="Choose from {}".format(", ".join(WHERE_MASK)))
    parser.add_argument('--which_arg', type=str, default="free", help="Choose from {}".format(", ".join(WHICH_ARG)))
    parser.add_argument('--random_rate', type=float, default=1)
    parser.add_argument('--minus', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)

    return parser


def read_file(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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

    print("Where to mask: '{}'".format(args.where_mask))
    print("Whether to mask the argument: '{}'".format(args.which_arg))
    print("Random rate: {}".format(args.random_rate))
    print("Minus: {}".format(args.minus))

    unique_id = OFFSETS
    fo = open(args.out_file, "w")
    for instance in tqdm(read_file(args.in_file)):
        for pas in instance["pas"]:
            mask_indices = create_mask_indices(instance=instance,
                                               pas=pas,
                                               where_mask=args.where_mask,
                                               which_arg=args.which_arg,
                                               random_rate=args.random_rate,
                                               minus=args.minus,
                                               argument_w=argument_w)

            new_instance = {"tokens": [PAD if idx in mask_indices else token_idx
                                       for idx, token_idx in enumerate(instance["tokens"])],
                            "surfaces": [MASK if idx in mask_indices else surf
                                         for idx, surf in enumerate(instance["surfaces"])],
                            "pas": [pas],
                            "pos": instance["pos"],
                            "sentence id": instance["sentence id"],
                            "file name": instance["file name"],
                            "unique_id": unique_id}
            print(json.dumps(new_instance), file=fo)
            unique_id += 1
    fo.close()

    print("# Last Unique ID: {}".format(unique_id))
    print("# Save: {}".format(args.out_file))
    print("# --- contents ---")
    with open(args.out_file) as fi:
        for line in islice(fi, 5):
            print("> {} ... {}".format(line[:50], line[-50:]) if len(line) > 100 else line, end="")


if __name__ == "__main__":
    main()
