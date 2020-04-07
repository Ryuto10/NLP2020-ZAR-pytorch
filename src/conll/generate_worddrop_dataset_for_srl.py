import argparse
from os import path
import json
from tqdm import tqdm
from itertools import islice
import random

NOUN = ["NN", "NNS", "NNP", "NNPS"]
ADJ = ["JJ", "JJR", "JJS"]
ADV = ["RB", "RBR", "RBS", "RP"]
VB = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
SYM = ["#", "$", ".", ",", ":", "(", ")", "\"", "'", "``", "`", "''"]
HIGH_FNC = ["IN", "DT", "CD", "AUX", "CC", "TO", "PRP", "POS", "MD"]

MODE = {"noun": NOUN, "adj": ADJ, "adv": ADV, "vb": VB, "fnc": NOUN + ADJ + ADV + VB, "symbol": SYM,
        "IN": ["IN"], "DT": ["DT"], "CD": ["CD"], "AUX": ["AUX"],
        "CC": ["CC"], "TO": ["TO"], "PRP": ["PRP"], "POS": ["POS"], "MD": ["MD"],
        "random": None, "low_fnc": NOUN + ADJ + ADV + VB + HIGH_FNC}
MODE_USAGE = "Choose from {}".format(",".join(MODE))


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_file', type=path.abspath, help="Path to input file.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, help="Path to output file.")
    parser.add_argument('--mode', type=str, help=MODE_USAGE)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2020)

    return parser


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

    random.seed(args.seed)

    if path.exists(args.out_file):
        raise FileExistsError("Already exists: {}".format(args.out_file))
    if args.mode not in MODE:
        raise ValueError("Unsupported mode = '{}'\nUsage: {}".format(args.mode, MODE_USAGE))
    if args.mode == "fnc" or args.mode == "low_fnc":
        is_target = lambda x: x not in MODE[args.mode]
    elif args.mode == "random":
        is_target = lambda x: random.random() < args.drop_rate
    else:
        is_target = lambda x: x in MODE[args.mode]

    fo = open(args.out_file, "w")
    for instance in tqdm(read_file(args.in_file)):
        for arg in instance["labels"]:
            tokens = ["[MASK]" if is_target(pos) and idx != arg["verb_idx"] else token
                      for idx, (token, pos) in enumerate(zip(instance["tokens"], instance["pos_tags"]))]
            instance["tokens"] = tokens
            instance["labels"] = [arg]
            print(json.dumps(instance), file=fo)
    fo.close()

    print("# Save: {}".format(args.out_file))
    print("# --- contents ---")
    with open(args.out_file) as fi:
        for line in islice(fi, 5):
            print("> {} ... {}".format(line[:50], line[-50:]) if len(line) > 100 else line, end="")


if __name__ == "__main__":
    main()
