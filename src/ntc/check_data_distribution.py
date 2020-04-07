import json
import argparse
from collections import defaultdict

VOCAB = "/home/ryuto/data/NTC_Matsu_original/wordIndex.txt"
DEV = "/work01/ryuto/BERT_PAS/outdir/instances-dev.txt"


def get_pred_arg(file):
    vocab = get_vocab(VOCAB)
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            tokens = json.loads(line)["tokens"]
            pas = json.loads(line)["pas"]
            for dic in pas:
                predicate = tokens[dic["p_id"]]
                args = [tokens[idx] for idx, label in enumerate(dic["args"]) if label != 3]
                for arg in args:
                    yield predicate, vocab[arg]


def get_vocab(file, flip=True):
    vocab = {}
    with open(file) as fi:
        for line in fi:
            line = line.rstrip("\n")
            if line:
                k, v = line.split("\t", 1)
                if flip:
                    vocab[v] = k
                else:
                    vocab[k] = v
    return vocab


def main(args):
    vocab = defaultdict(int)
    for predicate, arg in get_pred_arg(args.in_file):
        vocab[arg] += 1
    with open(args.out_file, "w") as fo:
        json.dump(vocab, fo)


def get_arguments():
    parser = argparse.ArgumentParser(description='make dataset')
    parser.add_argument("--in_file", '-i', type=str)
    parser.add_argument("--out_file", '-o', type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)
