import argparse
import json
import copy

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
MODES = ["normal", "bert", "surf", "acl", "color"]


def get_vocab(file):
    vocab = {}
    with open(file) as fi:
        for line in fi:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                token, index = line.split("\t", 1)
            except:
                print(line)
            vocab[int(index)] = token
    return vocab


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


def main(args):
    if args.mode in ["normal", "bert"]:
        vocab = get_vocab(args.vocab)
    for instance in read_file(args.in_file):
        if args.mode == "normal":
            print("tokens:", " ".join(vocab[int(i)] for i in instance["tokens"]))
        elif args.mode == "surf":
            for pas in instance["pas"]:
                tokens = copy.deepcopy(instance["surfaces"])
                case_ids = [(c_id, case) for c_id, case in enumerate(pas["args"]) if case != 3]
                tokens[pas["p_id"]] = add_color(tokens[pas["p_id"]], "underline")
                for c_id, case in case_ids:
                    tokens[c_id] = add_color(tokens[c_id], CASE_COLOR[case])
                print("".join(tokens))
        elif args.mode == "bert":
            print("tokens:", " ".join(vocab[int(i)] for i in instance["tokens"]))
            print("Orig:", "".join(instance["surfaces"]))
            print("MASK:", "".join(instance["text_a"]))
            print("Pred:", "".join(instance["bert_predicts"]))
            print()
        elif args.mode == "acl":
            case_ids = [(c_id, case) for c_id, case in enumerate(instance["pas"][0]["args"]) if case != 3]
            if args.before <= len(instance["surfaces"]) < args.after:
                tokens = copy.deepcopy(instance["surfaces"])
                tokens[instance["pas"][0]["p_id"]] = add_color(tokens[instance["pas"][0]["p_id"]], "underline")
                for c_id, case in case_ids:
                    tokens[c_id] = add_color(tokens[c_id], CASE_COLOR[case])
                print("".join(tokens))
                print("".join(["[MASK]" if ti == 0 else su
                               for ti, su in zip(instance["tokens"], instance["original_surfaces"])]))
                tokens = copy.deepcopy(instance["original_surfaces"])
                tokens[instance["pas"][0]["p_id"]] = add_color(tokens[instance["pas"][0]["p_id"]], "underline")
                for c_id, case in case_ids:
                    tokens[c_id] = add_color(tokens[c_id], CASE_COLOR[case])
                print("".join(tokens))
                print()
        elif args.mode == "color":
            print("Orig:", "".join(instance["surfaces"]))
            print("Pred:", "".join(instance["bert_predicts"]))
            for idx, pas in enumerate(instance["pas"], 1):
                case_ids = [(c_id, case) for c_id, case in enumerate(pas["args"]) if case != 3]
                tokens = copy.deepcopy(instance["bert_predicts"])
                tokens[pas["p_id"]] = add_color(tokens[pas["p_id"]], "underline")
                for c_id, case in case_ids:
                    tokens[c_id] = add_color(tokens[c_id], CASE_COLOR[case])
                for insert_id in instance["insert_ids"]:
                    tokens[insert_id] = add_color(tokens[insert_id], "reverse")
                print("Arg{}".format(idx), "".join(tokens))
            print()
        else:
            raise ValueError("Unsupported value: {}".format(args.mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check tokens')
    parser.add_argument('in_file', help='input file')
    parser.add_argument('--vocab', default="/home/ryuto/data/NTC_Matsu_original/wordIndex.txt",
                        help="path to wordIndex.txt. (Please add this option when 'mode' is 'normal' or 'bert')")
    parser.add_argument('--mode', default="normal",
                        help="Choose from {}.".format(", ".join(MODES)))
    parser.add_argument('--before', type=int, default=10)
    parser.add_argument('--after', type=int, default=20)

    main(parser.parse_args())
