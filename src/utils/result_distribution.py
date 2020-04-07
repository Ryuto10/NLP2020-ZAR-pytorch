import argparse
import json
from os.path import exists
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pandas as pd

CASE = {0: "ga", 1: "o", 2: "ni"}
CASE_TYPE = {"dep": 0, "zero": 1}


class UnkSelector(object):
    def __init__(self, freq, unk=False):
        self.freq = freq
        self.unk = unk

    def __call__(self, token_idx):
        """input: instance of validation dataset."""
        if token_idx not in self.freq and self.unk:  # contain unk arg
            return True

        elif token_idx in self.freq and not self.unk:
            return True


class FreqSelector(object):
    def __init__(self, freq, freq_min=0, freq_max=5):
        self.freq = freq
        self.freq_min = freq_min
        self.freq_max = freq_max

    def __call__(self, token_idx):
        """
        If 'freq_min <= freq <= freq_max', return True.
        If v_max = -1, there is no upper limit.
        """
        if self.freq_max == -1 and self.freq_min <= self.freq[token_idx]:
            return True

        if self.freq_min <= self.freq[token_idx] <= self.freq_max:
            return True


def create_arg_freq(file):
    freq = defaultdict(int)
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            tokens = instance["tokens"]
            for pas in instance["pas"]:
                for idx, label in enumerate(pas["args"]):
                    if label != 3:
                        freq[tokens[idx]] += 1

    return freq


def iter_gold(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            tokens = instance["tokens"]
            for pas in instance["pas"]:
                p_id = pas["p_id"]
                case_dict = {}
                for idx, label in enumerate(pas["args"]):
                    if label != 3:
                        case_dict[CASE[label]] = idx

                yield tokens, case_dict, p_id, instance["sentence id"], instance["file name"]


def iter_predict(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue

            yield json.loads(line)


def iter_case_type(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue

            yield line.split()


def calculate_score(predict_arg, gold_arg, case_types, tokens, selectors, scores, instance_counter, false_positive):
    # axis 0: [dep, zero]
    # axis 1: [ga, o, ni]
    # axis 2: [tp, fp, tn]
    for label_idx, label in CASE.items():
        if label in gold_arg:
            case_type = case_types[gold_arg[label]]
            assert case_type != "null"
            type_idx = CASE_TYPE[case_type]

            if label in predict_arg:
                if predict_arg[label] == gold_arg[label]:
                    score = [1, 0, 0]
                else:
                    score = [0, 1, 1]
            else:
                score = [0, 0, 1]

            for key, selector in selectors.items():
                if selector(tokens[gold_arg[label]]):
                    scores[key][type_idx][label_idx] += score
                    instance_counter[key][type_idx] += 1

        # false positive
        elif label in predict_arg:
            case_type = case_types[predict_arg[label]]
            if case_type == "dep":
                false_positive[0][label_idx] += [0, 1, 0]
            else:
                false_positive[1][label_idx] += [0, 1, 0]


def calculate_false_positive(scores, instance_counter, false_positive):
    n = sum(sum(v) for v in instance_counter.values())
    for key, score in scores.items():
        for label_idx in [0, 1, 2]:
            score[0][label_idx] += false_positive[0][label_idx] * sum(instance_counter[key]) / n
            score[1][label_idx] += false_positive[1][label_idx] * sum(instance_counter[key]) / n


def calculate_f1(tp, fp, tn):
    prec = tp / (tp + fp)
    recall = tp / (tp + tn)
    if prec == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)

    return [prec * 100, recall * 100, f1 * 100]


def get_selector(args):
    freq = create_arg_freq(args.train_fn)
    if args.mode == "unk":
        selectors = {"unk": UnkSelector(freq, True),
                     "no_unk": UnkSelector(freq, False)}
    else:
        selectors = {"1~2": FreqSelector(freq, freq_min=1, freq_max=2),
                     "3~5": FreqSelector(freq, freq_min=3, freq_max=5),
                     "6~10": FreqSelector(freq, freq_min=6, freq_max=10),
                     "11~20": FreqSelector(freq, freq_min=11, freq_max=20),
                     "21~35": FreqSelector(freq, freq_min=21, freq_max=35),
                     "36~60": FreqSelector(freq, freq_min=36, freq_max=60),
                     "61~100": FreqSelector(freq, freq_min=61, freq_max=100),
                     "101~150": FreqSelector(freq, freq_min=101, freq_max=150),
                     "151~400": FreqSelector(freq, freq_min=151, freq_max=400),
                     "401~": FreqSelector(freq, freq_min=401, freq_max=-1)}

    return selectors


def get_df_evaluation(score):
    result = []
    index = []
    for case_type, type_idx in CASE_TYPE.items():
        for case_idx, case in CASE.items():
            result.append(calculate_f1(*score[type_idx][case_idx]))
            index.append("{} {}".format(case, case_type))
        result.append(calculate_f1(*sum(score[type_idx][:])))
        index.append("all {}".format(case_type))
    result.append(calculate_f1(*sum(sum(score))))
    index.append("all")

    return pd.DataFrame(result, index=index, columns=["precision", "recall", "f1"])


def main(args):
    selectors = get_selector(args)
    scores = {key: np.zeros((2, 3, 3)) for key in selectors}
    false_positive = np.zeros((2, 3, 3))
    instance_counter = {key: [0, 0] for key in selectors}

    if args.mode == "unk":
        dataset = args.dev_fn
    elif args.mode == "train_freq":
        dataset = args.train_fn
    elif args.mode == "dev_freq":
        dataset = args.dev_fn
    else:
        raise ValueError

    for gold, predict, case_type in tqdm(zip(iter_gold(dataset), iter_predict(args.prediction), iter_case_type(args.case_type))):
        tokens = gold[0]
        gold_arg = gold[1]
        assert predict["pred"] == gold[2]
        assert predict["sent"] == gold[3]
        assert predict["file"] == gold[4]

        calculate_score(predict, gold_arg, case_type, tokens, selectors, scores, instance_counter, false_positive)
    calculate_false_positive(scores, instance_counter, false_positive)

    with open(args.out_fn, "w") as fo:
        all_score = np.zeros((2, 3, 3))
        for key, score in scores.items():
            all_score += score
            df = get_df_evaluation(score)
            print("# {}".format(key), file=fo)
            print(df, file=fo)
            print("", file=fo)

        df = get_df_evaluation(all_score)
        print("# ALL", file=fo)
        print(df, file=fo)

        # all_score = all_score[0] + all_score[1]
        # result = []
        # for case_idx, case in CASE.items():
        #     result.append(calculate_f1(*all_score[case_idx]))
        # result.append(calculate_f1(*sum(all_score)))
        # df = pd.DataFrame(result, index=["ga", "o", "ni", "all"], columns=["precision", "recall", "f1"])
        # print("# ALL", file=fo)
        # print(df, file=fo)

        print(instance_counter, file=fo)


def get_arguments():
    parser = argparse.ArgumentParser(description='select instance')
    parser.add_argument('--train_fn', '-t', type=str, default="/work01/ryuto/data/NTC_ratio05_train/train_ratio05.json")
    parser.add_argument('--dev_fn', '-d', type=str, default="/work01/ryuto/BERT_PAS/outdir/instances-dev.txt")
    parser.add_argument('--prediction', '-p', type=str)
    parser.add_argument('--case_type', '-c', type=str, default="/work01/ryuto/data/NTC_Matsu_converted/dev.casetype")
    parser.add_argument('--out_fn', '-o', type=str)
    parser.add_argument('--mode', '-m', type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_arguments()
    # if exists(args.out_fn):
    #     raise FileExistsError("'{}' already exists.".format(args.out_fn))

    main(args)
