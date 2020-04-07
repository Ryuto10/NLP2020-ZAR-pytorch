import argparse
import json
from collections import defaultdict
from glob import glob
from os import path
from typing import List
import pandas as pd

import logzero
import numpy as np
from logzero import logger

RANDOM_RATE_FILE = path.join(path.dirname(__file__), "random_rates.txt")


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_dir', type=path.abspath, help="Path to input directory.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, help="Path to output file.")
    parser.add_argument('--mode', '-m', type=str, help="Choose from 'graph', 'type1', 'type2', 'type3'")

    return parser


def extract_name(file):
    name = path.basename(file)
    name, _ = name.split(".", 1)

    return name


def get_result(file):
    args_fn = path.join(file, "args.json")
    score_fn = path.join(file, "test.score")
    if not path.exists(score_fn):
        return None

    with open(score_fn) as fi:
        test_score = json.load(fi)

    with open(args_fn) as fi:
        args = json.load(fi)

    model_name = "mp" if "multi_predicate" in args and args["multi_predicate"] else ""
    model_name += "-zero_drop" if "zero_drop" in args and args["zero_drop"] else ""
    model_name += "-no_mask" if "mapping_pseudo_train" in args and args["mapping_pseudo_train"] else ""
    model_name += "-" + extract_name(args["pseudo"]) if args["pseudo"] else "base"
    if args["pseudo_bert_embed_file"] and args["pseudo"]:
        pseudo_embed = extract_name(args["pseudo_bert_embed_file"])
        if pseudo_embed != extract_name(args["pseudo"]):
            model_name += "@" + extract_name(args["pseudo_bert_embed_file"])

    return {"model": model_name, "method": args["train_method"], "seed": args["seed"], "score": test_score}


def calculate_average_score(scores: List[dict]):
    seed_scores = defaultdict(lambda: [])
    n_seed = len(scores)
    for score in scores:
        for k, v in score.items():
            assert len(v) == 3
            for s, name in zip(v, ["_PREC", "_REC", "_F1"]):
                seed_scores[k + name].append(s)
    average_scores = {"n_seed": str(n_seed)}
    for k, v in seed_scores.items():
        ave = sum(v) / n_seed
        std = np.std(v)
        average_scores[k] = "{:.2f}±{:.2f}".format(ave, std)

    average_scores["ALL"] = average_scores["ALL_F1"]
    average_scores["DEP"] = average_scores["DEP_ALL_F1"]
    average_scores["ZERO"] = average_scores["ZERO_ALL_F1"]

    return average_scores


def type1(table):
    col1 = ["mp-"]
    col2 = ["base", "train", "noun", "particle", "verb", "symbol", "content", "function", "all"]
    col3 = ["false", "true", "free"]

    result = {}
    for c1 in col1:
        for c2 in col2:
            x_scores = []
            for c3 in col3:
                model_start_name = c2 + "-" + c3 if c2 != "base" and c2 != "train" else c1 + c2
                buffer = []
                for model, scores in table.items():
                    if model.startswith(model_start_name):
                        average_score = calculate_average_score(scores)
                        float_score = float(average_score["ALL"].split("±")[0])
                        str_score = average_score["ALL"]
                        buffer.append((float_score, str_score))
                if buffer:
                    max_score_index = np.argmax([i[0] for i in buffer])
                    max_score = buffer[int(max_score_index)][1]
                    x_scores.append(max_score)
            if x_scores:
                result[c1 + c2] = x_scores
    print(pd.DataFrame(result, col3).T.to_latex().replace("±", "$\\pm$"))


def type2(table):
    col = ["base", "train", "noun", "particle", "verb", "symbol", "content", "function", "all",
           "minus_noun", "minus_verb", "minus_particle", "minus_symbol", "minus_verb_symbol"]
    result_columns = ["F1", "SD", "DEP", "ZERO"]

    result = {}
    for c in col:
        model_start_name = "mp-" + c + "-free" if c != "base" and c != "train" else "mp-" + c
        buffer = []
        for model, scores in table.items():
            if model.startswith(model_start_name):
                average_score = calculate_average_score(scores)
                float_score = float(average_score["ALL"].split("±")[0])
                buffer.append((float_score, average_score))
        if buffer:
            max_score_index = np.argmax([i[0] for i in buffer])
            r = buffer[int(max_score_index)][1]
            result[c] = [r["ALL"].split("±")[0],
                         "±" + r["ALL"].split("±")[1],
                         r["DEP"].split("±")[0],
                         r["ZERO"].split("±")[0]]
    print(pd.DataFrame(result, result_columns).T.to_latex().replace("±", "$\\pm$"))


def type3(table):
    col = ["mpbase", "mp-train", "mp-minus_verb_symbol-free",
           "mp-zero_drop-no_mask-minus_verb_symbol-free", "mp-zero_drop-minus_verb_symbol-free",
           "mp-multi-argmax", "mp-multi-sample", "mp-single-argmax", "mp-single-sample",
           "single-minus_verb_symbol-free", "mix_pseudox2", "mix_train_pseudo",
           "mp-minus_verb-free", "mp-zero_drop-no_mask-minus_verb-free", "mp-zero_drop-minus_verb-free",
           "mp-zero_drop-no_mask-minus_verb-free", "mp-mv5-multi-argmax"]
    result_columns = ["F1", "SD", "ALL", "NOM", "ACC", "DAT", "ALL", "NOM", "ACC", "DAT"]
    result = {}
    for c in col:
        buffer = []
        for model, scores in table.items():
            model = model.split("@")[-1]
            if model.startswith(c):
                average_score = calculate_average_score(scores)
                float_score = float(average_score["ALL"].split("±")[0])
                buffer.append((float_score, average_score))
            if buffer:
                max_score_index = np.argmax([i[0] for i in buffer])
                r = buffer[int(max_score_index)][1]
                result[c] = [r["ALL"].split("±")[0],
                             "±" + r["ALL"].split("±")[1],
                             r["ZERO_ALL_F1"].split("±")[0],
                             r["ZERO_GA_F1"].split("±")[0],
                             r["ZERO_WO_F1"].split("±")[0],
                             r["ZERO_NI_F1"].split("±")[0],
                             r["DEP_ALL_F1"].split("±")[0],
                             r["DEP_GA_F1"].split("±")[0],
                             r["DEP_WO_F1"].split("±")[0],
                             r["DEP_NI_F1"].split("±")[0]]

    print(pd.DataFrame(result, result_columns).T.to_latex().replace("±", "$\\pm$"))


def graph(table, out_file):
    col3 = ["noun", "particle", "verb", "symbol", "content", "function", "all",
            "minus_noun", "minus_particle", "minus_verb", "minus_symbol", "minus_verb_symbol"]
    col4 = ["01", "03", "05", "07", "09", "1"]

    with open(RANDOM_RATE_FILE) as fi:
        random_rates = {line.split()[0]: float(line.rstrip("\n").split()[-1]) for line in fi}

    datas = {}
    for c3 in col3:
        model_start_name = "mp-" + c3 + "-free"
        x = []
        y = []
        for c4 in col4:
            rate = float("." + c4) * 10
            for model, scores in table.items():
                if model.startswith(model_start_name + "-" + c4):
                    average_score = calculate_average_score(scores)
                    x.append(rate * random_rates["free-" + c3])
                    y.append(float(average_score["ZERO"].split("±")[0]))
                    break
        if y:
            graph_name = c3.replace("minus", "all").replace("_", "-")
            data_frame = {"random rate": x, graph_name: y}
            datas[graph_name] = data_frame

    with open(out_file, "w") as fo:
        json.dump(datas, fo)


def main():
    parser = create_parser()
    args = parser.parse_args()

    logfn = ".logzero.log"
    logzero.logfile(logfn, disableStderrLogger=True)
    dir_name = path.abspath(__file__)
    logger.debug(dir_name)

    table = defaultdict(lambda: [])
    for file in glob(args.in_dir + "/*"):
        result = get_result(file)
        if result:
            table[result["model"]].append(result["score"])

    logger.info("mode: {}".format(args.mode))
    if args.mode == "graph":
        graph(table, args.out_file)
    elif args.mode == "type1":
        type1(table)
    elif args.mode == "type2":
        type2(table)
    elif args.mode == "type3":
        type3(table)
    else:
        raise ValueError("Unsupported value: {}".format(args.mode))


if __name__ == "__main__":
    main()
