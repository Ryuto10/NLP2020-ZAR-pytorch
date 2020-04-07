import argparse
import json
from glob import glob
from os import path
from collections import defaultdict
from typing import List
import numpy as np


def create_arg_parser():
    parser = argparse.ArgumentParser(description='marge')
    parser.add_argument('--in_dir', '-i', help='input directory')

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

    model_name = "model"
    model_name += "-" + args["comment"] if args["comment"] else ""
    model_name += "-" + extract_name(args["pseudo"]) if args["pseudo"] else ""
    model_name += "-wiki" if args["wiki"] else ""
    model_name += "-bert" if args["bert"] else ""
    model_name += "-mp" if "multi_predicate" in args and args["multi_predicate"] else ""
    model_name += "-zero_drop" if "zero_drop" in args and args["zero_drop"] else ""
    model_name += "-no_mask" if "mapping_pseudo_train" in args and args["mapping_pseudo_train"] else ""
    model_name += "-zero_drop" if "zero_drop" in args and args["zero_drop"] else ""
    model_name += "-lr" + str(args["lr"])
    model_name += "-plr" + str(args["pseudo_lr"]) if args["train_method"] == "pre-train" else ""
    model_name += "-embdrop" + str(args["embed_dropout"]) if "embed_dropout" in args else ""

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


def main():
    args = create_arg_parser().parse_args()

    table = defaultdict(lambda: [])
    for file in glob(args.in_dir + "/*"):
        result = get_result(file)
        if result:
            table[result["model"] + "," + result["method"]].append(result["score"])

    columns = ["n_seed",
               "ALL", "DEP", "ZERO",
               "GA_PREC", "GA_REC", "GA_F1",
               "WO_PREC", "WO_REC", "WO_F1",
               "NI_PREC", "NI_REC", "NI_F1",
               "ALL_PREC", "ALL_REC", "ALL_F1",
               "DEP_GA_PREC", "DEP_GA_REC", "DEP_GA_F1",
               "DEP_WO_PREC", "DEP_WO_REC", "DEP_WO_F1",
               "DEP_NI_PREC", "DEP_NI_REC", "DEP_NI_F1",
               "DEP_ALL_PREC", "DEP_ALL_REC", "DEP_ALL_F1",
               "ZERO_GA_PREC", "ZERO_GA_REC", "ZERO_GA_F1",
               "ZERO_WO_PREC", "ZERO_WO_REC", "ZERO_WO_F1",
               "ZERO_NI_PREC", "ZERO_NI_REC", "ZERO_NI_F1",
               "ZERO_ALL_PREC", "ZERO_ALL_REC", "ZERO_ALL_F1"]
    print("model, method," + ",".join(columns))
    for model, scores in table.items():
        average_score = calculate_average_score(scores)
        print(model + "," + ",".join([average_score[k] for k in columns]))


if __name__ == "__main__":
    main()
