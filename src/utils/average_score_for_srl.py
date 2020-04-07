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
    model_name += "-highway" if args["highway"] else ""
    model_name += "-attention" if args["attention"] else ""
    model_name += "-mp" if args["multi_predicate"] else ""
    model_name += "-glove" if args["glove"] else ""
    model_name += "-elmo" if args["elmo"] else ""
    model_name += "-bert" if args["bert"] else ""
    model_name += "-xlnet" if args["xlnet"] else ""
    model_name += "-lr" + str(args["learning_rate"])
    model_name += "-" + path.basename(args["pseudo"]).split(".", 1)[0] if args["pseudo"] else ""

    return {"model": model_name, "method": args["train_method"], "seed": args["seed"], "score": test_score}


def calculate_average_score(scores: List[dict]):
    seed_scores = defaultdict(lambda: [])
    n_seed = len(scores)
    for score in scores:
        for k, v in score.items():
            seed_scores[k].append(v)
    average_scores = {"n_seed": str(n_seed)}
    for k, v in seed_scores.items():
        v = [i * 100 for i in v]
        ave = sum(v) / n_seed
        std = np.std(v)
        average_scores[k] = "{:.2f}±{:.2f}".format(ave, std)

    return average_scores


def main():
    args = create_arg_parser().parse_args()

    table = defaultdict(lambda: [])
    for file in glob(args.in_dir + "/*"):
        result = get_result(file)
        if result:
            table[result["model"] + "," + result["method"]].append(result["score"])

    columns = ["n_seed", "precision-overall", "recall-overall", "f1-measure-overall"]
    print("model, method," + ",".join(columns))
    for model, scores in table.items():
        average_score = calculate_average_score(scores)
        print(model + "," + ",".join([average_score[k] for k in columns]))


if __name__ == "__main__":
    main()
