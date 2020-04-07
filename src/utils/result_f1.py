import argparse
import json
import os.path
from glob import glob
import re
import pandas as pd

HYPARAS = ['attention', 'dropout_rate', 'BiDAF', 'force', 'Hidden_state_dim', 'learning_rate',
           'Layers', 'multi_head', 'Onesignal', 'seed', 'Times', 'update_disable']

IN_S_GA = ('Args PRED IN_S ga', 'GA')
IN_S_O = ('Args PRED IN_S o', 'WO')
IN_S_NI = ('Args PRED IN_S ni', 'NI')
IN_S_ALL = ('Args PRED IN_S ALL', 'ALL')

IN_S = [IN_S_GA, IN_S_O, IN_S_NI, IN_S_ALL]

DEP_GA = ('Args PRED DEP ga', 'DEP_GA')
DEP_O = ('Args PRED DEP o', 'DEP_WO')
DEP_NI = ('Args PRED DEP ni', 'DEP_NI')
DEP_ALL = ('Args PRED DEP ALL', 'DEP_ALL')

DEP = [DEP_GA, DEP_O, DEP_NI, DEP_ALL]

ZERO_GA = ('Args PRED IN_S_NOT_DEP ga', 'ZERO_GA')
ZERO_O = ('Args PRED IN_S_NOT_DEP o', 'ZERO_WO')
ZERO_NI = ('Args PRED IN_S_NOT_DEP ni', 'ZERO_NI')
ZERO_ALL = ('Args PRED IN_S_NOT_DEP ALL', 'ZERO_ALL')

ZERO = [ZERO_GA, ZERO_O, ZERO_NI, ZERO_ALL]

SEARCH = IN_S + DEP + ZERO

PREC = r'P:\s+?(.+?)\('
REC = r'R:\s+?(.+?)\('
F1 = r'F:\s+?(.+?)pp'

EVAL_REGEX = re.compile("evaluation-test-model-")


def get_files(obj):
    if os.path.isfile(obj):
        return [obj]
    files = []
    for in_obj in glob(obj + "/*"):
        files += get_files(in_obj)
    return files


def get_eval_files(obj):
    evals = []
    files = get_files(obj)
    for file in files:
        basename = os.path.basename(file)
        if EVAL_REGEX.search(basename):
            evals.append(file)
    return evals


def get_score(line):
    prec = re.findall(PREC, line)[0].strip()
    rec = re.findall(REC, line)[0].strip()
    f1 = re.findall(F1, line)[0].strip()

    return float(prec), float(rec), float(f1)


def get_result(file) -> dict:
    result = {}
    fi = open(file)
    for line in fi:
        for search_line, key in SEARCH:
            if search_line in line:
                result[key] = get_score(line)
    fi.close()

    return result


def get_hypara_name(file):
    dirname = os.path.dirname(file)
    model_id = os.path.basename(dirname)
    exp_theme, hypara = model_id.split("-", 1)
    seed = re.search(r"_sub([0-9]+?)$", hypara).group(1)
    hypara = re.sub(r"_sub[0-9]+?$", "", hypara)

    return exp_theme, hypara, seed


def average_scores(results):
    df = None
    for idx, result in enumerate(results):
        if idx == 0:
            df = pd.DataFrame(result, ["prec", "rec", "f1"])
        else:
            df += pd.DataFrame(result, ["prec", "rec", "f1"])
    len_seed = len(results)
    df = df / len_seed
    f1 = df["ALL"][2]

    return df, f1, len_seed


def main(args):
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print("# Make Directory: {}".format(args.out_dir))

    results = {}
    eval_files = sorted(get_eval_files(args.in_dir))
    for file in eval_files:
        result = get_result(file)
        out_file = os.path.dirname(file) + "/result.txt"
        if not os.path.exists(out_file):
            with open(out_file, "w") as fo:
                json.dump(result, fo)
        exp_theme, hypara, seed = get_hypara_name(out_file)

        if not exp_theme in results:
            results[exp_theme] = {hypara: [result]}
        elif not hypara in results[exp_theme]:
            results[exp_theme][hypara] = [result]
        else:
            results[exp_theme][hypara].append(result)

    for exp_theme in results:
        out_fn = os.path.join(args.out_dir, exp_theme + ".result")
        if os.path.exists(out_fn):
            continue
        with open(out_fn, "w") as fo:
            best_f1 = 0
            best_score = None
            best_hypara = None
            best_len_seed = None
            for hypara in results[exp_theme]:
                df, f1, len_seed = average_scores(results[exp_theme][hypara])
                if best_f1 < f1:
                    best_f1 = f1
                    best_score = df
                    best_hypara = hypara
                    best_len_seed = len_seed
                print("# Parameter: {}".format(hypara), file=fo)
                print("# Number of seed: {}".format(len_seed), file=fo)
                print(df.T, file=fo)
                print(file=fo)
            print("# Best score", file=fo)
            print("# Parameter: {}".format(best_hypara), file=fo)
            print("# Number of seed: {}".format(best_len_seed), file=fo)
            print(best_score.T, file=fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get score (prec, rec, f1)')
    parser.add_argument('--in_dir', '-i',  help='input directory')
    parser.add_argument('--out_dir', '-o', help='output directory')
    main(parser.parse_args())
