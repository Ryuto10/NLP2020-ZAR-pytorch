import re
import os
import argparse
from glob import glob
import shutil


LOG_TRAIN = "log-train-"
LOG_DEV = "log-dev-"
LOG_TEST = "log-test-"
PREDICT = "predict-test-"
PREDICT_DEV = "predict-dev-"
EVAL = "evaluation-test-"
MODEL = "model-"
HEADS = [LOG_TRAIN, LOG_DEV, LOG_TEST, PREDICT, PREDICT_DEV, EVAL, MODEL]


def get_expname(file):
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)
    for head in HEADS:
        basename = basename.replace(head, "")
    basename = basename.replace("pre-train", "pre_train")
    exp_name = os.path.join(dirname, re.sub(re.search("_sub[0-9]+?(.+)", basename).group(1), "", basename))

    return exp_name


def main(args):
    exp_names = set()

    # get dir names
    for file in glob(args.in_dir + "/*"):
        if os.path.isfile(file):
            exp_names.add(get_expname(file))

    # make dir
    for exp_name in exp_names:
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)

    # move file
    for file in glob(args.in_dir + "/*"):
        if os.path.isfile(file):
            exp_name = get_expname(file)
            shutil.move(file, exp_name)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='marge')
    parser.add_argument('--in_dir', '-i', help='input directory')

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    main(args)
