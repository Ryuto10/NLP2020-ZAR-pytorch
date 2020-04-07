import argparse
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

LOG_REGEX = re.compile("log-train-")
LOSS_REGEX = re.compile(r"loss: tensor\((.+?)\) lr:")
F1_REGEX = re.compile(r"all 	p: .+? 	r: .+? 	f1: (.+?)\s")
ORIGINAL_REGEX = re.compile("---------- Original data ----------")


def get_loss_f1(file):
    loss_pseudo = []
    loss_original = []
    f1_pseudo = []
    f1_original = []
    original = False

    with open(file) as fi:
        for line in fi:
            if not original and ORIGINAL_REGEX.search(line):
                original = True

            if original:
                loss = LOSS_REGEX.search(line)
                if loss:
                    loss_original.append(float(loss.group(1)))
                f1 = F1_REGEX.search(line)
                if f1:
                    f1_original.append(float(f1.group(1)))
            else:
                loss = LOSS_REGEX.search(line)
                if loss:
                    loss_pseudo.append(float(loss.group(1)))
                f1 = F1_REGEX.search(line)
                if f1:
                    f1_pseudo.append(float(f1.group(1)))

    return loss_pseudo, loss_original, f1_pseudo, f1_original


def make_figure(pseudo, original, out_file, title=""):
    df = pd.DataFrame({"train-pseudo": pseudo + len(original) * [None], "original": len(pseudo) * [None] + original},
                      range(1, len(pseudo) + len(original) + 1))
    plt.figure()
    df.plot(title=title)
    if not os.path.exists(out_file):
        plt.savefig(out_file)
    plt.close('all')


def get_files(obj):
    if os.path.isfile(obj):
        return [obj]
    files = []
    for in_obj in glob(obj + "/*"):
        files += get_files(in_obj)
    return files


def get_log_files(obj):
    logs = []
    files = get_files(obj)
    for file in files:
        basename = os.path.basename(file)
        if LOG_REGEX.search(basename):
            logs.append(file)
    return logs


def main(args):
    log_files = get_log_files(args.in_dir)
    for file in tqdm(log_files):
        loss_pseudo, loss_original, f1_pseudo, f1_original = get_loss_f1(file)
        loss_file = LOG_REGEX.sub("loss-", file).replace(".txt", ".png")
        f1_file = LOG_REGEX.sub("f1-", file).replace(".txt", ".png")
        make_figure(loss_pseudo[7:], loss_original, loss_file, "loss")
        make_figure(f1_pseudo, f1_original, f1_file, "f1")


def create_arg_parser():
    parser = argparse.ArgumentParser(description='marge')
    parser.add_argument('--in_dir', '-i', help='input directory')

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    main(args)
