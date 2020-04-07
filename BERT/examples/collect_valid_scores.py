# -*- coding: utf-8 -*-
"""
"""
import argparse
import glob
import os


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--path', default='./result', type=str, help='write here')
    args = parser.parse_args()
    return args


def get_best_valid_score(path):
    with open(path, 'r') as fi:
        eval_values = [float(l.strip().split()[-1]) for l in fi if l.startswith('eval_accuracy')]
        return sorted(eval_values, reverse=True)[0]


def main(args):
    query = os.path.join(args.path + 'eval_results.txt')
    result_files = glob.glob(query)

    for result_file in result_files:
        valid_score = get_best_valid_score(result_file)
        print('{}\t{}'.format(os.path.dirname(result_file), valid_score))


if __name__ == "__main__":
    args = get_args()
    main(args)
