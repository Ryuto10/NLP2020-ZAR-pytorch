import argparse
from os import path
import json
from tqdm import tqdm

NULL = 3
NULL_TYPE = "null"
IGNORE = 4


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_file', type=path.abspath, help="Path to input file.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, help="Path to output file.")
    parser.add_argument('--case_type', type=str, help="Choose from 'dep' or 'zero'.")

    return parser


def read_instance(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.case_type != "dep" and args.case_type != "zero":
        raise ValueError("Unsupported value: '{}'".format(args.case_type))
    if path.exists(args.out_file):
        raise FileExistsError("Already exists: '{}'".format(args.out_file))

    print("in: {}\nout: {}".format(args.in_file, args.out_file))
    fo = open(args.out_file, "w")
    for instance in tqdm(read_instance(args.in_file)):
        for pas in instance["pas"]:
            indices = [idx for idx, t in enumerate(pas["types"]) if t != args.case_type and t != NULL_TYPE]
            pas["args"] = [IGNORE if idx in indices else case for idx, case in enumerate(pas["args"])]
        print(json.dumps(instance), file=fo)
    fo.close()


if __name__ == "__main__":
    main()