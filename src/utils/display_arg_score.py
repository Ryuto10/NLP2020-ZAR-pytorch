import argparse
import json
import re
from os import path

import pandas

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


def get_score(line):
    prec = re.findall(PREC, line)[0].strip()
    rec = re.findall(REC, line)[0].strip()
    f1 = re.findall(F1, line)[0].strip()

    return float(prec), float(rec), float(f1)


def get_result(file):
    result = {}
    fi = open(file)
    for line in fi:
        for search_line, key in SEARCH:
            if search_line in line:
                result[key] = get_score(line)
    fi.close()

    return result


def main():
    parser = argparse.ArgumentParser(description='get score (prec, rec, f1)')
    parser.add_argument('in_file', help='evaluation file')
    args = parser.parse_args()

    if not path.exists(args.in_file):
        raise FileNotFoundError("{} doesn't exist.".format(args.in_file))

    result = get_result(args.in_file)
    df = pandas.DataFrame(result, ["Precision", "Recall", "F1"])
    print(df.T)

    with open(path.join(path.dirname(args.in_file), "test.score"), "w") as fo:
        json.dump(result, fo)


if __name__ == '__main__':
    main()
