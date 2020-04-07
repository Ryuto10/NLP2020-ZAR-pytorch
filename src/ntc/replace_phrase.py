import argparse
import copy
import json
import random
from itertools import chain
from os import path

import numpy as np
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', '-i', dest='in_file', type=path.abspath, help="Path to input file.")
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, help="Path to output file.")
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--use_original', action='store_true')

    return parser


def read_instance(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def create_can_replace_indices(instance):

    def create_bunsetsu_chunks(instance):
        chunks = []
        chunk = []
        for idx, v in enumerate(instance["bunsetsu"]):
            if idx != 0 and v:
                chunks.append(chunk)
                chunk = [idx]
            else:
                chunk.append(idx)
            if idx == len(instance["bunsetsu"]) - 1:
                chunks.append(chunk)
        assert sum(len(chunk) for chunk in chunks) == len(instance["tokens"])

        return chunks

    def extract_child_range_longest_match(parallel, tree):
        for idx in parallel:
            if idx == 0:
                yield 0, 1
            else:
                candidates = [idx]
                for index, head in tree[idx - 1::-1]:
                    if head in candidates:
                        candidates.append(index)
                start = min(candidates)
                end = idx + 1

                yield start, end

    def is_valid_indices(indices, instance):
        p_ids = [pas["p_id"] for pas in instance["pas"]]

        # Create c_ids
        c_ids = [0] * len(instance["tokens"])
        for pas in instance["pas"]:
            for idx, case in enumerate(pas["args"]):
                if case != 3:
                    c_ids[idx] = 1

        black_list = ["「", "」"]

        if all(c_ids[idx] == 0 for idx in indices):
            return False

        for idx in indices:
            if idx in p_ids or instance["surfaces"][idx] in black_list:
                return False

        return True

    chunks = create_bunsetsu_chunks(instance)
    tree = [tpl for tpl in instance["tree"] if tpl]
    assert len(tree) == len(chunks)

    parallels = []
    for idx in range(len(tree)):
        parallel = [tpl[0] for tpl in tree if tpl[1] == idx]
        if len(parallel) > 1:
            parallel_indices = [[i for i in chain.from_iterable(chunks[start:end])]
                                for start, end in extract_child_range_longest_match(parallel, tree)]
            valid_indices = []
            for indices in parallel_indices:
                if is_valid_indices(indices, instance):
                    valid_indices.append(indices)
            if len(valid_indices) > 1:
                parallels.append(valid_indices)

    return parallels


def create_replaced_instance(instance):

    def replace_list(before, after, ls):
        npls = np.array(ls)
        l = npls[:before[0]].tolist()
        m = npls[before[-1] + 1:after[0]].tolist()
        r = npls[after[-1] + 1:len(ls)].tolist()
        replaced = l + npls[after].tolist() + m + npls[before].tolist() + r

        return replaced

    parallels = create_can_replace_indices(instance)
    if not parallels:
        return None

    copy_instance = copy.deepcopy(instance)
    for parallel in parallels:
        before, after = random.sample(parallel, k=2)
        if before[0] > after[0]:
            before, after = after, before

        copy_instance["surfaces"] = replace_list(before, after, copy_instance["surfaces"])
        copy_instance["tokens"] = replace_list(before, after, copy_instance["tokens"])
        copy_instance["bases"] = replace_list(before, after, copy_instance["bases"])
        for pas in copy_instance["pas"]:
            pas["args"] = replace_list(before, after, pas["args"])
            pas["types"] = replace_list(before, after, pas["types"])

    return copy_instance


def main():
    parser = create_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    if not path.exists(args.in_file):
        raise FileNotFoundError("Input file doesn't exist: {}".format(args.in_file))
    if path.exists(args.out_file):
        raise FileExistsError("Already exists: {}".format(args.out_file))

    counter = 0
    fo = open(args.out_file, "w")
    for instance in tqdm(read_instance(args.in_file)):
        org_surf = "".join(instance["surfaces"])
        replaced_instance = create_replaced_instance(instance)

        if replaced_instance and counter < 10:
            counter += 1
            print("org:", org_surf)
            print("rpc:", "".join(replaced_instance["surfaces"]))

        if replaced_instance:
            print(json.dumps(replaced_instance), file=fo)
        elif args.use_original:
            print(json.dumps(instance), file=fo)

    fo.close()


if __name__ == '__main__':
    main()
