import argparse
import json
from os import path

import h5py
from logzero import logger
from tqdm import tqdm

MASK = "[MASK]"


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo", type=path.abspath, required=True, help="json")
    parser.add_argument("--pseudo_matrix", type=path.abspath, required=True, help="hdf5")
    parser.add_argument("--out", type=path.abspath, required=True, help="hdf5")

    # Option of (pseudo, train) -> new
    parser.add_argument("--train_matrix", type=path.abspath, help="hdf5")
    parser.add_argument("--map", type=path.abspath, help="json")

    # Option of (pseudo, single_pseudo) -> new
    parser.add_argument("--single_pseudo_matrix", type=path.abspath, help="hdf5")

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

    logger.info(args)

    if path.exists(args.out):
        raise FileExistsError("'{}' already exists.".format(args.out))

    source = h5py.File(args.pseudo_matrix, "r")
    fo = h5py.File(args.out, 'w')

    # load
    logger.info("Loading")
    if args.train_matrix and args.map:
        mode = "from_train"
        with open(args.map) as fi:
            mapping_pseudo_train = json.load(fi)
        target = h5py.File(args.train_matrix, "r")
    elif args.single_pseudo_matrix:
        mode = "from_pseudo"
        target = h5py.File(args.single_pseudo_matrix, "r")
    else:
        raise ValueError("Set the options required to swap.")
    logger.debug("Mode: {}".format(mode))

    for instance in tqdm(read_instance(args.pseudo)):
        unique_id = str(instance["unique_id"])

        # extract vector
        source_vec = source.get(unique_id)[()]
        if mode == "from_train":
            target_vec = target.get(mapping_pseudo_train[unique_id])[()]
        elif mode == "from_pseudo":
            target_vec = target.get(unique_id)[()]
        else:
            raise RuntimeError

        assert source_vec.shape == target_vec.shape

        # swap
        mask_ids = [idx for idx, token in enumerate(instance["surfaces"]) if token == MASK]
        for mask_idx in mask_ids:
            source_vec[mask_idx] = target_vec[mask_idx]

        # save
        fo.create_dataset(unique_id, data=source_vec)

    # close
    target.close()
    source.close()
    fo.close()


if __name__ == "__main__":
    main()
