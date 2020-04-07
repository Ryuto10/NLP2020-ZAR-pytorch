import argparse
from os import path
import json

import h5py
from logzero import logger
from tqdm import tqdm
import torch

MASK = "[MASK]"


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=path.abspath,
                        default="/groups2/gcb50246/ryuto/work/data/ntc-processed/train.jsonl")
    parser.add_argument('--pseudo', type=path.abspath,
                        default="/groups2/gcb50246/ryuto/work/data/ntc-worddrop/minus_verb_symbol-free-07.train.jsonl")
    parser.add_argument('--embed_dir', type=path.abspath,
                        default="/groups2/gcb50246/ryuto/work/data/ntc-bert-embed")
    parser.add_argument('--map', type=path.abspath,
                        default="/groups2/gcb50246/ryuto/work/data/ntc-bert-embed/pseudo_to_train_for_unique_id.json")

    return parser


def extract_matrix_name(dir_name: str, file_name: str):
    base_name = path.basename(file_name)
    pseudo_name = ".".join(base_name.split(".")[:-1])
    matrix_name = path.join(dir_name, str(pseudo_name) + ".hdf5")
    if not path.exists(matrix_name):
        raise FileNotFoundError("'{}' is not found.".format(matrix_name))

    return matrix_name


def read_instance(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def length_fileobj(file):
    with open(file) as fi:
        for i, _ in enumerate(fi, 1):
            pass
    return i


def main():
    parser = create_parser()
    args = parser.parse_args()

    with open(args.map) as fi:
        mapping_pseudo_train = json.load(fi)

    logger.info(args)
    train_matrix_name = extract_matrix_name(args.embed_dir, args.train)
    logger.info(train_matrix_name)
    pseudo_matrix_name = extract_matrix_name(args.embed_dir, args.pseudo)
    logger.info(pseudo_matrix_name)

    train_matrix = h5py.File(train_matrix_name, "r")
    pseudo_matrix = h5py.File(pseudo_matrix_name, "r")

    original = 0
    mask = 0
    unmask = 0
    mask_original = 0
    unmask_original = 0
    n_mask = 0
    n_unmask = 0

    len_file = length_fileobj(args.pseudo)

    for idx, instance in tqdm(enumerate(read_instance(args.pseudo)), total=len_file):
        unique_id = str(instance["unique_id"])

        # extract vector
        train_vec = torch.Tensor(train_matrix.get(mapping_pseudo_train[unique_id])[()])
        pseudo_vec = torch.Tensor(pseudo_matrix.get(unique_id)[()])

        mask_ids = [idx for idx, token in enumerate(instance["surfaces"]) if token == MASK]
        unmask_ids = [idx for idx, token in enumerate(instance["surfaces"]) if token != MASK]

        assert len(instance["surfaces"]) == len(mask_ids) + len(unmask_ids)

        # sum
        original += torch.sum(train_vec.norm(dim=1))
        mask += torch.sum(pseudo_vec[mask_ids].norm(dim=1))
        unmask += torch.sum(pseudo_vec[unmask_ids].norm(dim=1))
        mask_original += torch.sum((train_vec[mask_ids] - pseudo_vec[mask_ids]).norm(dim=1))
        unmask_original += torch.sum((train_vec[unmask_ids] - pseudo_vec[unmask_ids]).norm(dim=1))
        n_mask += len(mask_ids)
        n_unmask += len(unmask_ids)

        if idx < 4:
            logger.info("Iteration: {}".format(idx))
            logger.debug("Original: {}".format(original / (n_mask + n_unmask)))
            logger.debug("Mask: {}".format(mask / n_mask))
            logger.debug("Unmask: {}".format(unmask / n_unmask))
            logger.debug("Mask - Original: {}".format(mask_original / n_mask))
            logger.debug("Unmask - Original: {}".format(unmask_original / n_unmask))

    logger.info(args.pseudo)
    print("'Original': {},".format(original / (n_mask + n_unmask)))
    print("'Mask': {},".format(mask / n_mask))
    print("'Unmask': {},".format(unmask / n_unmask))
    print("'Mask - Original': {},".format(mask_original / n_mask))
    print("'Unmask - Original': {}".format(unmask_original / n_unmask))

    # close
    train_matrix.close()
    pseudo_matrix.close()


if __name__ == "__main__":
    main()
