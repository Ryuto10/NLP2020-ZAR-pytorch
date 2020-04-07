import argparse
from os import path
import json
import logzero
from logzero import logger
from os import path


def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=path.abspath, help="Path to train file.")
    parser.add_argument('--pseudo', type=path.abspath, help="Path to pseudo file.")
    parser.add_argument('--map', type=path.abspath)
    parser.add_argument('--out', '-o', dest='out_file', type=path.abspath, help="Path to output file.")

    return parser


def read_file(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = create_parser()
    args = parser.parse_args()

    dir_name = path.abspath(__file__)
    logger.debug(dir_name)

    if args.map:
        with open(args.map) as fi:
            mapping_index = json.load(fi)

    train_surf_unique_id = [(inst["surfaces"], inst["unique_id"])
                            for inst in read_file(args.train)
                            for _ in inst["pas"]]
    logger.info(len(train_surf_unique_id))
    counter = 0
    with open(args.out_file, "w") as fo:
        for inst, (original_surfaces, unique_id) in zip(read_file(args.pseudo), train_surf_unique_id):
            inst["original_surfaces"] = original_surfaces
            if args.map:
                assert mapping_index[str(inst["unique_id"])] == str(unique_id)
            print(json.dumps(inst), file=fo)
            counter += 1

            if counter < 50:
                logger.debug(inst["surfaces"])
                logger.debug(original_surfaces)
    assert counter == len(train_surf_unique_id)


if __name__ == "__main__":
    main()