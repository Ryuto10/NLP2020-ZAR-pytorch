import argparse
import os
import re
from glob import glob
from tqdm import tqdm


def get_files(obj):
    if os.path.isfile(obj):
        return [obj]
    files = []
    for in_obj in glob(obj + "/*"):
        files += get_files(in_obj)
    return files


def main(args):
    for file in tqdm(get_files(args.in_dir)):
        match = re.search(r"_PLR(.*?)-", file)
        if match:
            dir_name = os.path.dirname(file)
            base_name = os.path.basename(file)

            plr = match.group(1)
            base_name = base_name.replace(match.group(0), "-")
            base_name = re.sub(r"(_lr.+?_)", r"\1plr{}_".format(plr), base_name)
            after_fn = os.path.join(dir_name, base_name)
            if not os.path.exists(after_fn):
                os.rename(file, after_fn)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='marge')
    parser.add_argument('--in_dir', '-i', help='input directory')

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    main(args)
