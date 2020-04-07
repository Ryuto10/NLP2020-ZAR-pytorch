import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

import alignment


def get_arguments():
    parser = argparse.ArgumentParser(description='Adjusts the subword vector to the original word vector.')
    parser.add_argument('--json_fn', '-j', type=str,
                        help='File name of json. (.json)')
    parser.add_argument('--text_fn', '-t', type=str,
                        help='File name of text. (.txt)')
    parser.add_argument('--split_fn', '-s', type=str, default="",
                        help='File name of sentence. (.split)')
    parser.add_argument('--vec_fn', '-v', type=str,
                        help='File name of vector extracted by bert. (.vec)')
    parser.add_argument('--out_dir', '-o', type=str, default='/work01/ryuto/data/NTC_BERT',
                        help='Output file name.')
    args = parser.parse_args()

    return args


def main(args):
    basename = os.path.basename(args.json_fn)
    out_fn = os.path.join(args.out_dir, basename)

    # Open file
    json_f = open(args.json_fn, "r")
    text_f = open(args.text_fn, "r")
    vec_f = open(args.vec_fn, "r")
    fo = open(out_fn, "w")

    # Declaration
    all_unk_counter = defaultdict(int)
    features_idx = 0
    if args.split_fn:
        with open(args.split_fn) as f:
            divided_sent_indices_set = {json.loads(line)["index"] for line in f}
            print("# Indices of divided sentences : {}".format(sorted(divided_sent_indices_set)))
    else:
        divided_sent_indices_set = []

    # Each sentence
    for idx, (line_json, line_text) in tqdm(enumerate(zip(json_f, text_f))):
        features = json.loads(next(vec_f))["features"]
        subwords = [feature["token"] for feature in features]
        vectors = [feature["layers"][0]["values"] for feature in features]

        # The number of subwords exceeds 128
        if features_idx in divided_sent_indices_set:
            features = json.loads(next(vec_f))["features"]
            subwords += [feature["token"] for feature in features]
            vectors += [feature["layers"][0]["values"] for feature in features]

        features_idx += 1

        # Alignment
        words = line_text.rstrip("\n").split(" ")
        sent_indices = alignment.subword_alignment(subword_seq=subwords,
                                                   word_seq=words,
                                                   unk="[UNK]",
                                                   mask="#",
                                                   ignore=["[CLS]", "[SEP]"])

        # Check alignment and count UNK tokens
        unk_counter = alignment.fill_mask(sent_indices, subwords, words, mask="#", unk="[UNK]")
        for key, value in unk_counter.items():
            all_unk_counter[key] += value

        # Make head vectors
        head_vectors = [vectors[word_indices[0]] for word_indices in sent_indices]

        # Print out
        out_dict = json.loads(line_json)
        out_dict["BERT_vecs"] = head_vectors
        assert len(out_dict["BERT_vecs"]) == len(out_dict["tokens"])
        out_line = json.dumps(out_dict)
        print(out_line, file=fo)

    # Save UNK counter
    counter_fn = os.path.join(args.out_dir, "UNK_" + basename)
    with open(counter_fn, "w") as fo:
        for key, value in sorted(all_unk_counter.items(), key=lambda x: -x[1]):
            print("{}: {}".format(key, value), file=fo)
        print("# Save '{}'".format(counter_fn))

    # Close file
    vec_f.close()
    json_f.close()
    text_f.close()
    fo.close()

    print("# Save '{}'".format(out_fn))
    print("Done.")


if __name__ == '__main__':
    args = get_arguments()
    if not os.path.exists(args.json_fn):
        raise FileNotFoundError("'{}' doesn't exist.".format(args.json_fn))
    if not os.path.exists(args.text_fn):
        raise FileNotFoundError("'{}' doesn't exist.".format(args.text_fn))
    if args.split_fn and not os.path.exists(args.split_fn):
        raise FileNotFoundError("'{}' doesn't exist.".format(args.split_fn))
    if not os.path.exists(args.vec_fn):
        raise FileNotFoundError("'{}' doesn't exist.".format(args.vec_fn))

    if not os.path.exists(args.out_dir):
        print("# Make '{}'".format(args.out_dir))
        os.makedirs(args.out_dir)
    elif args.out_dir == os.path.dirname(args.json_fn):
        raise FileExistsError("Out directory '{}' is the same directory as '{}'".format(args.out_dir, args.json_fn))

    main(args)
