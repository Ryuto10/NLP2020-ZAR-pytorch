import argparse
import json
from itertools import groupby
from os import path
from typing import List, Optional

from itertools import islice
from toolz import sliding_window
from tqdm import tqdm


# -- Create Jsonl File --
#
# Per line: {"tokens": tokens, "labels": labels, "pos_tags": pos-tags, "tree": tree}
#
#   * tokens: List of surfaces (length = number of words).
#   * labels: [{"verb_idx": verb_idx, "tags": tags}, ...]
#       - verb_idx: Index indicating the position of the predicate.
#       - tags: List of predicate arguments (BIO format).
#   * pos-tags: List of pos-tagging (length = number of words).
#   * tree: Syntax tree (length = number of words).


def create_parser():
    """
    'label_file'に引数を与える場合：
        in_file: 元となるjsonlファイル (このスクリプトで作成済みのファイル)
        label_file: 'in_file'のラベルのみが書き込まれているCoNLL形式のファイル
    """
    parser = argparse.ArgumentParser(description='Create Semantic Role Labeling Dataset as jsonl.')
    parser.add_argument('--in_file', type=path.abspath, help="Path to input file. (CoNLL format file or jsonl file.")
    parser.add_argument('--out_file', type=path.abspath, help="Path to output file.")
    parser.add_argument('--label_file', type=path.abspath, default=None, help="Path to label file. (CoNLL format file)")
    parser.add_argument('--data_type', type=str, default=None, help="Choose from 'CoNLL-05' or 'CoNLL-12'")
    parser.add_argument('--prop', action='store_true')

    return parser


def process_span_annotations_for_word(annotations: List[str]):
    """
    example
    ----------
        [(A0, *), (A1), *, (AM-LOC*, *, *), *] -> [B-A0, I-A0, B-A1, O, B-AM-LOC, I-AM-LOC, I-AM-LOC, O]

    Parameters
    ----------
    annotations: ``List[str]``
        A list of labels to compute BIO tags for.
    """
    bio_labels = []
    current_label = None
    for annotation in annotations:
        # strip all bracketing information to
        # get the actual propbank label.
        label = annotation.strip("()*")

        if "(" in annotation:
            assert current_label is None
            current_label = label
            bio_label = "B-" + label
            bio_labels.append(bio_label)
        elif current_label is not None:
            bio_label = "I-" + current_label
            bio_labels.append(bio_label)
        else:
            bio_labels.append("O")
        if ")" in annotation:
            current_label = None
    assert len(bio_labels) == len(annotations)

    return bio_labels


def test_process_span_annotations_for_word():
    instances = [(["(A0", "*)", "(A1)", "*", "(AM-LOC*", "*", "*)", "*"],
                 ["B-A0", "I-A0", "B-A1", "O", "B-AM-LOC", "I-AM-LOC", "I-AM-LOC", "O"]),
                 (["(AM-LOC*", "*", "*", "*", "*", "*", "*)", "*", "*"],
                  ["B-AM-LOC", "I-AM-LOC", "I-AM-LOC", "I-AM-LOC", "I-AM-LOC", "I-AM-LOC", "I-AM-LOC", "O", "O"]),
                 (["*", "*", "*"], ["O", "O", "O"]),
                 (["*", "(A0*", "*)", "*"], ["O", "B-A0", "I-A0", "O"]),
                 (["*", "*", "(A0)"], ["O", "O", "B-A0"])]
    for raw, processed in instances:
        assert process_span_annotations_for_word(raw) == processed


def get_predicate_span(tags):
    indices = []
    for idx, tag in enumerate(tags):
        label = tag.replace("B-", "").replace("I-", "")
        if label == "V":
            indices.append(idx)

    return indices


def test_get_predicate_span():
    instance = [(["B-V", "I-V", "O"], [0, 1]),
                (["O", "B-V", "O"], [1]),
                (["O", "B-A1", "O", "B-A0", "I-A0", "B-V", "I-V", "O"], [5, 6])]
    for x, y in instance:
        assert get_predicate_span(x) == y


def create_conll05_dataset(in_file: str, out_file: str):
    """ CoNLL-2005 -> jsonl

        CoNLL-2005 columns (paper):
            * 0: words
            - 1: PoS tags
            - 2: base chunks
            - 3: clauses
            - 4: full syntactic tree
            - 5: named entities
            * 6: marks target verbs
            * 7~: each predicate arguments

        CoNLL-2005 columns (real):
            0: words
            5: marks target verbs
            6~: each predicate arguments
    """
    TOKEN_IDX = 0
    POS_IDX = 1
    SYNTAX_IDX = 2
    VERB_IDX = 4 if "test" in in_file else 5
    TAG_IDX = 5 if "test" in in_file else 6
    with open(in_file) as fi, open(out_file, "w") as fo:
        for value, chunk in tqdm(groupby(fi, key=lambda x: bool(x.strip()))):
            if not value:
                continue
            lines = [line.rstrip("\n").split() for line in chunk]
            verb_indices = [idx for idx, line in enumerate(lines) if line[VERB_IDX] != "-"]

            # Check
            assert len(lines[0]) == TAG_IDX + len(verb_indices)
            assert all(len(pair[0]) == len(pair[1]) for pair in sliding_window(2, lines))

            tokens = [line[TOKEN_IDX] for line in lines]
            pos_tags = [line[POS_IDX] for line in lines]
            tree = [line[SYNTAX_IDX] for line in lines]
            labels = []
            for n, verb_idx in enumerate(verb_indices):
                tags = process_span_annotations_for_word([line[TAG_IDX + n] for line in lines])
                predicate_span = get_predicate_span(tags)
                assert verb_idx in predicate_span
                labels.append({"verb_span": predicate_span, "tags": tags})

            # Write
            json_line = json.dumps({"tokens": tokens, "labels": labels, "pos_tags": pos_tags, "tree": tree})
            print(json_line, file=fo)


def create_conll05_dataset_with_prop(in_file: str, out_file: str):
    """ CoNLL-2005 -> jsonl

        CoNLL-2005 columns (paper):
            * 0: words
            - 1: PoS tags
            - 2: base chunks
            - 3: clauses
            - 4: full syntactic tree
            - 5: named entities
            * 6: marks target verbs
            * 7~: each predicate arguments

        CoNLL-2005 columns (real):
            0: words
            5: marks target verbs
            6~: each predicate arguments
    """
    TOKEN_IDX = 0
    VERB_IDX = 4 if "test" in in_file else 5
    TAG_IDX = 5 if "test" in in_file else 6
    with open(in_file) as fi, open(out_file, "w") as fo:
        for value, chunk in tqdm(groupby(fi, key=lambda x: bool(x.strip()))):
            if not value:
                continue
            lines = [line.rstrip("\n").split() for line in chunk]
            verb_indices = [idx for idx, line in enumerate(lines) if line[VERB_IDX] != "-"]
            lemma = [line[VERB_IDX] for line in lines]

            # Check
            assert len(lines[0]) == TAG_IDX + len(verb_indices)
            assert all(len(pair[0]) == len(pair[1]) for pair in sliding_window(2, lines))

            for n, verb_idx in enumerate(verb_indices):
                tags = process_span_annotations_for_word([line[TAG_IDX + n] for line in lines])
                for idx, tag in enumerate(tags):
                    prd = lemma[idx] if idx == verb_idx else "-"
                    print("{}\t{}".format(prd, tag), file=fo)
                print("", file=fo)


def create_conll12_dataset(in_file: str, out_file: str):
    """ CoNLL-2012 -> jsonl

        CoNLL-2012 columns (paper):
            - 1: Document ID
            - 2: part number
            - 3: Word number
            * 4: Word
            * 5: Part of Speech
            * 6: Parse bit
            - 7: Lemma
            * 8: Predicate Frameset ID
            - 9: Word sense
            - 10: Speaker/Author
            - 11: Named Entities
            * 12:N: Predicate Arguments
            - N: Co-reference
    """
    TOKEN_IDX = 3
    POS_IDX = 4
    SYNTAX_IDX = 5
    VERB_IDX = 7
    TAG_IDX = 11
    MIN_LEN = 12
    with open(in_file) as fi, open(out_file, "w") as fo:
        _ = next(fi)
        for value, chunk in tqdm(groupby(fi, key=lambda x: bool(x.strip()))):
            if not value:
                continue
            lines = [line.rstrip("\n").split() for line in chunk]
            if lines[0][0].startswith("#end"):
                continue
            verb_indices = [idx for idx, line in enumerate(lines) if line[VERB_IDX] != "-"]

            # Check
            assert len(lines[0]) == MIN_LEN + len(verb_indices)
            assert all(len(pair[0]) == len(pair[1]) for pair in sliding_window(2, lines))

            tokens = [line[TOKEN_IDX] for line in lines]
            pos_tags = [line[POS_IDX] for line in lines]
            tree = [line[SYNTAX_IDX] for line in lines]
            labels = []
            for n, verb_idx in enumerate(verb_indices):
                tags = process_span_annotations_for_word([line[TAG_IDX + n] for line in lines])
                predicate_span = get_predicate_span(tags)
                assert verb_idx in predicate_span
                labels.append({"verb_span": predicate_span, "tags": tags})

            # Write
            json_line = json.dumps({"tokens": tokens, "labels": labels, "pos_tags": pos_tags, "tree": tree})
            print(json_line, file=fo)


def merge_conll05_dataset(in_file: str, label_file: str, out_file: str):
    """ jsonl + CoNLL -> jsonl

    Columns of CoNLL-2005 label file:
        0: marks target verbs
        1: each predicate arguments
    """
    VERB_IDX = 0
    TAG_IDX = 1
    with open(in_file) as f_jsonl, open(label_file) as f_conll, open(out_file, "w") as fo:
        for value, chunk in tqdm(groupby(f_conll, key=lambda x: bool(x.strip()))):
            if not value:
                continue
            lines = [line.rstrip("\n").split() for line in chunk]
            verb_indices = [idx for idx, line in enumerate(lines) if line[VERB_IDX] != "-"]

            # read jsonl
            sent_instance = json.loads(next(f_jsonl))

            # Check
            assert len(lines) == len(sent_instance["tokens"])
            assert len(lines[0]) == TAG_IDX + len(verb_indices)
            assert all(len(pair[0]) == len(pair[1]) for pair in sliding_window(2, lines))

            labels = []
            for n, verb_idx in enumerate(verb_indices):
                tags = process_span_annotations_for_word([line[TAG_IDX + n] for line in lines])
                predicate_span = get_predicate_span(tags)
                labels.append({"verb_span": predicate_span, "tags": tags})
            sent_instance["labels"] = labels

            # Write
            json_line = json.dumps(sent_instance)
            print(json_line, file=fo)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Test
    test_process_span_annotations_for_word()
    test_get_predicate_span()

    # Merge
    if args.label_file:
        if not args.in_file.endswith(".jsonl"):
            raise RuntimeError("\n\tThe expected 'in_file' is a jsonl file."
                               "\n\tBut '{}' is not a jsonl file.".format(args.in_file))

        if args.data_type == "CoNLL-05":
            print("Merge CoNLL-05:")
            merge_conll05_dataset(in_file=args.in_file, label_file=args.label_file, out_file=args.out_file)
        else:
            raise RuntimeError("Please fill in the valid 'data_type' (Choose from 'CoNLL-05' or 'CoNLL-12')")
    # Create
    else:
        if args.data_type == "CoNLL-05":
            if args.prop:
                create_conll05_dataset_with_prop(in_file=args.in_file, out_file=args.out_file)
            else:
                print("Create CoNLL-05:")
                create_conll05_dataset(in_file=args.in_file, out_file=args.out_file)
        elif args.data_type == "CoNLL-12":
            print("Create CoNLL-12:")
            create_conll12_dataset(in_file=args.in_file, out_file=args.out_file)
        else:
            raise RuntimeError("Please fill in the valid 'data_type' (Choose from 'CoNLL-05' or 'CoNLL-12')")

    print("# Save: {}".format(args.out_file))
    print("# --- contents ---")
    with open(args.out_file) as fi:
        for line in islice(fi, 5):
            print("> {} ... {}".format(line[:50], line[-50:]) if len(line) > 100 else line, end="")


if __name__ == "__main__":
    main()


