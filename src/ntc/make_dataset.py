import argparse
import json
import os.path
from glob import glob

from tqdm import tqdm

import ouchi_corpus
from ouchi_corpus import Sentence
from setting import NONE, DEP, ZERO, CASES


def get_arguments():
    parser = argparse.ArgumentParser(description='make dataset')
    parser.add_argument('in_dir',
                        help='Directory name which contains ntc data set.')
    parser.add_argument('--out_dir', '-o', default='processed_ntc_dataset',
                        help='Directory name to put output data set. (default=processed_ntc_dataset)')
    parser.add_argument('--pre_word_vocab', '-p', default='',
                        help='File path of pre-trained word vocab.')
    parser.add_argument('--ratio', "-r", type=int, default=100)
    args = parser.parse_args()

    return args


class Converter(object):
    """craete word indices"""
    def __init__(self, pre_trained_vocab, mode: str = "all"):
        self.vocab = self.set_vocab(pre_trained_vocab)

    def convert(self, word, pos) -> str:
        """Receive word (str) and pos (str), return word (int)."""
        if word in self.vocab:
            word_idx = self.vocab[word]
        elif pos in self.vocab:
            word_idx = self.vocab[pos]
        else:
            raise RuntimeError("Out of vocabulary.")

        return word_idx

    @staticmethod
    def set_vocab(pre_trained_vocab):
        """Open file of pre-trained vocab and convert to dict format."""
        if not pre_trained_vocab:
            raise ValueError("Please input pre-trained word indices.")

        if not os.path.exists(pre_trained_vocab):
            raise FileNotFoundError("{} doesn't exist.".format(pre_trained_vocab))

        print("\n# Load '{}'".format(pre_trained_vocab))
        vocab = {}

        f = open(pre_trained_vocab)
        for line in f:
            split_line = line.rstrip().split("\t")
            word, idx = split_line[0], split_line[1]
            vocab[word] = idx
        f.close()

        return vocab


class ProcessedSentence(object):
    """Receive <Sentence class> and process to the format for machine learning"""
    def __init__(self, sent: Sentence, converter, doc_fn=None):
        self.unique_id = None
        self.sent = sent
        self.converter = converter
        self.base = [morph.base_form for morph in sent.morphs]
        self.surface = [morph.surface_form for morph in sent.morphs]
        self.pos = [morph.pos for morph in sent.morphs]
        self.bunsetsu = None
        self.tree = None
        self.sent_idx = sent.index
        self.doc_fn = doc_fn
        self.converted_indices = None
        self.pas = None
        self.prev_sents = None
        self.convert_token_indices()
        self.create_bunetsu()
        self.create_tree()
        self.create_pas()

    def convert_token_indices(self):
        morphs = self.sent.morphs
        base = []
        for i, morph in enumerate(morphs):
            # 「サ変名詞 + する」の場合は，モデルへの入力を「サ変名詞 + サ変名詞する」に置換する
            if i != 0 and morph.pos.startswith('動詞') and morphs[i - 1].pos.split('-')[1] == 'サ変名詞':
                base.append(morphs[i - 1].base_form + morph.base_form)
            else:
                base.append(morph.base_form)
        self.converted_indices = [self.converter.convert(word, pos) for word, pos in zip(base, self.pos)]

    def create_bunetsu(self):
        self.bunsetsu = []
        for busnetsu in self.sent.bunsetsus:
            self.bunsetsu.append(1)
            self.bunsetsu += [0] * (len(busnetsu.morphs) - 1)
        assert len(self.bunsetsu) == self.sent.n_morphs

    def create_tree(self):
        self.tree = []
        for busnetsu in self.sent.bunsetsus:
            self.tree.append((busnetsu.index, busnetsu.head))
            self.tree += [None] * (len(busnetsu.morphs) - 1)
        assert len(self.tree) == self.sent.n_morphs

    def create_pas(self):
        self.sent.set_intra_case_dict()
        bunsetsus = self.sent.bunsetsus

        self.pas = []
        for prd in self.sent.prds:
            pas = {"p_id": prd.index,
                   "args": [CASES[NONE]] * self.sent.n_morphs,
                   "types": [NONE] * self.sent.n_morphs}

            for case_name, (case_id, case_type) in prd.intra_case_dict.items():
                self_bunsetsu = bunsetsus[prd.bunsetsu_index]
                for bunsetsu in bunsetsus:
                    for morph in bunsetsu.morphs:
                        if self_bunsetsu.index == bunsetsu.index:
                            continue
                        if case_id == morph.id and case_id != -1:
                            pas["args"][morph.index] = CASES[case_name]  # set case
                            if self_bunsetsu.head == bunsetsu.index:
                                pas["types"][morph.index] = DEP  # set case type
                            elif self_bunsetsu.index == bunsetsu.head:
                                pas["types"][morph.index] = DEP  # set case type
                            else:
                                pas["types"][morph.index] = ZERO  # set case type
            self.pas.append(pas)

    def create_instance(self) -> dict:
        if self.pas:
            instance = {"pas": self.pas,
                        "tokens": self.converted_indices,
                        "sentence id": self.sent_idx,
                        "file name": self.doc_fn,
                        "prev sentences": [sent.converted_indices for sent in self.prev_sents],
                        "surfaces": self.surface,
                        "bases": self.base,
                        "pos": self.pos,
                        "bunsetsu": self.bunsetsu,
                        "tree": self.tree,
                        "unique_id": self.unique_id}
            return instance


def main():
    args = get_arguments()

    # Check exist of in_dir
    if not os.path.exists(args.in_dir):
        raise FileNotFoundError("Input directory doesn't exist: {}".format(args.in_dir))

    # Check exist of dataset
    for name in ["train", "dev", "test"]:
        dir_name = os.path.join(args.in_dir, name)
        if not os.path.exists(dir_name):
            raise FileNotFoundError("Dataset doesn't exist: {}".format(dir_name))

    # make output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print("# Make '{}'".format(args.out_dir))

    converter = Converter(args.pre_word_vocab)

    file_extension = ".jsonl"
    for dataset_name in ["train", "dev", "test"]:
        # Make name of output file
        print("\n# Make {}\n".format(dataset_name + file_extension))
        out_fn = os.path.join(args.out_dir, dataset_name + file_extension)

        # Check exist of out_fn
        if os.path.exists(out_fn):
            raise FileExistsError("Output file already exists.: {}".format(out_fn))

        # Ratio
        data_fns = sorted(glob(os.path.join(args.in_dir, dataset_name, "*")))
        data_fns = data_fns[:int(len(data_fns) * args.ratio / 100)]

        # Make corpus
        corpus = [ouchi_corpus.load_ntc(fn) for fn in data_fns]
        ouchi_corpus.print_stats(corpus)

        # Make output file
        fo = open(out_fn, "w")
        unique_id = 0
        for doc in tqdm(corpus, maxinterval=len(corpus)):
            sentences = [ProcessedSentence(sent, converter, doc.fn) for sent in doc.sents]
            for processed_sent in sentences:
                processed_sent.unique_id = unique_id
                processed_sent.prev_sents = sentences[:processed_sent.sent_idx]
                instance = processed_sent.create_instance()
                if instance:
                    print(json.dumps(instance), file=fo)
                    unique_id += 1

        fo.close()
        print("# Save '{}'".format(out_fn))


if __name__ == '__main__':
    main()
