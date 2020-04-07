import json
import os.path

import jctconv
import numpy as np

from src.setting import CASES, DEP, ZERO, NONE


class Processor(object):
    """Make dataset for each model"""
    def __init__(self):
        self.train = False
        self.ratio = None

    def __call__(self, *args):
        """
        Return the same value as "get_lines".
        """
        return self.get_lines(*args)

    def get_lines(self, *args):
        """
        This is main function to convert dataset to the used form for model.
        Return:
            line
        """
        raise NotImplementedError()

    def save_index(self, *args):
        """Save vocabulary."""
        raise NotImplementedError()

    def save_log(self, *args):
        """Save log information."""
        raise NotImplementedError()


class MatsuProcessor(Processor):
    """Convert word (str) and case (str) to index number (int)."""
    def __init__(self, pre_trained_word_vocab):
        """
        Args:
            pre_trained_word_vocab: File path of pre-trained word vocab. (Str)
        """
        super(MatsuProcessor, self).__init__()
        self.pre_trained_word_vocab = pre_trained_word_vocab
        self.word2index = set_vocab(pre_trained_word_vocab)
        self.case2index = CASES

        self.set_known_position = set()
        self.pas = None
        self.tokens = None
        self.sentence_id = None
        self.file_name = None

    def get_lines(self, xs, ys, sent_idx, doc_fn):
        """See base class."""
        line = None

        surf_seq, base_seq, pos_seq, prd_f_seq, trg_f_seq = xs
        case_seq, case_type_seq = ys

        case_indices = [self.case2index[case] for case in case_seq]

        # make instance
        prd = {"p_id": get_p_id(trg_f_seq),
               "args": case_indices}

        if sent_idx != self.sentence_id or doc_fn != self.file_name:
            # Check overlap
            position = (sent_idx, doc_fn)
            if position in self.set_known_position:
                raise RuntimeError("Can't covert because filename and sentence id is not sorted.")

            # Write
            if self.pas is not None:
                write_dict = {"pas": self.pas, "tokens": self.tokens,
                              "sentence id": self.sentence_id, "file name": self.file_name}
                line = json.dumps(write_dict)

            # New
            self.set_known_position.add(position)
            self.pas = []
            self.tokens = [self.__word2index(base, pos) for base, pos in zip(base_seq, pos_seq)]

        self.pas.append(prd)
        self.sentence_id = sent_idx
        self.file_name = doc_fn

        return line

    def pop(self):
        if not any(i is None for i in [self.pas, self.tokens, self.sentence_id, self.file_name]):
            write_dict = {"pas": self.pas, "tokens": self.tokens,
                          "sentence id": self.sentence_id, "file name": self.file_name}
            line = json.dumps(write_dict)
            self.pas = None
            self.tokens = None
            self.sentence_id = None
            self.file_name = None
            return line
        else:
            raise ValueError("The processor doesn't have all items.")

    def save_index(self, out_dir):
        """See base class."""
        pass

    def save_log(self, in_dir, out_dir):
        """See base class."""
        out_file = os.path.join(out_dir, 'data_logs.txt')
        fo = open(out_file, 'w')
        print('-NTC Dataset for Matsubayashi model-', file=fo)
        print('# Vocab Size', file=fo)
        print(len(self.word2index), file=fo)
        print('\n# Input dir', file=fo)
        print(os.path.abspath(in_dir), file=fo)
        print('\n# Output dir', file=fo)
        print(os.path.abspath(out_dir), file=fo)
        print('\n# pre-trained word.index', file=fo)
        print(os.path.abspath(self.pre_trained_word_vocab), file=fo)
        print('\n# Ratio', file=fo)
        print(self.ratio, file=fo)
        fo.close()

        print('\n# saved "{}"'.format(out_file))

    def __word2index(self, word, pos):
        """Receive word (str) and pos (str), return word (int)."""
        if word in self.word2index:
            word_idx = self.word2index[word]
        elif pos in self.word2index:
            word_idx = self.word2index[pos]
        else:
            raise RuntimeError("Out of vocabulary.")

        return word_idx


class SurfaceProcessor(Processor):
    def __init__(self):
        super(SurfaceProcessor, self).__init__()
        self.set_known_position = set()
        self.sentence_id = None
        self.file_name = None

    def get_lines(self, xs, ys, sent_idx, doc_fn):
        """See base class."""
        surf_seq, *rest = xs

        if sent_idx != self.sentence_id or doc_fn != self.file_name:
            # Check overlap
            position = (sent_idx, doc_fn)
            if position in self.set_known_position:
                raise RuntimeError("Can't covert because filename and sentence id is not sorted.")
            self.sentence_id = sent_idx
            self.file_name = doc_fn
            tokens = " ".join([jctconv.h2z(surf, digit=True, ascii=True) for surf in surf_seq])

            return tokens

    def save_index(self, out_dir):
        """See base class."""
        pass

    def save_log(self, in_dir, out_dir):
        """See base class."""
        out_file = os.path.join(out_dir, 'data_logs.txt')
        fo = open(out_file, 'w')
        print('-NTC Dataset for BERT model-', file=fo)
        print('\n# Input dir', file=fo)
        print(os.path.abspath(in_dir), file=fo)
        print('\n# Output dir', file=fo)
        print(os.path.abspath(out_dir), file=fo)
        print('\n# Ratio', file=fo)
        print(self.ratio, file=fo)
        fo.close()

        print('\n# saved "{}"'.format(out_file))


class CaseProcessor(Processor):
    def __init__(self):
        super(CaseProcessor, self).__init__()
        self.set_known_position = set()
        self.sentence_id = None
        self.file_name = None
        self.case_types = None

    def get_lines(self, xs, ys, sent_idx, doc_fn):
        """See base class."""
        line = None

        case_seq, case_type_seq = ys

        if sent_idx != self.sentence_id or doc_fn != self.file_name:
            # Check overlap
            position = (sent_idx, doc_fn)
            if position in self.set_known_position:
                raise RuntimeError("Can't covert because filename and sentence id is not sorted.")

            # Write
            if self.case_types is not None:
                line = " ".join(self.case_types)

            # New
            self.set_known_position.add(position)
            self.case_types = [NONE] * len(case_type_seq)

        # add case type
        for idx, case_type in enumerate(case_type_seq):
            if case_type == ZERO:
                self.case_types[idx] = ZERO
            elif case_type == DEP and self.case_types[idx] == NONE:
                self.case_types[idx] = DEP

        self.sentence_id = sent_idx
        self.file_name = doc_fn

        return line

    def pop(self):
        if not any(i is None for i in [self.case_types, self.sentence_id, self.file_name]):
            line = " ".join(self.case_types)
            self.sentence_id = None
            self.file_name = None
            self.case_types = None
            return line
        else:
            raise ValueError("The processor doesn't have all items.")

    def save_index(self, out_dir):
        """See base class."""
        pass

    def save_log(self, in_dir, out_dir):
        """See base class."""
        out_file = os.path.join(out_dir, 'data_logs.txt')
        fo = open(out_file, 'w')
        print('-NTC Dataset for BERT model-', file=fo)
        print('\n# Input dir', file=fo)
        print(os.path.abspath(in_dir), file=fo)
        print('\n# Output dir', file=fo)
        print(os.path.abspath(out_dir), file=fo)
        print('\n# Ratio', file=fo)
        print(self.ratio, file=fo)
        fo.close()

        print('\n# saved "{}"'.format(out_file))


class CaseTypeProcessor(Processor):
    def __init__(self):
        super(CaseTypeProcessor, self).__init__()
        self.dep = 0
        self.zero = 0
        self.dep_list = []
        self.zero_list = []

    def get_lines(self, xs, ys, sent_idx, doc_fn):
        """See base class."""
        case_seq, case_type_seq = ys

        # add case type
        case_types = [NONE] * len(case_type_seq)
        for idx, case_type in enumerate(case_type_seq):
            if case_type == ZERO:
                case_types[idx] = ZERO
                self.zero += 1
            elif case_type == DEP:
                case_types[idx] = DEP
                self.dep += 1

        return " ".join(case_types)

    def pop(self):
        self.dep_list.append(self.dep)
        self.zero_list.append(self.zero)
        self.dep = 0
        self.zero = 0

    def save_index(self, out_dir):
        """See base class."""
        pass

    def save_log(self, in_dir, out_dir):
        """See base class."""
        out_file = os.path.join(out_dir, 'data_logs.txt')
        fo = open(out_file, 'w')
        print('-NTC case type-', file=fo)
        print('\n# Input dir', file=fo)
        print(os.path.abspath(in_dir), file=fo)
        print('\n# Output dir', file=fo)
        print(os.path.abspath(out_dir), file=fo)
        print('\n# Ratio', file=fo)
        print(self.ratio, file=fo)
        print('\n# distribution', file=fo)
        print('DEP: {}'.format(self.dep_list), file=fo)
        print('ZERO: {}'.format(self.zero_list), file=fo)
        fo.close()
        print('\n# saved "{}"'.format(out_file))


class MultiSentenceProcessor(Processor):
    """Convert word (str) and case (str) to index number (int)."""
    def __init__(self, pre_trained_word_vocab):
        """
        Args:
            pre_trained_word_vocab: File path of pre-trained word vocab. (Str)
        """
        super(MultiSentenceProcessor, self).__init__()
        self.pre_trained_word_vocab = pre_trained_word_vocab
        self.word2index = set_vocab(pre_trained_word_vocab)
        self.case2index = CASES

        self.set_known_position = set()
        self.pas = None
        self.tokens = None
        self.sentence_id = None
        self.file_name = None

        self.prev_sentences = None

    def get_lines(self, xs, ys, sent_idx, doc_fn, prev_sentences):
        """See base class."""
        line = None

        surf_seq, base_seq,  pos_seq, prd_f_seq, trg_f_seq = xs
        case_seq, case_type_seq = ys

        case_indices = [self.case2index[case] for case in case_seq]

        # make instance
        prd = {"p_id": get_p_id(trg_f_seq),
               "args": case_indices}

        if sent_idx != self.sentence_id or doc_fn != self.file_name:
            # Check overlap
            position = (sent_idx, doc_fn)
            if position in self.set_known_position:
                raise RuntimeError("Can't covert because filename and sentence id is not sorted.")

            # Make output line
            if self.pas is not None:
                write_dict = {"pas": self.pas, "tokens": self.tokens,
                              "sentence id": self.sentence_id, "file name": self.file_name,
                              "prev sentences": self.prev_sentences}
                line = json.dumps(write_dict)

            # New
            self.prev_sentences = [[self.__word2index(word, pos) for word, pos in zip(words, poss)]
                                   for words, poss in prev_sentences]
            self.tokens = [self.__word2index(base, pos) for base, pos in zip(base_seq, pos_seq)]
            self.pas = []
            self.set_known_position.add(position)

        self.pas.append(prd)
        self.sentence_id = sent_idx
        self.file_name = doc_fn

        return line

    def pop(self):
        if not any(i is None for i in [self.pas, self.tokens, self.sentence_id, self.file_name]):
            write_dict = {"pas": self.pas, "tokens": self.tokens,
                          "sentence id": self.sentence_id, "file name": self.file_name,
                          "prev sentences": self.prev_sentences}
            line = json.dumps(write_dict)
            self.pas = None
            self.tokens = None
            self.sentence_id = None
            self.file_name = None
            self.prev_sentences = None
            return line
        else:
            raise ValueError("The processor doesn't have all items.")

    def save_index(self, out_dir):
        """See base class."""
        pass

    def save_log(self, in_dir, out_dir):
        """See base class."""
        out_file = os.path.join(out_dir, 'data_logs.txt')
        fo = open(out_file, 'w')
        print('-NTC Dataset for Matsubayashi model-', file=fo)
        print('# Vocab Size', file=fo)
        print(len(self.word2index), file=fo)
        print('\n# Input dir', file=fo)
        print(os.path.abspath(in_dir), file=fo)
        print('\n# Output dir', file=fo)
        print(os.path.abspath(out_dir), file=fo)
        print('\n# pre-trained word.index', file=fo)
        print(os.path.abspath(self.pre_trained_word_vocab), file=fo)
        print('\n# Ratio', file=fo)
        print(self.ratio, file=fo)
        fo.close()

        print('\n# saved "{}"'.format(out_file))

    def __word2index(self, word, pos):
        """Receive word (str) and pos (str), return word (int)."""
        if word in self.word2index:
            word_idx = self.word2index[word]
        elif pos in self.word2index:
            word_idx = self.word2index[pos]
        else:
            raise RuntimeError("Out of vocabulary.")

        return word_idx


def get_p_id(trg):
    assert sum(trg) == 1
    assert max(trg) == 1
    assert min(trg) == 0

    return int(np.argmax(trg))


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
