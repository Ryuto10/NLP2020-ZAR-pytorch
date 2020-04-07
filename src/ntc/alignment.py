import copy
import re
from collections import defaultdict
from typing import List


def fill_mask(alignment_indices: List[List[int]], subword_seq: List[str], word_seq: List[str], mask="", unk=""):
    special_tokens = {".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"}
    set_unk = {i for i, subword in enumerate(subword_seq) if subword == unk}

    if len(set_unk) == 0:
        return {}

    # Make match object
    combined_sentence = get_combined_sentence(alignment_indices, subword_seq, mask)
    for token in special_tokens:
        combined_sentence.replace(token, "\\" + token)
    combined_sentence = combined_sentence.replace(unk, "(.+)")
    original_sentence = " ".join(word_seq)
    m = re.match(combined_sentence, original_sentence)
    assert m.lastindex == len(set_unk)

    # Check alignment
    unk_list = [m.group(m.lastindex - i) for i in range(m.lastindex)]
    words = []
    for indices in alignment_indices:
        word = "".join(unk_list.pop() if idx in set_unk else subword_seq[idx].lstrip(mask) for idx in indices)
        words.append(word)
    fill_mask_sentence = " ".join(words)
    assert fill_mask_sentence == original_sentence

    # Count match object
    unk_counter = defaultdict(int)
    for i in range(m.lastindex):
        unk_counter[m.group(i + 1)] += 1

    return unk_counter


def get_combined_sentence(alignment_indices: List[List[int]], subword_seq: List[str], mask=""):
    combined = " ".join("".join(subword_seq[i].lstrip(mask) for i in indices)
                        for indices in alignment_indices)
    return combined


def subword_alignment(subword_seq: List[str], word_seq: List[str], unk=None, mask=None, ignore=None):
    """
    Args:
        subword_seq: Subword tokens (List of string)
        word_seq: Word tokens (List of string)
        unk: UNK token (String)
        mask: Mask token (String)
        ignore: Ignore tokens (List of string)
    Return:
        alignment_indices

    Example:
        Args:
            sub_seq = ["[CLS]", "a", "b", "d", "[UNK]", "[UNK]", "g", "g", "[UNK]", "[UNK]", "ij", "ij", "kl", "[SEP]"]
            word_seq = ["ab", "defefg", "g", "h", "hij", "ijkl"]
            unk = "[UNK]"
            ignore = ["[CLS]", "[SEP]"]

        Return:
            alignment_indices = [[1, 2], [3, 4, 5, 6], [7], [8], [9, 10], [11, 12]]
    """
    if type(unk) != str:
        raise ValueError("Unsupported type: {}".format(unk))

    if type(mask) != str:
        raise ValueError("Unsupported type: {}".format(mask))

    if type(ignore) == List:
        raise ValueError("Unsupported type: {}".format(ignore))

    start_subword_idx = 0
    start_word_idx = 0
    start_alignment_indices = []
    start_unk_buffer_indices = []

    unk_flag = False

    for _ in range(5):
        # Set the variables in the previous loop
        alignment_indices = copy.deepcopy(start_alignment_indices)
        unk_buffer_indices = copy.deepcopy(start_unk_buffer_indices)
        word_idx = start_word_idx
        shift_idx = start_subword_idx

        finish = True
        buffer = None
        buffer_indices = []

        for subword_idx, subword in enumerate(subword_seq[start_subword_idx:]):
            # Adjustment of index shifted by slice
            subword_idx += shift_idx

            # ignore token
            if ignore and subword in ignore:
                continue

            # UNK token
            if unk_flag:
                # Reached end of word index
                if word_idx == len(word_seq):
                    unk_buffer_indices.append(subword_idx)
                    alignment_indices.append(unk_buffer_indices)
                    unk_buffer_indices = []
                    break

                # Subword matches head of next word
                if _is_head(word_seq[word_idx], subword_seq, subword_idx, mask, unk):
                    start_subword_idx = subword_idx + 1
                    start_word_idx = word_idx
                    start_alignment_indices = copy.deepcopy(alignment_indices)
                    start_unk_buffer_indices = copy.deepcopy(unk_buffer_indices)
                    start_unk_buffer_indices.append(subword_idx)

                    alignment_indices.append(unk_buffer_indices)
                    unk_buffer_indices = []
                    unk_flag = False

                # Don't match
                else:
                    unk_buffer_indices.append(subword_idx)
                    continue

            elif subword == unk:
                unk_flag = True
                unk_buffer_indices = copy.deepcopy(buffer_indices)
                unk_buffer_indices.append(subword_idx)
                buffer = None
                buffer_indices = []
                word_idx += 1
                start_subword_idx = subword_idx + 1
                continue

            # New buffer
            if buffer is None:
                buffer = subword
                buffer_indices.append(subword_idx)

            # Add to buffer
            else:
                subword = _strip_mask(subword, mask)
                buffer += subword
                buffer_indices.append(subword_idx)

            # Bad result
            word = word_seq[word_idx]
            head_idx = buffer_indices[0]
            if word_idx == len(word_seq) or not _is_head(word, subword_seq, head_idx, mask, unk, allow_unk=True):
                finish = False
                break

            # Buffer equals the word
            elif buffer == word_seq[word_idx]:
                alignment_indices.append(buffer_indices)
                buffer = None
                buffer_indices = []
                word_idx += 1

        # UNK buffer is not empty
        if unk_buffer_indices:
            unk_flag = False
            start_word_idx = word_idx
            alignment_indices.append([unk_buffer_indices[0]])
            start_alignment_indices = alignment_indices
            start_unk_buffer_indices = []

        # Check alignment_indices
        elif finish and _check(alignment_indices, subword_seq, word_seq, unk, mask):
            return alignment_indices

        else:
            unk_flag = True

    raise RuntimeError("These two sequences can not be aligned.:\n\t{}\n\t{}".format(subword_seq, word_seq))


def _check(alignment_indices: List[List[int]], subword_seq: List[str], word_seq: List[str], unk: str, mask: str):
    if len(alignment_indices) != len(word_seq):
        return False

    for word, indices in zip(word_seq, alignment_indices):
        head_subword = subword_seq[indices[0]]
        if head_subword != unk and not word.startswith(head_subword):
            return False

    for word, indices in zip(word_seq, alignment_indices):
        tail_subword = subword_seq[indices[-1]].lstrip(mask)
        if tail_subword != unk and not word.endswith(tail_subword):
            return False

    return True


def _is_head(word, subword_seq, head_idx, mask, unk, allow_unk=False):
    subword = subword_seq[head_idx]

    if subword == unk:
        if allow_unk:
            return True
        else:
            return False

    # Word consist of one subword. (Subword equals word.)
    if word == subword:
        return True

    # Words consists of two or more subwords.
    if head_idx + 1 < len(subword_seq):
        next_subword = _strip_mask(subword_seq[head_idx + 1], mask)
        if word.startswith(subword + next_subword):
            return True

    return False


def _strip_mask(subword, mask):
    # Strip mask token
    if mask:
        subword = subword.lstrip(mask)

    return subword
