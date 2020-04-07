import io
import json
import os
import re
import sys
from collections import deque
from os import path

import numpy as np
import torch
from tqdm import tqdm

PAD = 0


def set_log_file(log_dir, tag, model_id):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    log_fn = 'stdout-' + tag + '-' + model_id + ".txt"
    fd = os.open(path.join(log_dir, log_fn), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def load_dataset(data_path, data_rate):
    print("# Load: {}".format(data_path))
    data = open(data_path).readlines()
    len_data = len(data)
    data = data[:int(len_data * data_rate / 100)]

    return [json.loads(line.strip()) for line in tqdm(data) if line.strip()]


def pretrained_word_vecs(data_path, index_file_name, embed_dim: int = 256):
    # Loading Word Index
    wIdx = {}
    lexicon_size = 0
    wIdxData = open(data_path + index_file_name)
    for line in wIdxData:
        w, idx = line.rstrip().split("\t")
        idx = int(idx)
        wIdx[w] = idx
        if lexicon_size < idx:
            lexicon_size = idx + 1
    wIdxData.close()

    # Loading Word Vector
    vocab_size = 0
    wVecData = open("{0}/lemma-oov_vec-{1}-jawiki20160901-filtered.txt".format(data_path, embed_dim))
    matrix = np.random.uniform(-0.05, 0.05, (lexicon_size, embed_dim))
    for line in wVecData:
        values = line.rstrip().split(" ")
        word = values[0]
        if word in wIdx:
            matrix[wIdx[word]] = np.asarray(values[1:], dtype='float32')
            vocab_size += 1
    wVecData.close()

    # pad
    matrix[0] = np.zeros(embed_dim)

    print("vocab size: ", vocab_size)

    return torch.from_numpy(matrix).float()

