# coding: utf-8
import argparse
import os
import re
from os import path

import torch

from batchiterator import multi_sent_test_batch_generator, multi_sent_test_end2end_single_seq_instance
from decode import decode
from models import MultiSentenceE2EStackedBiRNN
from utils import load_dataset, pretrained_word_vecs, set_log_file, parse_train_log


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--tag', type=str, default="dev",
                        help='type of evaluation data split')
    parser.add_argument('--train_log', type=path.abspath,
                        help='Path to log of training.')
    parser.add_argument('--multi_sent_data', type=str,
                        help='Path to multi sentence file')

    return parser


def parse_model_id(model_id):
    prev_n = int(re.search(r"multi_sent([0-9]+)-", model_id).group(1))
    hidden_dim = int(re.search(r"_h([0-9]+)_", model_id).group(1))
    n_layers = int(re.search(r"_layer([0-9]+)_", model_id).group(1))
    dropout = float(re.search(r"_d([0-9.]+)_", model_id).group(1))

    return prev_n, hidden_dim, n_layers, dropout


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    out_dir = path.join(args.data_path, "result")
    if not path.exists(out_dir):
        os.mkdir(out_dir)
        print("# Make Directory: {}".join(out_dir))

    model_id, _, threshold = parse_train_log(args.train_log)
    model_id = model_id + "-" + "-".join(str(i) for i in threshold)
    prev_n, hidden_dim, n_layers, dropout = parse_model_id(model_id)
    print(model_id)

    log_dir = path.dirname(args.train_log)
    set_log_file(args, args.tag, model_id)

    data = load_dataset(args.multi_sent_data + "/test.json", 100)
    data = list(multi_sent_test_end2end_single_seq_instance(data, multi_sent_test_batch_generator, prev_n))

    word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt")
    model = MultiSentenceE2EStackedBiRNN(hidden_dim=hidden_dim,
                                         n_layers=n_layers,
                                         out_dim=4,
                                         embedding_matrix=word_embedding_matrix,
                                         fixed_word_vec=True,
                                         dropout=dropout)
    model.load_state_dict(torch.load(args.model_file))

    if torch.cuda.is_available():
        model = model.cuda()

    decode(out_dir, data, args.tag, model, model_id, threshold)


if __name__ == '__main__':
    main()
