# coding: utf-8
import argparse
import json
import os
import random
from datetime import datetime
from os import path
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from batchiterator import NtcBucketIterator
from decode import decode
from evaluation import packed_evaluate_multiclass_without_none
from log import StandardLogger, write_args_log
from models import PackedE2EStackedBiRNN
from utils import load_dataset, pretrained_word_vecs, set_log_file

random.seed(2020)
np.random.seed(2020)
# MAX_SENTENCE_LENGTH = 90
MAX_SENTENCE_LENGTH = 10000
BERT_DIM = 768


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', type=str, default=None, required=True)
    parser.add_argument('--dev', type=str, default=None, required=True)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--pseudo', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='result')
    parser.add_argument('--wiki_embed_dir', type=path.abspath, default=None)
    parser.add_argument('--train_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--dev_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--test_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--pseudo_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")

    # Training Option
    parser.add_argument('--train_method', type=str, default="concat",
                        help="Choose from 'concat' or 'pre-train'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=100,
                        help='data size (%)')
    parser.add_argument('--epoch', dest='max_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--loss_stop', action='store_true')
    parser.add_argument('--wiki', action='store_true')
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--multi_predicate', action='store_true')
    parser.add_argument('--zero_drop', action='store_true')
    parser.add_argument('--mapping_pseudo_train', type=path.abspath, default=None)
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--load_cpu', action='store_true')
    parser.add_argument('--half_checkpoint', action='store_true')
    parser.add_argument('--epoch_shuffle', action='store_true')
    parser.add_argument('--comment', type=str, default="")

    # Hyper Parameter
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pseudo_lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='unit vector size in rnn')
    parser.add_argument('--n_layers', type=int, default=10,
                        help='the number of hidden layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate of rnn unit')
    parser.add_argument('--embed_dropout', type=float, default=0.0,
                        help='dropout rate of embeddings')
    parser.add_argument('--tune-word-vec', dest='fixed_word_vec', action='store_false',
                        help='do not re-train word vec')

    parser.set_defaults(fixed_word_vec=True)

    return parser


class BertVecHolder(object):
    def __init__(self, train_json: str, train_hdf5: str, pseudo_json: str, pseudo_hdf5: str, data_size: float):
        self.train_json = train_json
        self.train_hdf5 = train_hdf5
        self.pseudo_json_files = sorted(glob(pseudo_json + ".seed*"))
        self.pseudo_hdf5_files = sorted(glob(pseudo_hdf5 + ".seed*"))
        self.data_size = data_size
        self.current_index = 0
        self.max_length = len(self.pseudo_json_files)
        self.indices = list(range(self.max_length))

        assert self.max_length == len(self.pseudo_hdf5_files)
        print("# Number of pseudo files: {}".format(self.max_length), flush=True)
        for jf, hf in zip(self.pseudo_json_files, self.pseudo_hdf5_files):
            print("\t{}, {}".format(path.basename(jf), path.basename(hf)), flush=True)

        self.data_train = load_dataset(self.train_json, self.data_size)
        random.shuffle(self.indices)

    def create_dataset(self):
        if self.current_index >= self.max_length:
            random.shuffle(self.indices)
            self.current_index = 0
        pseudo_json_file = self.pseudo_json_files[self.current_index]
        pseudo_hdf5_file = self.pseudo_hdf5_files[self.current_index]
        basename = path.basename(pseudo_json_file).replace(".jsonl", "")

        print("# Load: {}\n\tCurrent Index: {}".format(basename, self.current_index), flush=True)
        assert basename == path.basename(pseudo_hdf5_file).replace(".hdf5", "")

        data_pseudo = load_dataset(pseudo_json_file, self.data_size)
        dataset = self.data_train + data_pseudo
        self.current_index += 1

        return dataset, pseudo_hdf5_file


def train(out_dir, data_train, data_dev, model, model_id, epoch, lr_start, lr_min,
          half_checkpoint, bert_vec_holder, train_type=""):
    len_train = len(data_train)
    len_dev = len(data_dev)

    early_stopping_thres = 4
    early_stopping_count = 0
    best_performance = -1.0
    best_epoch = 0
    best_thres = None
    best_lr = lr_start
    lr = lr_start
    lr_reduce_factor = 0.5
    lr_epsilon = lr_min * 1e-4

    loss_function = nn.NLLLoss(ignore_index=4)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)
    losses = []

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    best_f1_history = []

    if path.exists(out_dir + "/pretrained_model-" + model_id + ".h5"):
        print("-" * 10 + " Original data " + "-" * 10, flush=True)

        # Load model
        print("# Load pre-trained model", flush=True)
        model.load_state_dict(torch.load(out_dir + "/pretrained_model-" + model_id + ".h5"))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Save model
        torch.save(model.state_dict(), out_dir + "/model-".format(train_type) + model_id + ".h5")

        # Get Start F1
        print("# First test", flush=True)
        data_dev.create_batches()
        model.eval()
        thres, obj_score, num_test_batch_instance = packed_evaluate_multiclass_without_none(model, data_dev, len_dev,
                                                                                            labels, thres_lists)
        best_thres = thres
        best_performance = obj_score * 100
        print("## Init F1: {}".format(best_performance), flush=True)
    else:
        print("-" * 10 + " Pseudo data " + "-" * 10, flush=True)

    for ep in range(epoch):
        total_loss = torch.Tensor([0])
        early_stopping_count += 1

        print(model_id, 'epoch {0}'.format(ep + 1), flush=True)

        if bert_vec_holder is not None and ep != 0:
            dataset, pseudo_hdf5 = bert_vec_holder.create_dataset()
            data_train.reset_dataset_with_pseudo(dataset, pseudo_hdf5)

        print('# Train...', flush=True)
        data_train.create_batches()
        model.train()
        for n, (xss, yss) in tqdm(enumerate(data_train), total=len_train, mininterval=5):
            if xss[0].size(1) > MAX_SENTENCE_LENGTH:
                continue

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = [Variable(ys).cuda() for ys in yss]
            else:
                yss = [Variable(ys) for ys in yss]

            scores = model(xss)

            loss = 0
            for i in range(len(yss)):
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()

            if half_checkpoint and n == int(len_train / 2):
                print("## loss:", total_loss[0], "lr:", lr)
                losses.append(total_loss)
                print("", flush=True)
                print('# Test... (half point)', flush=True)
                data_dev.create_batches()
                model.eval()
                thres, obj_score, num_test_batch_instance = packed_evaluate_multiclass_without_none(model, data_dev,
                                                                                                    len_dev,
                                                                                                    labels, thres_lists)
                f = obj_score * 100
                if f > best_performance:
                    best_performance = f
                    early_stopping_count = 0
                    best_epoch = ep + 1
                    best_thres = thres
                    best_lr = lr
                    print("## save model", flush=True)
                    torch.save(model.state_dict(), out_dir + "/{}model-".format(train_type) + model_id + ".h5")

                    best_f1_history.append((best_performance, best_epoch))

                elif early_stopping_count >= early_stopping_thres:
                    # break
                    if lr > lr_min + lr_epsilon:
                        new_lr = lr * lr_reduce_factor
                        lr = max(new_lr, lr_min)
                        print("load model: epoch{0}".format(best_epoch), flush=True)
                        model.load_state_dict(torch.load(out_dir + "/{}model-".format(train_type) + model_id + ".h5"))
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                        early_stopping_count = 0
                    else:
                        break
                print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
                      "f:", best_performance)
                total_loss = torch.Tensor([0])
                early_stopping_count += 1

                print(model_id, 'epoch {0}'.format(ep + 1), flush=True)

                print('# Train...', flush=True)
                data_train.create_batches()
                model.train()

        print("## loss:", total_loss[0], "lr:", lr)
        losses.append(total_loss)
        print("", flush=True)
        print('# Test...', flush=True)

        data_dev.create_batches()
        model.eval()
        thres, obj_score, num_test_batch_instance = packed_evaluate_multiclass_without_none(model, data_dev, len_dev,
                                                                                            labels, thres_lists)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            print("## save model", flush=True)
            torch.save(model.state_dict(), out_dir + "/{}model-".format(train_type) + model_id + ".h5")

            best_f1_history.append((best_performance, best_epoch))

        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                print("load model: epoch{0}".format(best_epoch), flush=True)
                model.load_state_dict(torch.load(out_dir + "/{}model-".format(train_type) + model_id + ".h5"))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
              "f:", best_performance)

    print(model_id, "\tbest in epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
          "f:", best_performance)

    print("Update History: {}".format(best_f1_history), flush=True)

    return best_thres


def extract_name(file):
    name = path.basename(file)
    name, _ = name.split(".", 1)

    return name


def create_model_id(args):
    model_id = datetime.today().strftime("%m%d%H%M")
    model_id += "-" + extract_name(args.pseudo) if args.pseudo else ""
    model_id += "-" + extract_name(args.test) if args.test else ""
    model_id += "-" + args.train_method
    model_id += "-wiki" if args.wiki else ""
    model_id += "-bert" if args.bert else ""
    model_id += "-mp" if args.multi_predicate else ""
    model_id += "-zero_drop" if args.zero_drop else ""
    model_id += "-no_mask" if args.mapping_pseudo_train else ""
    model_id += "-loss_stop" if args.loss_stop else ""
    model_id += "-half_point" if args.half_checkpoint else ""
    model_id += "-lr" + str(args.lr)
    model_id += "-plr" + str(args.pseudo_lr) if args.train_method == "pre-train" else ""
    model_id += "-embdrop" + str(args.embed_dropout)
    model_id += "-seed" + str(args.seed)
    model_id += "-" + args.comment if args.comment else ""

    return model_id


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    if args.pseudo == "None":
        args.pseudo = None

    if not path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print("# Make directory: {}".format(args.out_dir))

    # Log
    model_id = create_model_id(args)
    log_dir = path.join(args.out_dir, model_id)
    if path.exists(log_dir):
        raise FileExistsError("'{}' Already exists.".format(log_dir))
    os.mkdir(log_dir)
    print(log_dir)
    set_log_file(log_dir, "train", model_id)
    log = StandardLogger(path.join(log_dir, "log-" + model_id + ".txt"))
    log.write(args=args, comment=model_id)
    write_args_log(args, path.join(log_dir, "args.json"))

    # Seed
    torch.manual_seed(args.seed)

    # Load Dataset
    data_train = load_dataset(args.train, args.data_size)
    data_pseudo = load_dataset(args.pseudo, args.data_size) if args.pseudo else []
    if args.train_method == "concat":
        data_train += data_pseudo
    data_dev = load_dataset(args.dev, 100)

    data_train = NtcBucketIterator(data_train, args.batch_size, shuffle=True, multi_predicate=args.multi_predicate,
                                   zero_drop=args.zero_drop, bert=args.bert, loss_stop=args.loss_stop,
                                   load_cpu=args.load_cpu, mapping_pseudo_train=args.mapping_pseudo_train,
                                   bert_embed_file=args.train_bert_embed_file,
                                   pseudo_bert_embed_file=args.pseudo_bert_embed_file)
    data_dev = NtcBucketIterator(data_dev, args.batch_size, multi_predicate=args.multi_predicate, bert=args.bert,
                                 load_cpu=args.load_cpu, bert_embed_file=args.dev_bert_embed_file)
    if args.train_method == "pre-train":
        data_pseudo = NtcBucketIterator(data_pseudo, args.batch_size, shuffle=True,
                                        multi_predicate=args.multi_predicate,
                                        zero_drop=args.zero_drop, bert=args.bert, loss_stop=args.loss_stop,
                                        load_cpu=args.load_cpu, mapping_pseudo_train=args.mapping_pseudo_train,
                                        pseudo_bert_embed_file=args.pseudo_bert_embed_file)

    bert_vec_holder = None
    if args.epoch_shuffle:
        bert_vec_holder = BertVecHolder(train_json=args.train,
                                        train_hdf5=args.train_bert_embed_file,
                                        pseudo_json=args.pseudo,
                                        pseudo_hdf5=args.pseudo_bert_embed_file,
                                        data_size=args.data_size)

    word_embedding_matrix = pretrained_word_vecs(args.wiki_embed_dir, "/wordIndex.txt") if args.wiki else None
    model = PackedE2EStackedBiRNN(hidden_dim=args.hidden_dim,
                                  n_layers=args.n_layers,
                                  out_dim=4,
                                  embedding_matrix=word_embedding_matrix,
                                  fixed_word_vec=args.fixed_word_vec,
                                  multi_predicate=args.multi_predicate,
                                  use_wiki_vec=args.wiki,
                                  use_bert_vec=args.bert,
                                  bert_dim=BERT_DIM,
                                  train_bert_embed_file=args.train_bert_embed_file,
                                  dev_bert_embed_file=args.dev_bert_embed_file,
                                  pseudo_bert_embed_file=args.pseudo_bert_embed_file,
                                  load_cpu=args.load_cpu,
                                  dropout=args.dropout,
                                  embed_dropout=args.embed_dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    # Training Method
    print("# Training Method: {}".format(args.train_method), flush=True)
    if args.train_method == "pre-train":
        pretrain_best_thresh = train(log_dir, data_pseudo, data_dev, model, model_id, args.max_epoch,
                                     args.pseudo_lr, args.pseudo_lr / 20,
                                     args.half_checkpoint, bert_vec_holder, "pretrained_")
        with open(path.join(log_dir, "best.pretrain_thresh"), "w") as fo:
            json.dump(pretrain_best_thresh, fo)
    best_thresh = train(log_dir, data_train, data_dev, model, model_id, args.max_epoch,
                        args.lr, args.lr / 20, args.half_checkpoint, bert_vec_holder)
    with open(path.join(log_dir, "best.thresh"), "w") as fo:
        json.dump(best_thresh, fo)
    log.write_endtime()

    if args.decode:
        data_decode = load_dataset(args.test, 100) if args.test else load_dataset(args.dev, 100)
        data_decode = NtcBucketIterator(data_decode, args.batch_size, bert=args.bert,
                                        multi_predicate=args.multi_predicate,
                                        decode=True, load_cpu=args.load_cpu,
                                        bert_embed_file=args.test_bert_embed_file if args.test else args.dev_bert_embed_file)
        tag = "test" if args.test else "dev"
        if args.train_method == "pre-train":
            new_model_id = model_id + "-" + "-".join(str(i) for i in pretrain_best_thresh)
            model.load_state_dict(torch.load(log_dir + "/pretrained_model-" + model_id + ".h5"))
            if args.test:
                model.dev_bert_vec = h5py.File(args.test_bert_embed_file, "r")
            decode(log_dir, data_decode, "pretrained_" + tag, model, new_model_id, pretrain_best_thresh)
        new_model_id = model_id + "-" + "-".join(str(i) for i in best_thresh)
        model.load_state_dict(torch.load(log_dir + "/model-" + model_id + ".h5"))
        decode(log_dir, data_decode, tag, model, new_model_id, best_thresh)


if __name__ == '__main__':
    main()
