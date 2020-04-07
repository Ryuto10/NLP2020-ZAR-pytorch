# coding: utf-8
import argparse
import json
import os
import random
from datetime import datetime
from os import path

import logzero
import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from torch.autograd import Variable
from tqdm import tqdm

from batchiterator import NtcBucketIterator
from decode_joint_softmax import decode
from evaluation import evaluate_joint_softmax_multiclass_without_none
from log import StandardLogger, write_args_log
from models import JointSoftmaxE2EStackedBiRNN
from utils import load_dataset, pretrained_word_vecs

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
    parser.add_argument('--out_dir', type=str, default='result')
    parser.add_argument('--wiki_embed_dir', type=path.abspath, default=None)
    parser.add_argument('--train_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--dev_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--test_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")

    # Training Option
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=100,
                        help='data size (%)')
    parser.add_argument('--epoch', dest='max_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--loss_stop', action='store_true')
    parser.add_argument('--wiki', action='store_true')
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--multi_predicate', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--load_cpu', action='store_true')
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
    parser.add_argument('--tune-word-vec', dest='fixed_word_vec', action='store_false',
                        help='do not re-train word vec')

    parser.set_defaults(fixed_word_vec=True)

    return parser


def train(out_dir, data_train, data_dev, model, model_id, epoch, lr_start, lr_min, train_type=""):
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

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    best_f1_history = []

    for ep in range(epoch):
        total_loss_each_word = torch.Tensor([0])
        total_loss_all_words = torch.Tensor([0])
        early_stopping_count += 1

        logger.info('{}: epoch {}'.format(model_id, ep + 1))

        logger.info('# Train')
        data_train.create_batches()
        model.train()

        for n, (xss, yss) in tqdm(enumerate(data_train), total=len_train, mininterval=5):
            if xss[0].size(1) > MAX_SENTENCE_LENGTH:
                continue

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = [(Variable(ys[0]).cuda(), Variable(ys[1]).cuda()) for ys in yss]
            else:
                yss = [(Variable(ys[0]), Variable(ys[1])) for ys in yss]

            out_each_word, out_all_words = model(xss)

            loss_each_word = 0
            loss_all_words = 0
            for i in range(len(yss)):
                gold_each_word, gold_all_words = yss[i]
                loss_each_word += loss_function(out_each_word[i], gold_each_word)
                loss_all_words += loss_function(out_all_words[i], gold_all_words)
            loss = loss_each_word + loss_all_words
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss_each_word += loss_each_word.data.cpu()
            total_loss_all_words += loss_all_words.data.cpu()

        logger.info("## Loss: each word = {}, all words = {}".format(total_loss_each_word[0], total_loss_all_words[0]))
        logger.info("## LR: {}".format(lr))
        logger.info('# Test')

        data_dev.create_batches()
        model.eval()
        thres, obj_score, num_test_batch_instance = evaluate_joint_softmax_multiclass_without_none(
            model, data_dev, len_dev, labels, thres_lists, logger)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            logger.info("## save model")
            torch.save(model.state_dict(), out_dir + "/{}model-".format(train_type) + model_id + ".h5")

            best_f1_history.append((best_performance, best_epoch))

        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                logger.info("load model: epoch{0}".format(best_epoch))
                model.load_state_dict(torch.load(out_dir + "/{}model-".format(train_type) + model_id + ".h5"))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        logger.info("{}\tcurrent best epoch: {}\t{}\tlr: {}\tf: {}".format(
            model_id, best_epoch, best_thres, best_lr, best_performance))

    logger.info("{}\tbest in epoch: {}\t{}\tlr: {}\tf: {}".format(
        model_id, best_epoch, best_thres, best_lr, best_performance))

    logger.info("Update History: {}".format(best_f1_history))

    return best_thres


def extract_name(file):
    name = path.basename(file)
    name, _ = name.split(".", 1)

    return name


def create_model_id(args):
    model_id = datetime.today().strftime("%m%d%H%M")
    model_id += "-" + extract_name(args.test) if args.test else ""
    model_id += "-wiki" if args.wiki else ""
    model_id += "-bert" if args.bert else ""
    model_id += "-mp" if args.multi_predicate else ""
    model_id += "-lr" + str(args.lr)
    model_id += "-seed" + str(args.seed)
    model_id += "-" + args.comment if args.comment else ""

    return model_id


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

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

    logfn = path.join(log_dir, "logzero-train-{}.txt".format(model_id))
    logzero.logfile(logfn)
    logger.info(args)
    log = StandardLogger(path.join(log_dir, "log-" + model_id + ".txt"))
    log.write(args=args, comment=model_id)
    write_args_log(args, path.join(log_dir, "args.json"))

    # Seed
    torch.manual_seed(args.seed)

    # Load Dataset
    data_train = load_dataset(args.train, args.data_size)
    data_dev = load_dataset(args.dev, 100)

    data_train = NtcBucketIterator(data_train, args.batch_size, shuffle=True, multi_predicate=args.multi_predicate,
                                   bert=args.bert, load_cpu=args.load_cpu, bert_embed_file=args.train_bert_embed_file,
                                   joint_softmax=True)
    data_dev = NtcBucketIterator(data_dev, args.batch_size, multi_predicate=args.multi_predicate, bert=args.bert,
                                 load_cpu=args.load_cpu, bert_embed_file=args.dev_bert_embed_file, joint_softmax=True)

    word_embedding_matrix = pretrained_word_vecs(args.wiki_embed_dir, "/wordIndex.txt") if args.wiki else None

    model = JointSoftmaxE2EStackedBiRNN(hidden_dim=args.hidden_dim,
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
                                        load_cpu=args.load_cpu,
                                        dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    best_thresh = train(log_dir, data_train, data_dev, model, model_id, args.max_epoch,
                        args.lr, args.lr / 20)

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
        new_model_id = model_id + "-" + "-".join(str(i) for i in best_thresh)
        model.load_state_dict(torch.load(log_dir + "/model-" + model_id + ".h5"))
        decode(log_dir, data_decode, tag, model, new_model_id, best_thresh)


if __name__ == '__main__':
    main()
