# coding: utf-8
import argparse
import os
from os import path
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from batchiterator import multi_sent_end2end_single_seq_instance, multi_sent_batch_generator
from evaluation import evaluate_multiclass_without_none
from models import MultiSentenceE2EStackedBiRNN
from utils import load_dataset, pretrained_word_vecs, set_log_file

random.seed(2019)
# MAX_SENTENCE_LENGTH = 90
MAX_SENTENCE_LENGTH = 10000


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='data_path', type=path.abspath,
                        help='data path')
    parser.add_argument('--multi_sent_data', type=path.abspath,
                        help='Path to directory containing multi sentence file.')
    parser.add_argument('--prev_num', type=int,
                        help="Number of previous sentences.")
    parser.add_argument('--model_name', type=str, default="path-pair-bin")
    parser.add_argument('--out_dir', type=str, default='result')

    # Training Option
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--data_size', type=int, default=100,
                        help='data size (%)')
    parser.add_argument('--epoch', dest='max_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)

    # Hyper Parameter
    parser.add_argument('--lr', type=float, default=0.001)
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


def train(out_dir, data_train, data_dev, model, model_id, epoch, lr_start, lr_min):
    len_train = len(data_train)
    len_dev = len(data_dev)

    early_stopping_thres = 4
    early_stopping_count = 0
    best_performance = -1.0
    best_epoch = 0
    best_thres = [0.0, 0.0]
    best_lr = lr_start
    lr = lr_start
    lr_reduce_factor = 0.5
    lr_epsilon = lr_min * 1e-4

    loss_function = nn.NLLLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)
    losses = []

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    for ep in range(epoch):
        total_loss = torch.Tensor([0])
        early_stopping_count += 1

        print(model_id, 'epoch {0}'.format(ep + 1), flush=True)
        print('# Train...', flush=True)
        random.shuffle(data_train)

        model.train()
        for xss, yss in tqdm(data_train, total=len_train, mininterval=5):
            if yss.size(1) > MAX_SENTENCE_LENGTH:
                continue

            optimizer.zero_grad()
            model.zero_grad()

            if torch.cuda.is_available():
                yss = Variable(yss).cuda()
            else:
                yss = Variable(yss)

            scores = model(xss)

            loss = 0
            for i in range(yss.size()[0]):
                loss += loss_function(scores[i], yss[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.data.cpu()

        print("## loss:", total_loss[0], "lr:", lr)
        losses.append(total_loss)
        print("", flush=True)
        print('# Test...', flush=True)

        model.eval()
        thres, obj_score, num_test_batch_instance = evaluate_multiclass_without_none(model, data_dev, len_dev, labels,
                                                                                     thres_lists)
        f = obj_score * 100
        if f > best_performance:
            best_performance = f
            early_stopping_count = 0
            best_epoch = ep + 1
            best_thres = thres
            best_lr = lr
            print("save model", flush=True)
            torch.save(model.state_dict(), out_dir + "/model-" + model_id + ".h5")
        elif early_stopping_count >= early_stopping_thres:
            # break
            if lr > lr_min + lr_epsilon:
                new_lr = lr * lr_reduce_factor
                lr = max(new_lr, lr_min)
                print("load model: epoch{0}".format(best_epoch), flush=True)
                model.load_state_dict(torch.load(out_dir + "/model-" + model_id + ".h5"))
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                early_stopping_count = 0
            else:
                break
        print(model_id, "\tcurrent best epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
              "f:", best_performance)

    print(model_id, "\tbest in epoch", best_epoch, "\t", best_thres, "\t", "lr:", best_lr, "\t",
          "f:", best_performance)


def create_model_id(args):
    seed = "" if args.seed == -1 else "_sub{0}".format(args.seed)

    return "multi_sent{prev_n}-{model_name}-lr{lr}_h{hidden}_layer{layer}_d{drop}_{emb}_size{size}{seed}".format(
        prev_n=args.prev_num,
        model_name=args.model_name,
        lr=args.lr,
        hidden=args.hidden_dim,
        layer=args.n_layers,
        drop=args.dropout,
        emb=args.fixed_word_vec,
        size=args.data_size,
        seed=seed
    )


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print("# Make directory: {}".format(args.out_dir))

    log_dir = path.join(args.out_dir, "log")
    if not path.exists(log_dir):
        os.mkdir(log_dir)
        print("# Make directory: {}".format(log_dir))

    torch.manual_seed(args.seed)
    model_id = create_model_id(args)
    print(model_id)
    set_log_file(log_dir, "train", model_id)

    data_train = load_dataset(args.multi_sent_data + "/train.json", args.data_size)
    data_dev = load_dataset(args.multi_sent_data + "/dev.json", 100)

    data_train = list(multi_sent_end2end_single_seq_instance(data_train, multi_sent_batch_generator, args.prev_num))
    data_dev = list(multi_sent_end2end_single_seq_instance(data_dev, multi_sent_batch_generator, args.prev_num))

    word_embedding_matrix = pretrained_word_vecs(args.data_path, "/wordIndex.txt")
    model = MultiSentenceE2EStackedBiRNN(hidden_dim=args.hidden_dim,
                                         n_layers=args.n_layers,
                                         out_dim=4,
                                         embedding_matrix=word_embedding_matrix,
                                         fixed_word_vec=args.fixed_word_vec,
                                         dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    train(args.out_dir, data_train, data_dev, model, model_id, args.max_epoch, args.lr, args.lr / 20)


if __name__ == '__main__':
    main()
