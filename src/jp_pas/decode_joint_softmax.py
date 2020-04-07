# coding: utf-8
import argparse
import json
import math
from os import path

import torch
from logzero import logger
from tqdm import tqdm

from batchiterator import NtcBucketIterator
from evaluation import evaluate_joint_softmax_multiclass_without_none
from models import JointSoftmaxE2EStackedBiRNN
from utils import load_dataset, pretrained_word_vecs

BERT_DIM = 768


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_data', type=path.abspath, default=None)
    parser.add_argument('--test_bert_embed_file', type=path.abspath, default=None, help="hdf5 file")
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--tag', type=str, default="dev",
                        help='type of evaluation data split')
    parser.add_argument('--thres', action='store_true')
    parser.add_argument('--scores', action='store_true')

    return parser


def decode(out_dir, data, tag, model, model_id, thres):
    print('# Decode')
    file = open(out_dir + "/predict-" + tag + '-' + model_id + ".txt", "w")
    data.create_batches()
    model.eval()
    for xss, yss in tqdm(data, mininterval=5):
        out_each_word, out_all_words = model(xss)

        # for pred_no in range(yss.size()[0]):
        for pred_no in range(len(yss)):
            predict = out_each_word[pred_no].cpu()
            predict = torch.pow(torch.zeros(predict.size()) + math.e, predict.data)

            # add
            p_id, sent_id, doc_name = yss[pred_no]
            out_dict = {"pred": p_id, "sent": sent_id, "file": doc_name}
            for label_idx, label in enumerate(["ga", "o", "ni"]):
                max_idx = torch.argmax(predict[:, label_idx])
                max_score = predict[max_idx][label_idx] - thres[label_idx]
                if max_score >= 0:
                    out_dict[label] = int(max_idx)
            out_dict = json.dumps(out_dict)
            print(out_dict, file=file)


def decode_scores(out_dir, data, tag, model, model_id):
    logger.info('# Decode Scores')
    file = open(out_dir + "/score-" + tag + '-' + model_id + ".txt", "w")
    data.create_batches()
    model.eval()
    for xss, yss in tqdm(data, mininterval=5):
        out_each_word, out_all_words = model(xss)

        for pred_no in range(len(yss)):
            predict = out_each_word[pred_no].cpu()
            predict = torch.pow(torch.zeros(predict.size()) + math.e, predict.data)
            p_id, sent_id, doc_name = yss[pred_no]
            out_dict = {"pred": p_id, "sent": sent_id, "file": doc_name, "scores": predict.tolist()}
            out_dict = json.dumps(out_dict)
            print(out_dict, file=file)


def calculate_train_threshold(train_args, model):
    logger.info("Train args:")
    for k, v in train_args.items():
        logger.info("\t{}: {}".format(k, v))

    # Load Dataset
    data_train = load_dataset(train_args["train"], train_args["data_size"])
    if train_args["train_method"] == "concat":
        data_pseudo = load_dataset(train_args["pseudo"], train_args["data_size"]) if train_args["pseudo"] else []
        data_train += data_pseudo

    data_train = NtcBucketIterator(data_train, train_args["batch_size"],
                                   multi_predicate=train_args["multi_predicate"],
                                   bert=train_args["bert"],
                                   load_cpu=train_args.get("load_cpu"),
                                   bert_embed_file=train_args.get("train_bert_embed_file"),
                                   joint_softmax=True)

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    data_train.create_batches()
    model.eval()
    thres, *_ = evaluate_joint_softmax_multiclass_without_none(model, data_train, len(data_train),
                                                               labels, thres_lists, logger)
    logger.info("Best threshold: {}".format(thres))

    return thres


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    out_dir = path.dirname(args.model_file)
    model_id = path.basename(out_dir)
    with open(path.join(out_dir, "args.json")) as fi:
        train_args = json.load(fi)

    word_embedding_matrix = pretrained_word_vecs(train_args["wiki_embed_dir"], "/wordIndex.txt") if train_args[
        "wiki"] else None

    model = JointSoftmaxE2EStackedBiRNN(hidden_dim=train_args["hidden_dim"],
                                        n_layers=train_args["n_layers"],
                                        out_dim=4,
                                        embedding_matrix=word_embedding_matrix,
                                        fixed_word_vec=train_args["fixed_word_vec"],
                                        multi_predicate=train_args["multi_predicate"],
                                        use_wiki_vec=train_args["wiki"],
                                        use_bert_vec=train_args["bert"],
                                        bert_dim=BERT_DIM,
                                        dev_bert_embed_file=args.test_bert_embed_file,
                                        load_cpu=train_args["load_cpu"],
                                        dropout=train_args["dropout"])

    model.load_state_dict(torch.load(args.model_file))

    if torch.cuda.is_available():
        model = model.cuda()

    data = load_dataset(args.test_data, 100)
    data = NtcBucketIterator(data, train_args["batch_size"], multi_predicate=train_args["multi_predicate"],
                             bert=train_args["bert"], decode=True,
                             load_cpu=train_args["load_cpu"], bert_embed_file=args.test_bert_embed_file)

    if args.scores:
        decode_scores(out_dir, data, args.tag, model, model_id)
    else:
        if args.thres:
            fn = path.join(path.dirname(args.model_file), "best.thresh")
            with open(fn) as fi:
                threshold = json.load(fi)
            logger.info("Loaded Threshold: {}".format(threshold))
        else:
            threshold = calculate_train_threshold(train_args, model)

        with open(path.join(out_dir, "test.thresh"), "w") as fo:
            json.dump(threshold, fo)

        new_model_id = model_id + "-" + "-".join(str(i) for i in threshold)
        print(new_model_id)

        decode(out_dir, data, args.tag, model, new_model_id, threshold)


if __name__ == '__main__':
    main()
