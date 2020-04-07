# coding: utf-8
import argparse
import json
from glob import glob
from os import path

from logzero import logger

from evaluation import *

BERT_DIM = 768


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_dir', type=str)

    return parser


def decode_ensemble(out_dir, test_score_files, thres):
    print('# Decode')
    file = open(path.join(out_dir, "predict-ensemble.txt"), "w")

    file_objs = [open(file) for file in test_score_files]

    for lines in tqdm(zip(*file_objs)):
        ave_prob = average_prob(lines)
        p_id = json.loads(lines[0])["pred"]
        sent_id = json.loads(lines[0])["sent"]
        doc_name = json.loads(lines[0])["file"]

        out_dict = {"pred": p_id, "sent": sent_id, "file": doc_name}
        for label_idx, label in enumerate(["ga", "o", "ni"]):
            max_idx = torch.argmax(ave_prob[:, label_idx])
            max_score = ave_prob[max_idx][label_idx] - thres[label_idx]
            if max_score >= 0:
                out_dict[label] = int(max_idx)
        out_dict = json.dumps(out_dict)
        print(out_dict, file=file)


def calculate_train_ensemble_threshold(train_score_files):
    file_objs = [open(file) for file in train_score_files]

    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))
    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    labels = ["ga", "wo", "ni", "all"]

    results = defaultdict(dict)
    best_result = init_result_info(results, thres_lists)

    for lines in tqdm(zip(*file_objs)):
        ys = json.loads(lines[0])["golds"]
        ave_prob = average_prob(lines)
        ave_prob = torch.t(ave_prob)

        for label in range(len(thres_lists)):  # case label index
            p_ys = ave_prob[label]
            result = results[label]
            values, assignments = p_ys.max(0)
            assignment = assignments.item()
            for thres in thres_lists[label]:
                prob = values - thres  # fix

                if not any(y == label for y in ys):
                    if prob < 0:
                        result[thres]["nn"] += 1
                    else:
                        result[thres]["np"] += 1
                elif prob >= 0:
                    if ys[assignment] == label:
                        result[thres]["pp"] += 1
                    else:
                        result[thres]["np"] += 1
                        result[thres]["pn"] += 1
                else:
                    result[thres]["pn"] += 1

    best_thres, f = calc_best_thres(best_result, results, thres_lists, labels)
    logger.info("Best threshold: {}".format(best_thres))

    return best_thres


def average_prob(lines):
    dicts = [json.loads(line) for line in lines]
    assert all(dicts[0]["pred"] == dic["pred"] for dic in dicts)
    assert all(dicts[0]["sent"] == dic["sent"] for dic in dicts)
    assert all(dicts[0]["file"] == dic["file"] for dic in dicts)
    if "golds" in dicts[0]:
        assert all(dicts[0]["golds"] == dic["golds"] for dic in dicts)

    ave_prob = torch.mean(torch.Tensor([dic["scores"] for dic in dicts]), axis=0)

    return ave_prob


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # open files
    train_score_files = sorted(glob(args.model_dir + "/*/score-train-*"))
    test_score_files = sorted(glob(args.model_dir + "/*/score-test-*"))
    train_arg_files = sorted(glob(args.model_dir + "/*/args.json"))

    assert len(train_score_files) == len(test_score_files) == len(train_arg_files)
    logger.info("Ensemble: {} models".format(len(train_score_files)))

    threshold = calculate_train_ensemble_threshold(train_score_files)

    with open(path.join(args.model_dir, "ensemble.thresh"), "w") as fo:
        json.dump(threshold, fo)
    decode_ensemble(args.model_dir, test_score_files, threshold)


if __name__ == '__main__':
    main()
