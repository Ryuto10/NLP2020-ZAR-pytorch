# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import random
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from pyknp import Juman
from torch.distributions import Categorical
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

MASK = "[MASK]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "_padding_"
CASES = {"が": 0, "を": 1, "に": 2}
BLACK_LIST = ["が", "は", "を", "に", "の", "も", "「", "」", "。", "『", "』", "：", "（", "）", UNK, "へ", "と", "こと"]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

juman = Juman()


def prediction(model, seq_length, device, tokenizer, tokens, mask_ids, black_list, how_select):
    """how select"""
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids += [0] * (seq_length - len(input_ids))
    input_mask = [1] * len(tokens) + [0] * (seq_length - len(tokens))

    input_ids = torch.LongTensor([input_ids]).to(device)
    input_mask = torch.LongTensor([input_mask]).to(device)

    # Repeat updating 'input_ids'
    for mask_id in mask_ids:
        predict = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        # delete black list tokens probability
        bl = [t for t in black_list if t in tokenizer.vocab]
        predict[:, mask_id, tokenizer.convert_tokens_to_ids(BLACK_LIST + bl)] = -np.inf

        if how_select == "sample":
            dist = Categorical(logits=F.log_softmax(predict, dim=-1))
            pred_ids = dist.sample()
        elif how_select == "argmax":
            pred_ids = predict.argmax(dim=-1)
        else:
            raise ValueError("Selection mechanism %s not found!" % how_select)
        input_ids[0][mask_id] = pred_ids[0][mask_id]

    # ids to tokens
    filled_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:len(tokens)].tolist())

    return filled_tokens


def convert_bert_predicts_to_ids(predict_tokens, vocab):
    # Merge subword
    for i in range(len(predict_tokens) - 1, 0, -1):
        if predict_tokens[i].startswith("##"):
            predict_tokens[i - 1] += predict_tokens.pop(i).lstrip("#")

    # Convert tokens to ids
    morphs = juman.analysis("".join(predict_tokens))
    predict_tokens = [m.midasi for m in morphs]
    convert_ids = []
    for i, morph in enumerate(morphs):
        base = morphs[i-1].genkei + morph.genkei if i != 0 and morph.hinsi == "動詞" and morphs[i-1].bunrui == "サ変名詞" else morph.genkei
        pos = morph.hinsi + "-" + morph.bunrui + "-" + morph.katuyou1
        if base in vocab:
            convert_ids.append(vocab[base])
        elif pos in vocab:
            convert_ids.append(vocab[pos])
        else:
            convert_ids.append(vocab[PAD])

    return predict_tokens, convert_ids


def set_vocab(vocab_file):
    """Open file of pre-trained vocab and convert to dict format."""
    print("\n# Load '{}'".format(vocab_file))
    vocab = {}
    f = open(vocab_file)
    for line in f:
        split_line = line.rstrip().split("\t")
        word, idx = split_line[0], split_line[1]
        vocab[word] = idx
    f.close()

    return vocab


def read_file(file):
    with open(file) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_predicates(vocab, data_dir):
    reversed_vocab = {v: k for k, v in vocab.items()}
    predicates = set()
    print("# Load Dataset")
    for basename in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        fn = path.join(data_dir, basename)
        print(fn)
        for instance in tqdm(read_file(fn)):
            for pas in instance["pas"]:
                p_id = pas["p_id"]
                idx = instance["tokens"][p_id]
                token = reversed_vocab[idx]
                base = instance["bases"][p_id]
                # surf = instance["surfaces"][p_id]
                rep = base if "-" in token else token
                predicates.add(rep)

    return list(predicates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default="/work01/ryuto/data/NTC_processed", type=str)
    parser.add_argument("--bert_model", default="/home/ryuto/data/jap_BERT/", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--vocab", default="/home/ryuto/data/NTC_Matsu_original/wordIndex.txt", type=str)

    # model parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model. (If Japanese model, set false)")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    # Hyper parameter
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--insert_max', type=int, default=10)
    parser.add_argument('--insert_min', type=int, default=3)
    parser.add_argument('--target_max', type=int, default=3)
    parser.add_argument('--target_min', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--data_ratio', type=float, default=100)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)

    # vocab & tokenizer
    vocab = set_vocab(args.vocab)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Extract predicates
    predicates = extract_predicates(vocab=vocab, data_dir=args.data_dir)
    random.shuffle(predicates)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()

    counter = 0
    data_size = int(len(predicates) * args.data_ratio / 100)

    with open(args.output_file, "w", encoding='utf-8') as writer:
        for predicate in tqdm(predicates[:data_size]):
            for case in CASES:
                for _ in range(args.iteration):
                    # insert MASK and case
                    n_target = random.randint(args.target_min, args.target_max)
                    text_a = [CLS] + [MASK] * n_target + [case, predicate, SEP]
                    tokens = tokenizer.tokenize(" ".join(text_a))
                    mask_ids = [idx for idx, token in enumerate(tokens) if token == MASK]
                    trg_id = mask_ids[-1]
                    black_list = [predicate]

                    # predict MASK
                    tokens = prediction(model=model,
                                        seq_length=args.max_seq_length,
                                        device=device,
                                        tokenizer=tokenizer,
                                        tokens=tokens,
                                        mask_ids=mask_ids,
                                        black_list=black_list,
                                        how_select="sample")

                    # insert MASK
                    n_insert = random.randint(args.insert_min, args.insert_max)
                    tokens = tokens[:trg_id + 2] + [MASK] * n_insert + tokens[trg_id + 2:]
                    mask_ids2 = [idx for idx, token in enumerate(tokens) if token == MASK]

                    # predict MASK
                    tokens = prediction(model=model,
                                        seq_length=args.max_seq_length,
                                        device=device,
                                        tokenizer=tokenizer,
                                        tokens=tokens,
                                        mask_ids=mask_ids2,
                                        black_list=black_list,
                                        how_select="argmax")

                    target = tokens[mask_ids[0]:mask_ids[-1] + 2]
                    chunk = tokens[mask_ids2[0]:mask_ids2[-1] + 1]
                    prd = tokens[mask_ids2[-1] + 1:len(tokens) - 1]

                    target_tokens, target_ids = convert_bert_predicts_to_ids(target, vocab)
                    chunk_tokens, chunk_ids = convert_bert_predicts_to_ids(chunk, vocab)
                    predicate_tokens, predicate_ids = convert_bert_predicts_to_ids(prd, vocab)

                    concat_surfs = target_tokens + chunk_tokens + predicate_tokens
                    concat_ids = target_ids + chunk_ids + predicate_ids
                    p_id = len(concat_surfs) - 1
                    labels = [3] * len(concat_surfs)
                    labels[len(target_tokens) - 2] = CASES[case]
                    instance = {"tokens": concat_ids,
                                "surfaces": concat_surfs,
                                "pas": [{"p_id": p_id, "args": labels}]}

                    print(json.dumps(instance), file=writer)

                    if counter < 5:
                        counter += 1
                        logger.info("{} + {} = {} {} {}".format(predicate, case,
                                                                "".join(target_tokens),
                                                                "".join(chunk_tokens),
                                                                "".join(predicate_tokens)))


if __name__ == "__main__":
    main()
