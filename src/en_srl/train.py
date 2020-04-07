import argparse
import json
import os
from datetime import datetime
from os import path
from typing import List, Dict, Iterable
import numpy as np
import random
import torch
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder, PretrainedBertEmbedder, \
    PretrainedTransformerEmbedder
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from log import StandardLogger, write_args_log
from models import SemanticRoleLabelerWithAttention

random.seed(2020)
np.random.seed(2020)

GLOVE = "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz"
GLOVE_DIM = 100
ELMO_OPT = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_DIM = 1024
BERT_MODEL = "bert-base-uncased"
BERT_DIM = 768
XLNET_MODEL = "xlnet-base-cased"
XLNET_DIM = 768

def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pseudo', type=path.abspath,
                        help="Path to pseudo jsonl file.")
    parser.add_argument('--train', type=path.abspath,
                        help="Path to training jsonl file.")
    parser.add_argument('--dev', type=path.abspath,
                        help="Path to validation jsonl file.")
    # Optional
    parser.add_argument('--test', type=path.abspath, default=None,
                        help="Path to test jsonl file.")
    parser.add_argument('--out_dir', type=path.abspath, default="./",
                        help="Path to output directory.")
    parser.add_argument('--model', type=path.abspath, default=None,
                        help="Path to pre-trained model.")
    parser.add_argument('--comment', type=str, default=None)

    # Train option
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', type=int, default=128,
                        help="Mini batch size")
    parser.add_argument('--epoch', type=int, dest='max_epoch', default=500,
                        help="Max epoch")
    parser.add_argument('--early_stopping', type=int, default=10,
                        help="Number of epochs to be patient before early stopping")
    parser.add_argument('--data_ratio', type=float, default=100,
                        help="Ratio of training data size (full size is 100, default=100).")
    parser.add_argument('--glove', action='store_true',
                        help="use glove")
    parser.add_argument('--elmo', action='store_true',
                        help="use ELMo")
    parser.add_argument('--bert', action='store_true',
                        help="use BERT")
    parser.add_argument('--xlnet', action='store_true',
                        help="use XLNet")
    parser.add_argument('--highway', action='store_true',
                        help="use He+'17 model (highway LSTM)")
    parser.add_argument('--attention', action='store_true',
                        help="use attention")
    parser.add_argument('--multi_predicate', action='store_true',
                        help="use multi predicate")
    parser.add_argument('--train_method', type=str, default='concat',
                        help="Choose from 'concat' or 'pre-train'")
    parser.add_argument('--test_only', action='store_true',
                        help="Only test, no train.")

    # Hyper parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--embed_dim', type=int, default=100,
                        help="Number of dimensions of embedding")
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help="Number of dimensions of RNN hidden state.")
    parser.add_argument('--binary_dim', type=int, default=1,
                        help="Number of dimensions of binary feature.")
    parser.add_argument('--n_layers', type=int, default=8,
                        help="Number of Layers of RNN")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout of RNN")
    parser.add_argument('--embed_dropout', type=float, default=0.0,
                        help="Dropout of embedding")
    parser.add_argument('--grad_clipping', type=float, default=None,
                        help="Clips gradient norm (default=None)")
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Choose from 'Adam', 'SGD', or 'Adadelta'")

    return parser


def extract_name(file):
    name = path.basename(file)
    name, _ = name.split(".", 1)

    return name


def create_model_id(args):
    model_id = datetime.today().strftime("%m%d%H%M")
    model_id += "-" + extract_name(args.pseudo) if args.pseudo else ""
    model_id += "-" + extract_name(args.test) if args.test else ""
    model_id += "-" + args.train_method
    model_id += "-highway" if args.highway else ""
    model_id += "-attention" if args.attention else ""
    model_id += "-mp" if args.multi_predicate else ""
    model_id += "-glove" if args.glove else ""
    model_id += "-elmo" if args.elmo else ""
    model_id += "-bert" if args.bert else ""
    model_id += "-xlnet" if args.xlnet else ""
    model_id += "-lr" + str(args.learning_rate)
    model_id += "-seed" + str(args.seed)
    model_id += "-" + args.comment if args.comment else ""

    return model_id


class SrlDatasetReader(DatasetReader):
    """DatasetReader for Semantic Role Labeling"""

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.ratio = 100
        self.data_size = None

    def read_with_ratio(self, file_path: str, ratio: float = 100) -> Iterable[Instance]:
        self.ratio = ratio
        return self.read(file_path)

    def _read(self, file_path: str) -> Instance:
        """
        Per line: {"tokens": tokens, "labels": labels}
            * tokens: List of surfaces (length = number of words).
            * labels: [{"verb_idx": verb_idx, "tags": tags}, ...]
                - verb_idx: Index indicating the position of the predicate.
                - tags: List of predicate arguments.
        """
        with open(file_path) as fi:
            instances = [json.loads(line) for line in fi]
        self.data_size = int(len(instances) * self.ratio / 100)
        print("\n\n# Load Dataset: {}".format(file_path))
        print("# Data size: {} -> {} sentences (ratio = {})".format(len(instances), self.data_size, self.ratio),
              flush=True)

        for sentence_instance in instances[:self.data_size]:
            tokens = [Token(token) for token in sentence_instance["tokens"]]
            multi_verb_indices = [idx for label in sentence_instance["labels"] for idx in label["verb_span"]]
            for label in sentence_instance["labels"]:
                yield self.text_to_instance(tokens, label["verb_span"], multi_verb_indices, label["tags"])

    def text_to_instance(self, tokens: List[Token], verb_span: List[int], multi_verb_indices: List[int],
                         tags: List[str]) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        verb_indicator = [1 if idx in verb_span else 0 for idx in range(len(tokens))]
        multi_verb_indicator = [1 if idx in multi_verb_indices else 0 for idx in range(len(tokens))]
        verb_idx = verb_indicator.index(1)
        verb = tokens[verb_idx].text
        metadata_dict = {"words": [x.text for x in tokens],
                         "verb": verb,
                         "verb_index": verb_idx,
                         "gold_tags": tags}
        fields = {"tokens": sentence_field,
                  "verb_indicator": SequenceLabelField(labels=verb_indicator, sequence_field=sentence_field),
                  "multi_verb_indicator": SequenceLabelField(labels=multi_verb_indicator,
                                                             sequence_field=sentence_field),
                  "tags": SequenceLabelField(labels=tags, sequence_field=sentence_field),
                  "metadata": MetadataField(metadata_dict)}

        return Instance(fields)


def main():
    parser = create_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_id = create_model_id(args)

    if not path.exists(args.out_dir):
        print("# Create directory: {}".format(args.out_dir))
        os.mkdir(args.out_dir)

    # log file
    out_dir = path.join(args.out_dir, "out-" + model_id)
    print("# Create output directory: {}".format(out_dir))
    os.mkdir(out_dir)
    log = StandardLogger(path.join(out_dir, "log-" + model_id + ".txt"))
    log.write(args=args)
    write_args_log(args, path.join(out_dir, "args.json"))

    # dataset reader
    token_indexers = {"tokens": SingleIdTokenIndexer(),
                      "elmo": ELMoTokenCharactersIndexer(),
                      "bert": PretrainedBertIndexer(BERT_MODEL, use_starting_offsets=True),
                      "xlnet": PretrainedTransformerIndexer(XLNET_MODEL, do_lowercase=False)}

    reader = SrlDatasetReader(token_indexers)

    # dataset
    train_dataset = reader.read_with_ratio(args.train, args.data_ratio)
    validation_dataset = reader.read_with_ratio(args.dev, 100)
    pseudo_dataset = reader.read_with_ratio(args.pseudo, args.data_ratio) if args.pseudo else []
    all_dataset = train_dataset + validation_dataset + pseudo_dataset
    if args.test:
        test_dataset = reader.read_with_ratio(args.test, 100)
        all_dataset += test_dataset

    vocab = Vocabulary.from_instances(all_dataset)

    # embedding
    input_size = args.binary_dim * 2 if args.multi_predicate else args.binary_dim
    if args.glove:
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=GLOVE_DIM,
                                    trainable=True,
                                    pretrained_file=GLOVE)
        input_size += GLOVE_DIM
    else:
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=args.embed_dim,
                                    trainable=True)
        input_size += args.embed_dim
    token_embedders = {"tokens": token_embedding}

    if args.elmo:
        elmo_embedding = ElmoTokenEmbedder(options_file=ELMO_OPT,
                                           weight_file=ELMO_WEIGHT)
        token_embedders["elmo"] = elmo_embedding
        input_size += ELMO_DIM

    if args.bert:
        bert_embedding = PretrainedBertEmbedder(BERT_MODEL)
        token_embedders["bert"] = bert_embedding
        input_size += BERT_DIM

    if args.xlnet:
        xlnet_embedding = PretrainedTransformerEmbedder(XLNET_MODEL)
        token_embedders["xlnet"] = xlnet_embedding
        input_size += XLNET_DIM

    word_embeddings = BasicTextFieldEmbedder(token_embedders=token_embedders,
                                             allow_unmatched_keys=True,
                                             embedder_to_indexer_map={"bert": ["bert", "bert-offsets"],
                                                                      "elmo": ["elmo"],
                                                                      "tokens": ["tokens"],
                                                                      "xlnet": ["xlnet"]})
    # encoder
    if args.highway:
        lstm = PytorchSeq2SeqWrapper(StackedAlternatingLstm(input_size=input_size,
                                                            hidden_size=args.hidden_dim,
                                                            num_layers=args.n_layers,
                                                            recurrent_dropout_probability=args.dropout))
    else:
        pytorch_lstm = torch.nn.LSTM(input_size=input_size,
                                     hidden_size=args.hidden_dim,
                                     num_layers=int(args.n_layers / 2),
                                     batch_first=True,
                                     dropout=args.dropout,
                                     bidirectional=True)
        # initialize
        for name, param in pytorch_lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Wii, Wif, Wic, Wio
                for n in range(4):
                    torch.nn.init.orthogonal_(param.data[args.hidden_dim * n:args.hidden_dim * (n + 1)])
            elif 'bias' in name:
                param.data.fill_(0)

        lstm = PytorchSeq2SeqWrapper(pytorch_lstm)

    # model
    hidden_dim = args.hidden_dim if args.highway else args.hidden_dim * 2  # pytorch.nn.LSTMはconcatされるので2倍
    model = SemanticRoleLabelerWithAttention(vocab=vocab,
                                             text_field_embedder=word_embeddings,
                                             encoder=lstm,
                                             binary_feature_dim=args.binary_dim,
                                             embedding_dropout=args.embed_dropout,
                                             attention_dropout=0.0,
                                             use_attention=args.attention,
                                             use_multi_predicate=args.multi_predicate,
                                             hidden_dim=hidden_dim)

    if args.model:
        print("# Load model parameter: {}".format(args.model))
        with open(args.model, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    # optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    else:
        raise ValueError("unsupported value: '{}'".format(args.optimizer))

    # iterator
    # iterator = BucketIterator(batch_size=args.batch, sorting_keys=[("tokens", "num_tokens")])
    iterator = BasicIterator(batch_size=args.batch)
    iterator.index_with(vocab)

    if not args.test_only:
        # Train
        print("# Train Method: {}".format(args.train_method))
        print("# Start Train", flush=True)
        if args.train_method == "concat":
            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=train_dataset + pseudo_dataset,
                              validation_dataset=validation_dataset,
                              validation_metric="+f1-measure-overall",
                              patience=args.early_stopping,
                              num_epochs=args.max_epoch,
                              num_serialized_models_to_keep=5,
                              grad_clipping=args.grad_clipping,
                              serialization_dir=out_dir,
                              cuda_device=cuda_device)
            trainer.train()
        elif args.train_method == "pre-train":
            pre_train_out_dir = path.join(out_dir + "pre-train")
            fine_tune_out_dir = path.join(out_dir + "fine-tune")
            os.mkdir(pre_train_out_dir)
            os.mkdir(fine_tune_out_dir)

            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=pseudo_dataset,
                              validation_dataset=validation_dataset,
                              validation_metric="+f1-measure-overall",
                              patience=args.early_stopping,
                              num_epochs=args.max_epoch,
                              num_serialized_models_to_keep=3,
                              grad_clipping=args.grad_clipping,
                              serialization_dir=pre_train_out_dir,
                              cuda_device=cuda_device)
            trainer.train()

            if args.optimizer == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "Adadelta":
                optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
            else:
                raise ValueError("unsupported value: '{}'".format(args.optimizer))
            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=train_dataset,
                              validation_dataset=validation_dataset,
                              validation_metric="+f1-measure-overall",
                              patience=args.early_stopping,
                              num_epochs=args.max_epoch,
                              num_serialized_models_to_keep=3,
                              grad_clipping=args.grad_clipping,
                              serialization_dir=fine_tune_out_dir,
                              cuda_device=cuda_device)
            trainer.train()
        else:
            raise ValueError("Unsupported Value '{}'".format(args.train_method))

    # Test
    if args.test:
        print("# Test")
        result = evaluate(model=model,
                          instances=test_dataset,
                          data_iterator=iterator,
                          cuda_device=cuda_device,
                          batch_weight_key="")
        with open(path.join(out_dir, "test.score"), 'w') as fo:
            json.dump(result, fo)

    log.write_endtime()


if __name__ == "__main__":
    main()
