import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import h5py

OFFSETS = 10 ** 5


class JointSoftmaxE2EStackedBiRNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 out_dim: int,
                 embedding_matrix,
                 fixed_word_vec: bool = True,
                 multi_predicate: bool = True,
                 use_wiki_vec: bool = False,
                 use_bert_vec: bool = False,
                 bert_dim: int = None,
                 train_bert_embed_file: str = None,
                 dev_bert_embed_file: str = None,
                 load_cpu: bool = False,
                 dropout: float = 0):
        super(JointSoftmaxE2EStackedBiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_dim = out_dim
        self.multi_predicate = multi_predicate
        self.use_wiki_vec = use_wiki_vec
        self.use_bert_vec = use_bert_vec
        self.bert_dim = bert_dim
        self.load_cpu = load_cpu

        self.embedding_dim = 1
        if self.use_wiki_vec:
            self.embedding_dim += embedding_matrix.size(1)
        if self.use_bert_vec:
            self.embedding_dim += self.bert_dim
            self.train_bert_vec = h5py.File(train_bert_embed_file, "r") if train_bert_embed_file else None
            self.dev_bert_vec = h5py.File(dev_bert_embed_file, "r") if dev_bert_embed_file else None
        if self.multi_predicate:
            self.embedding_dim += 1

        if self.use_wiki_vec:
            self.word_emb = nn.Embedding(embedding_matrix.size(0), embedding_matrix.size(1), padding_idx=0)
            self.word_emb.weight = nn.Parameter(embedding_matrix)
        if self.use_wiki_vec and fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        self.gru = PackedBiGRUForSRL(input_dim=self.embedding_dim,
                                     hidden_dim=self.hidden_dim,
                                     num_layers=self.n_layers,
                                     dropout=self.dropout)

        self.output_layer_for_each_word = nn.Linear(self.hidden_dim, self.out_dim)
        self.output_layer_for_all_words = nn.Linear(self.hidden_dim, self.out_dim - 1)  # NULLの分を抜く

    def forward(self, x):
        """メモ：GRUに入れる前に先頭にNULL分の1単語(zero vector)をconcatする"""
        if self.use_bert_vec and not self.load_cpu:
            unique_ids = x.pop()
        xs_len = x.pop()
        xs_len += 1  # NULL分増えるので全体に1足す
        if self.use_bert_vec and self.load_cpu:
            bert_vecs = x.pop()
        if self.multi_predicate:
            is_predicate = x.pop()
        assert len(x) == 2
        words, is_target = x

        embeds = []
        if self.use_wiki_vec:
            words = Variable(words).cuda() if torch.cuda.is_available() else Variable(words)
            embeds.append(self.word_emb(words))
        if self.use_bert_vec:
            if self.load_cpu:
                bert_vecs = Variable(bert_vecs).cuda() if torch.cuda.is_available() else Variable(bert_vecs)
            else:
                if self.training:
                    bert_vecs = [torch.Tensor(self.train_bert_vec.get(str(i))[()])
                                 if i < OFFSETS else
                                 torch.Tensor(self.pseudo_bert_vec.get(str(i))[()])
                                 for i in unique_ids]
                else:
                    bert_vecs = [torch.Tensor(self.dev_bert_vec.get(str(i))[()]) for i in unique_ids]
                bert_vecs = pad_sequence(bert_vecs, batch_first=True, padding_value=0)
                bert_vecs = Variable(bert_vecs).cuda() if torch.cuda.is_available() else Variable(bert_vecs)
            embeds.append(bert_vecs)
        is_target = Variable(is_target).cuda() if torch.cuda.is_available() else Variable(is_target)
        embeds.append(is_target)
        if self.multi_predicate:
            is_predicate = Variable(is_predicate).cuda() if torch.cuda.is_available() else Variable(is_predicate)
            embeds.append(is_predicate)

        exs = torch.cat(embeds, dim=2)
        null_words = torch.zeros(exs.shape[0], 1, exs.shape[2])
        null_words = Variable(null_words).cuda() if torch.cuda.is_available() else Variable(null_words)
        expanded_exs = torch.cat([null_words, exs], dim=1)
        outputs = self.gru(expanded_exs, xs_len)

        out_each_word = [F.log_softmax(self.output_layer_for_each_word(out[:int(x_len)]), dim=1)
                         for out, x_len in zip(outputs, xs_len)]
        out_all_words = [F.log_softmax(self.output_layer_for_all_words(out[:int(x_len)]), dim=0).T
                         for out, x_len in zip(outputs, xs_len)]

        return out_each_word, out_all_words


class MultiSentenceE2EStackedBiRNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 out_dim: int,
                 embedding_matrix,
                 fixed_word_vec: bool,
                 dropout: float = 0):
        super(MultiSentenceE2EStackedBiRNN, self).__init__()

        self.vocab_size = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_dim = out_dim

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        if fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        self.gru = BiGRUForSRL(input_dim=self.embedding_dim + 3,
                               hidden_dim=self.hidden_dim,
                               num_layers=self.n_layers,
                               dropout=self.dropout)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        words, is_prd, is_target, is_sent = x

        if torch.cuda.is_available():
            words = Variable(words).cuda()
            is_target = Variable(is_target).cuda()
            is_prd = Variable(is_target).cuda()
            is_sent = Variable(is_sent).cuda()
        else:
            words = Variable(words)
            is_target = Variable(is_target)
            is_prd = Variable(is_target)
            is_sent = Variable(is_sent)

        embeds = self.word_emb(words)

        inputs = torch.cat([embeds, is_prd, is_target, is_sent], dim=2)
        outputs = self.gru(inputs)

        sent_len = int(sum(is_sent[0, :] == 0))
        outputs = outputs[:, -sent_len:, :]

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]


class PackedE2EStackedBiRNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 out_dim: int,
                 embedding_matrix,
                 fixed_word_vec: bool = True,
                 multi_predicate: bool = True,
                 use_wiki_vec: bool = False,
                 use_bert_vec: bool = False,
                 bert_dim: int = None,
                 train_bert_embed_file: str = None,
                 dev_bert_embed_file: str = None,
                 pseudo_bert_embed_file: str = None,
                 load_cpu: bool = False,
                 dropout: float = 0,
                 embed_dropout: float = 0):
        super(PackedE2EStackedBiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_dropout = embed_dropout
        self.out_dim = out_dim
        self.multi_predicate = multi_predicate
        self.use_wiki_vec = use_wiki_vec
        self.use_bert_vec = use_bert_vec
        self.bert_dim = bert_dim
        self.load_cpu = load_cpu

        self.embedding_dim = 1
        if self.use_wiki_vec:
            self.embedding_dim += embedding_matrix.size(1)
        if self.use_bert_vec:
            self.embedding_dim += self.bert_dim
            self.train_bert_vec = h5py.File(train_bert_embed_file, "r") if train_bert_embed_file else None
            self.dev_bert_vec = h5py.File(dev_bert_embed_file, "r") if dev_bert_embed_file else None
            self.pseudo_bert_vec = h5py.File(pseudo_bert_embed_file, "r") if pseudo_bert_embed_file else None
        if self.multi_predicate:
            self.embedding_dim += 1

        if self.use_wiki_vec:
            self.word_emb = nn.Embedding(embedding_matrix.size(0), embedding_matrix.size(1), padding_idx=0)
            self.word_emb.weight = nn.Parameter(embedding_matrix)
        if self.use_wiki_vec and fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        self.dropout_layer = nn.Dropout(p=embed_dropout)

        self.gru = PackedBiGRUForSRL(input_dim=self.embedding_dim,
                                     hidden_dim=self.hidden_dim,
                                     num_layers=self.n_layers,
                                     dropout=self.dropout)

        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        if self.use_bert_vec and not self.load_cpu:
            unique_ids = x.pop()
        xs_len = x.pop()
        if self.use_bert_vec and self.load_cpu:
            bert_vecs = x.pop()
        if self.multi_predicate:
            is_predicate = x.pop()
        assert len(x) == 2
        words, is_target = x

        embeds = []
        if self.use_wiki_vec:
            words = Variable(words).cuda() if torch.cuda.is_available() else Variable(words)
            embeds.append(self.word_emb(words))
        if self.use_bert_vec:
            if self.load_cpu:
                bert_vecs = Variable(bert_vecs).cuda() if torch.cuda.is_available() else Variable(bert_vecs)
                if self.training:
                    bert_vecs = self.dropout_layer(bert_vecs)
            else:
                if self.training:
                    bert_vecs = [self.dropout_layer(torch.Tensor(self.train_bert_vec.get(str(i))[()]))
                                 if i < OFFSETS else
                                 self.dropout_layer(torch.Tensor(self.pseudo_bert_vec.get(str(i))[()]))
                                 for i in unique_ids]
                else:
                    bert_vecs = [torch.Tensor(self.dev_bert_vec.get(str(i))[()]) for i in unique_ids]
                bert_vecs = pad_sequence(bert_vecs, batch_first=True, padding_value=0)
                bert_vecs = Variable(bert_vecs).cuda() if torch.cuda.is_available() else Variable(bert_vecs)
            embeds.append(bert_vecs)
        is_target = Variable(is_target).cuda() if torch.cuda.is_available() else Variable(is_target)
        embeds.append(is_target)
        if self.multi_predicate:
            is_predicate = Variable(is_predicate).cuda() if torch.cuda.is_available() else Variable(is_predicate)
            embeds.append(is_predicate)

        exs = torch.cat(embeds, dim=2)
        outputs = self.gru(exs, xs_len)

        return [F.log_softmax(self.output_layer(out[:int(x_len)]), dim=1) for out, x_len in zip(outputs, xs_len)]


class E2EStackedBiRNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_layers: int,
                 out_dim: int,
                 embedding_matrix,
                 fixed_word_vec: bool,
                 dropout: float = 0):
        super(E2EStackedBiRNN, self).__init__()

        self.vocab_size = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.dropout = dropout

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        if fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        # input is a set of embedding and predicate marker
        self.gru = BiGRUForSRL(input_dim=self.embedding_dim + 1,
                               hidden_dim=self.hidden_dim,
                               num_layers=self.n_layers,
                               dropout=self.dropout)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        words, is_target = x
        if torch.cuda.is_available():
            words = Variable(words).cuda()
            is_target = Variable(is_target).cuda()
        else:
            words = Variable(words)
            is_target = Variable(is_target)

        embeds = self.word_emb(words)
        inputs = torch.cat([embeds, is_target], dim=2)
        outputs = self.gru(inputs)

        return [F.log_softmax(self.output_layer(out), dim=1) for out in outputs]


class PackedBiGRUForSRL(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0):
        super(PackedBiGRUForSRL, self).__init__()

        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = dropout

        self.gru_in = nn.GRU(input_size=input_dim,
                             hidden_size=hidden_dim,
                             num_layers=1,
                             batch_first=True,
                             dropout=self.dropout)
        self.grus = nn.ModuleList([nn.GRU(input_size=self.hidden_dim,
                                          hidden_size=self.hidden_dim,
                                          num_layers=1,
                                          batch_first=True,
                                          dropout=self.dropout)
                                   for _ in range(num_layers - 1)])

    def forward(self, exs, xs_len):
        ids = torch.cat(
            [torch.cat([torch.arange(x_len - 1, -1, -1), torch.arange(x_len, exs.size(1))]) + i * exs.size(1)
             for i, x_len in enumerate(xs_len)]).long()

        if torch.cuda.is_available():
            exs = exs.cuda()

        packed_exs = pack_padded_sequence(exs, xs_len, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru_in(packed_exs)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        for gru in self.grus:
            flipped = self.reverse(out, ids)

            packed_flipped = pack_padded_sequence(flipped, xs_len, batch_first=True, enforce_sorted=False)
            packed_output, _ = gru(packed_flipped)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            out = flipped + output

        return self.reverse(out, ids)

    @staticmethod
    def reverse(exs, ids):
        cat_exs = exs.reshape(exs.size(0) * exs.size(1), exs.size(2))
        flipped_exs = cat_exs[ids]

        return flipped_exs.reshape(exs.shape)


class BiGRUForSRL(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0):
        super(BiGRUForSRL, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = dropout

        self.gru_in = nn.GRU(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=1,
                             batch_first=True,
                             dropout=self.dropout)
        self.grus = nn.ModuleList([nn.GRU(input_size=self.hidden_dim,
                                          hidden_size=self.hidden_dim,
                                          num_layers=1,
                                          batch_first=True,
                                          dropout=self.dropout)
                                   for _ in range(self.n_layers - 1)])

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        out, _ = self.gru_in(x)
        for gru in self.grus:
            flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
            output, _ = gru(flipped)
            out = flipped + output

        return self.reverse(out.transpose(0, 1)).transpose(0, 1)

    @staticmethod
    def reverse(x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if torch.cuda.is_available():
            idx = idx.cuda()
        return x[idx]
