from pytorch_pretrained_bert.tokenization import BertTokenizer
from typing import List


class SubWordAlignment(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        super(SubWordAlignment, self).__init__(vocab_file, do_lower_case, max_len, never_split)

    def alignment(self, tokens: List[str], subwords: List[str]):
        tokenized_tokens = [[tokenized_tokens for tokenized_tokens in self.basic_tokenizer.tokenize(token)]
                            for token in tokens]
        assert len(tokenized_tokens) == len(tokens)

        idx = 0
        indices = []
        for chunk in tokenized_tokens:
            buffer = []
            for token in chunk:
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    buffer.append(idx)
                    assert sub_token == subwords[idx]
                    idx += 1
            indices.append(buffer)

        assert len(subwords) == idx
        assert len(tokens) == len(indices)

        return indices
