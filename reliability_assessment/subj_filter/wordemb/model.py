import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.model_selection import train_test_split


class Preprocess:
    DATA_COLUMN = "text"
    LABEL_COLUMN = "polarity"

    def __init__(self, X: list, y: list, tokenizer: FullTokenizer, max_seq_len=192):
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        X, y = self._prepare(X, y)
        SEED = 2000
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.005, random_state=SEED)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [train_x, test_x])
        self.train_y = train_y
        self.test_y = test_y

    def _prepare(self, X: list, y: list):
        x = []
        for text in X:
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)
