from collections import deque
from typing import List

import numpy as np


class NaivePredictor:
    def __init__(self, ksize: int):
        self.ksize = ksize
        self.kernel = np.random.rand(self.ksize)

    def predict(self, x: List[int]):
        x_processed = x[-self.ksize :]
        if len(x_processed) < self.ksize:
            x_processed = [0] * (self.ksize - len(x_processed)) + x_processed
        x_one_hot = self.one_hot(x_processed)
        next_token = np.matmul(x_one_hot, self.kernel)
        ret = np.argmax(next_token)
        return ret

    def one_hot(self, x):
        x_one_hot = np.eye(self.ksize)[x]
        return x_one_hot


class PredictorWrapper:
    def __init__(self, ksize: int, predictor_cls: NaivePredictor):
        self.predictor: NaivePredictor = predictor_cls(ksize)
        self.seq = deque([], maxlen=ksize)

    def enqueue(self, seq: List[int]):
        for tok in seq:
            self.seq.append(tok)

    def predict(self, seqlen: int):
        ret_seq = []
        for _ in range(seqlen):
            tok = self.predictor.predict(list(self.seq))
            self.seq.append(tok)
            ret_seq.append(tok)
        return ret_seq


if __name__ == "__main__":
    pw = PredictorWrapper(10, NaivePredictor)
    pw.enqueue([0, 1, 2, 3, 4])
    total_seq = pw.predict(100)
    # ksize = 10
    # predictor = NaivePredictor(ksize=ksize)
    # seq = deque([0, 1, 2, 3, 4], maxlen=ksize)
    # total_seq = list(seq)
    # for _ in range(100):
    #     tok = predictor.predict(list(seq))
    #     seq.append(tok)
    #     total_seq.append(tok)
    print("total seq:", total_seq)
