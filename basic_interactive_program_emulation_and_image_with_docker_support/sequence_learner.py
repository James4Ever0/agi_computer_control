import numpy as np
from typing import List


class NaivePredictor:
    def __init__(self, ksize):
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


from collections import deque

if __name__ == "__main__":
    ksize = 10
    predictor = NaivePredictor(ksize=ksize)
    seq = deque([0, 1, 2, 3, 4], maxlen=ksize)
    total_seq = list(seq)
    for _ in range(100):
        tok = predictor.predict(list(seq))
        seq.append(tok)
        total_seq.append(tok)
    print("total seq:", total_seq)
