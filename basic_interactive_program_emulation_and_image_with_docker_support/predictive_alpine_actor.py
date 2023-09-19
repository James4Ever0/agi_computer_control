import random

from alpine_actor import AlpineActor, run_actor_forever
from sequence_learner import NaivePredictor, PredictorWrapper

# from vocabulary import AsciiVocab
from vocabulary import BytesVocab


class PredictiveAlpineActor(AlpineActor):
    def __init__(self, ksize = 256):
        self.predictorWrapper = PredictorWrapper(ksize, NaivePredictor)
        self.predictorWrapper.seq.extend(list(BytesVocab.generate()))
        # self.predictorWrapper.seq.extend([ord(c) for c in AsciiVocab.generate()])
        super().__init__()

    @property
    def write_len(self):
        return random.randint(10, 30)

    @AlpineActor.timeit
    def loop(self):
        read_content = self.read()
        self.predictorWrapper.enqueue(list(read_content))
        predicted_content = self.predictorWrapper.predict(self.write_len)
        write_content = bytes(predicted_content)
        self.write(write_content)
        return True


if __name__ == "__main__":
    run_actor_forever(PredictiveAlpineActor)
