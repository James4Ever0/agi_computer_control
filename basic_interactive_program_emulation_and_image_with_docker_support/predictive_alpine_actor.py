from alpine_actor import run_actor_forever, AlpineActor
from sequence_learner import NaivePredictor


class PredictiveAlpineActor(AlpineActor):
    def __init__(self):
        self.predictor = NaivePredictor(256)
        super().__init__()
    
    @AlpineActor.timeit
    def loop(self):
        read_content = self.read()
        write_content = bytes(predicted_content)
        self.write(write_content)
        return True


if __name__ == "__main__":
    run_actor_forever(PredictiveAlpineActor)
