# extract the kernel from the dead ones.
# the new kernel will be added to the random kernel.
import numpy as np
from sequence_learner import PredictorWrapper
from naive_actor import ActorStats


class MetaheuristicPredictiveWrapper:
    top_k = 100

    def __init__(self):
        self.wrapper = PredictorWrapper()
        self.actor = ...

    def new(
        self,
    ):
        del self.actor
        self.actor = ...

    def score(self):
        stats: ActorStats = self.actor.stats
        # score by what?
