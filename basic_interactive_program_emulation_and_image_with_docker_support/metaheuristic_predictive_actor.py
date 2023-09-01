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
        # example:
        """
        =====================summary======================
        start time:     2023-09-01T09:54:11.270057+08:00
        end time:       2023-09-01T09:54:43.327770+08:00
        up time:        32.05771350860596
        loop count:     290
        total bytes read:       237
        total bytes write:      2476
        r/w ratio: 0.09571890145395799
        w/r ratio: 10.447257383966244
        read bytes entropy: 4.946365138818157
        write bytes entropy: 6.148352516530523
        r/w entropy ratio: 0.8045025273875089
        w/r entropy ratio: 1.2430041745764768
        """
        # for now, just take the up time
        score = stats.up_time
        return score