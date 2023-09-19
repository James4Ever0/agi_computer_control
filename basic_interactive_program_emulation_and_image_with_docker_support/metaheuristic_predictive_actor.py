import math

# extract the kernel from the dead ones.
# the new kernel will be added to the random kernel.
# import copy
import weakref
from typing import Callable

import numpy as np
from typing_extensions import Literal

from alpine_actor import run_actor_forever
from naive_actor import ActorStats
from sequence_learner import PredictorWrapper

ACTIVATION_FUNCMAP = {
    "atan": lambda n: math.atan(n) / math.pi * 2,
    "tanh": math.tanh,
}  # input: -oo, oo; output: -1, 1


class MetaheuristicActorStats(ActorStats):
    ...


class MetaheuristicPredictiveWrapper:
    top_k = 100

    def __init__(
        self,
        ksize: int,
        actorClass,
        predictorClass,
        activation: Literal["atan", "tanh"],
        eps=1e-5,
    ):
        class MetaheuristicActor(actorClass):
            actorStatsClass = MetaheuristicActorStats
            metaWrapperWeakref: Callable[[], MetaheuristicPredictiveWrapper]  # slot

            def __del__(self):
                metaWrapper = self.metaWrapperWeakref()
                super().__del__()
                ...

            def getStatsDict(self):
                statsDict = super().getStatsDict()
                statsDict.update(dict())
                return statsDict

        self.actorClass = MetaheuristicActor
        self.predictorClass = predictorClass
        self.ksize = ksize
        self.trial_count = 0
        self.average_performance = 0
        self.activation = ACTIVATION_FUNCMAP[activation]
        self.eps = eps

    def __next__(self):
        # use inheritance instead of this!
        # use weakref of self
        actor_instance = self.actorClass()
        actor_instance.metaWrapperWeakref = weakref.ref(self)
        return actor_instance

    def new(
        self,
    ):
        del self.actor
        del self.wrapper
        self.wrapper = PredictorWrapper(self.ksize, self.predictorClass)
        self.actor = self.actorClass()

    def get_kernel(self) -> np.ndarray:
        return self.wrapper.predictor.kernel.copy()

    def set_kernel(self, kernel: np.ndarray):
        kernel_shape = kernel.shape
        desired_shape = (self.ksize,)
        assert (
            kernel_shape == desired_shape
        ), f"kernel shape mismatch: {kernel_shape} != {desired_shape}"
        self.wrapper.predictor.kernel = kernel

    kernel = property(fget=get_kernel, fset=set_kernel)

    def remix(self):
        old_kernel = self.kernel
        old_score = self.score()
        avg_performance = self.refresh_average_performance(
            old_score
        )  # warning! base shall never be 1
        # log (avg performance as base) & tanh/atan
        old_add_weight = math.log(old_score / avg_performance, avg_performance)
        old_add_weight = self.activation(old_add_weight) / 2
        # old_add_weight = self.activation(old_add_weight*self.trial_count) / 2
        # old_add_weight = self.activation(old_add_weight*(1+math.log(self.trial_count)) / 2
        self.new()
        new_kernel = self.kernel
        # emit noise if not doing well?
        # zharmony vice versa?
        self.kernel = new_kernel * (0.5 - old_add_weight) + old_kernel * (
            0.5 + old_add_weight
        )

    def refresh_average_performance(self, score: float):
        self.average_performance = (
            self.average_performance * self.trial_count + score
        ) / (self.trial_count + 1)
        self.trial_count += 1
        if self.average_performance == 1:
            self.average_performance += self.eps
        return self.average_performance

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
        # uptime seems to be less universal.
        # let's use loop count for now.
        score = stats.loop_count + self.eps
        # score = stats.up_time + self.eps
        return score


if __name__ == "__main__":
    from alpine_actor import AlpineActor
    from predictive_alpine_actor import PredictorWrapper

    actor_generator = MetaheuristicPredictiveWrapper(
        ksize=100,
        actorClass=AlpineActor,
        predictorClass=PredictorWrapper,
        activation="tanh",
    )
    # breakpoint()
    run_actor_forever(actor_generator)
