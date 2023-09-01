# extract the kernel from the dead ones.
# the new kernel will be added to the random kernel.
import numpy as np
from sequence_learner import PredictorWrapper

class MetaheuristicPredictiveWrapper:
    top_k = 100
    def __init__(self):
        self.wrapper = PredictorWrapper()
        self.actor = ...
    def new(self,):
        self.actor = ...
    @staticmethod
    def score():
        ...