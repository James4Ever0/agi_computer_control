from recording_train_parse import getTrainingData

import pytest
import sys
sys.path.append("../")
@pytest.fixture()
def basePath():
    return "recordings/2023-06-02T07_59_45.711256/"

def test_get_training_data(basePath:str):
    for trainingDataFrame in getTrainingData(basePath):
        print(trainingDataFrame)
        
# test fetching training data.
from conscious_struct import trainModelWithDataBasePath, TestEnqueue

def test_fetching_training_data(basePath:str):
    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)