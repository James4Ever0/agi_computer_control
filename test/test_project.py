import sys
sys.path.append("../")

from recording_train_parse import getTrainingData

import pytest
@pytest.fixture()
def basePath():
    return "../recordings/2023-06-02T07_59_45.711256/"

def test_get_training_data(basePath:str):
    for trainingDataFrame in getTrainingData(basePath):
        print(trainingDataFrame)
        
# test fetching training data.

def test_fetching_training_data(basePath:str):
    from conscious_struct import trainModelWithDataBasePath, TestEnqueue
    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)