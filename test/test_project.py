
# use "--log-level" in pytest.

import sys
sys.path.append("../")

import logging
logging.basicConfig(level=logging.DEBUG)
from recording_train_parse import getTrainingData

import pytest
@pytest.fixture()
def basePath():
    return "../recordings/2023-06-02T07_59_45.711256/"

def test_get_training_data(basePath:str):
    for trainingDataFrame in getTrainingData(basePath):
        logging.debug(trainingDataFrame)
        
# test fetching training data.

def test_fetching_training_data(basePath:str):
    from conscious_struct import trainModelWithDataBasePath, TestEnqueue
    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)