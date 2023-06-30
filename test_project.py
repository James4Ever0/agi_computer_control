
# test fetching training data.
from conscious_struct import trainModelWithDataBasePath, TestEnqueue

def test_fetching_training_data():
    basePath = "recordings/2023-06-02T07_59_45.711256/"
    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)