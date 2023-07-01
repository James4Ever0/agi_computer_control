# use "--log-level" in pytest.

import sys

sys.path.append("../")

import logging

filename = "mytest.log"

# may log to other places.
# infinite append.
from logging.handlers import RotatingFileHandler

myHandler = RotatingFileHandler(
    filename, maxBytes=1024 * 1024 * 3, backupCount=3, encoding="utf-8"
)

myHandler.setLevel(logging.INFO) # will it log less things?
myHandler.setFormatter(myFormatter)

logging.basicConfig(
    # filename=filename,
    level=logging.getLogger().getEffectiveLevel(),
    # stream=sys.stderr,
    force=True,
    handlers=[myHandler],
)

# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
from recording_train_parse import getTrainingData
logging.critical("")
import datetime
current_time = datetime.datetime.now().isoformat()
logging.critical("logging starts: {}")
import pytest


@pytest.fixture()
def basePath():
    return "../recordings/2023-06-02T07_59_45.711256/"


def test_get_training_data(basePath: str):
    for trainingDataFrame in getTrainingData(basePath):
        logging.debug("training data frame: %s", trainingDataFrame)


# test fetching training data.


def test_fetching_training_data(basePath: str):
    from conscious_struct import trainModelWithDataBasePath, TestEnqueue

    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)
