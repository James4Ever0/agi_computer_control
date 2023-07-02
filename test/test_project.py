# use "--log-level" in pytest.

# with this limited model structure, you may not find it "evolving".
# you must let the AI design itself, evolve on its own.

import sys

sys.path.append("../")

import logging
import torch

filename = "mytest.log"

# may log to other places.
# infinite append.
from logging.handlers import RotatingFileHandler


from conscious_struct import (
        trainModelWithDataBasePath,
        Trainer,
        SequentialTrainingQueue,
        CustomModel,)
myHandler = RotatingFileHandler(
    filename, maxBytes=1024 * 1024 * 3, backupCount=3, encoding="utf-8"
)
myHandler.setLevel(logging.DEBUG)
# myHandler.setLevel(logging.INFO) # will it log less things? yes.
FORMAT = (
    "<%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s() ] %(message)s"
)
# FORMAT = "<%(name)s:%(levelname)s> [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
myFormatter = logging.Formatter(fmt=FORMAT)
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

# logging.critical("")
import datetime

current_time = datetime.datetime.now().isoformat()
logging.critical(f"logging starts: {current_time}".center(100, "="))
# logging.critical("")
import pytest


@pytest.fixture
def basePath():
    return "../recordings/2023-06-02T07_59_45.711256/"


def test_get_training_data(basePath:str):
    for trainingDataFrame in getTrainingData(basePath):
        logging.debug("training data frame: %s", trainingDataFrame)


# test fetching training data.


def test_fetching_training_data(basePath:str):
    from conscious_struct import trainModelWithDataBasePath, TestEnqueue

    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)

@pytest.fixture
def vit_model_path():
    return "/Volumes/Toshiba XG3/model_cache/vit_b_16-c867db91.pth"

@pytest.fixture
def vit_model(vit_model_path:str):
    import torchvision

    # code from OA bot
    # return torchvision.models.vit_b_16(pretrained=True)
    vmodel = torchvision.models.vit_b_16()
    mStateDict = torch.load(vit_model_path)
    vmodel.load_state_dict(mStateDict)
    return vmodel

from torchvision.models import VisionTransformer
@pytest.fixture
def model(vit_model:VisionTransformer):
    model = CustomModel(vit_model)
    return model

@pytest.fixture
def loss_fn():
    from torch.nn import CrossEntropyLoss
    
    return CrossEntropyLoss(reduction="mean")

@pytest.fixture
def optimizer(model:CustomModel):
    from torch.optim import Adam
    lr = 0.00001
    return Adam(model.parameters(), lr=lr)

def test_train_model_with_training_data(model:CustomModel, loss_fn, optimizer, basePath:str):

    context_length = 10
    batch_size = 3

    myTrainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    myQueue = SequentialTrainingQueue(
        context_length=context_length, batch_size=batch_size, trainer=myTrainer
    )
    trainModelWithDataBasePath(basePath, myQueue)
