# use "--log-level" in pytest.

# with this limited model structure, you may not find it "evolving".
# you must let the AI design itself, evolve on its own.

import sys

sys.path.append("../")

import logging
from typing import Generator, Union, AsyncGenerator, AwaitableGenerator
import torch

filename = "mytest.log"

# may log to other places.
# infinite append.
from logging.handlers import RotatingFileHandler
from logging import StreamHandler

stdout_handler = StreamHandler(sys.stdout)


from conscious_struct import (
        trainModelWithDataBasePath,
        Trainer,
        SequentialTrainingQueue,
        CustomModel
)

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
    handlers=[myHandler, stdout_handler],
)

# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
from recording_train_parse import getTrainingData

# logging.critical("")
import datetime

current_time = datetime.datetime.now().isoformat()
logging.critical(f"logging starts: {current_time}".center(100, "="))
# logging.critical("")
import pytest

# TODO: modify function at source code level, not here!
# def auto_teardown(func):
#     def inner_func(*args, **kwargs):
#         val = func(*args, **kwargs)
#         if not isinstance(val, Union[Generator, AsyncGenerator, AwaitableGenerator]):
#             yield val
#             del val
#         return val
#     return inner_func

@pytest.fixture(scope='session')
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

import os
from pathlib import Path

@pytest.fixture(scope='session')
def vit_model_path():
    path = Path(os.path.abspath(relpath:="../../../model_cache/vit_b_16-c867db91.pth")) 
    if not path.exists():
        raise Exception(f"Current directory: {os.curdir}\nModel weight does not exist: {path}")
    # return "/Volumes/Toshiba XG3/model_cache/vit_b_16-c867db91.pth"
    return path

@pytest.fixture(scope='session')
def vit_model(vit_model_path:str):
    import torchvision

    # code from OA bot
    # return torchvision.models.vit_b_16(pretrained=True)
    vmodel = torchvision.models.vit_b_16()
    # breakpoint()
    mStateDict = torch.load(vit_model_path)
    vmodel.load_state_dict(mStateDict)
    yield vmodel
    del vmodel

from torchvision.models import VisionTransformer
@pytest.fixture(scope='session')
def model(vit_model:VisionTransformer):
    model = CustomModel(vit_model)
    yield model
    del model

# def pretrained_model_path():
#     path = ...
#     return path

# @pytest.fixture(scope='session')
# def model_pretrained(model:CustomModel,pretrained_model_path:str):
#     model.load_state_dict(torch.load(pretrained_model_path))
#     yield model
#     del model

# you don't need the model to be trained at all to act.

@pytest.fixture(scope='session')
def loss_fn():
    from torch.nn import CrossEntropyLoss
    
    loss = CrossEntropyLoss(reduction="mean")
    yield loss
    del loss

@pytest.fixture(scope='session')
def optimizer(model:CustomModel):
    from torch.optim import Adam
    lr = 0.00001
    opt = Adam(model.parameters(), lr=lr)
    yield opt
    del opt

from hypothesis import given, settings
from hypothesis.strategies import integers
# from hypothesis import HealthCheck

import stopit
@given(random_seed=integers())
# @settings(suppress_health_check=(HealthCheck.function_scoped_fixture,),max_examples = 10, deadline=None)
@settings(deadline=None, max_examples=2)
def test_train_model_with_training_data(model:CustomModel, loss_fn, optimizer, basePath:str, random_seed:int):
    # TODO: annotate our code with "nptyping" & "torchtyping" | "jaxtyping"
    # TODO: haskell? functional python?
    # (variadic types) ref: https://peps.python.org/pep-0646/
    # use sympy for symbolic checks?

    context_length = 2
    batch_size = 1

    myTrainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    myQueue = SequentialTrainingQueue(
        context_length=context_length, batch_size=batch_size, trainer=myTrainer
    )
    # TODO: allow timeout exception to be raised, disallow any other exceptions.
    # you might want to shuffle its order, for testing.

    with stopit.ThreadingTimeout(5): # timeout exception suppressed!
        trainModelWithDataBasePath(basePath, myQueue, shuffle_for_test=True, random_seed=random_seed)
    print("SESSION TIMEOUT NOW".center(60,"_"))

def test_act_with_model(model: CustomModel):
    ...
    # do not load any weight yet.