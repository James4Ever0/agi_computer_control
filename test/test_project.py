# use "--log-level" in pytest.

# with this limited model structure, you may not find it "evolving".
# you must let the AI design itself, evolve on its own.

import sys
import numpy as np

sys.path.append("../")

# import logging
from log_utils import logger_print

# import log_utils
# it is been commented out.
# from typing import Generator, Union, AsyncGenerator
# try:
#     from typing import AwaitableGenerator
# except:
#     from typing_extensions import AwaitableGenerator
import torch

# may log to other places.
# infinite append.
# from logging import StreamHandler

# stdout_handler = StreamHandler(sys.stdout)


from conscious_struct import (
    trainModelWithDataBasePath,
    Trainer,
    SequentialTrainingQueue,
    CustomModel,
    ConsciousFlow,  # consists of `ConsciousBlock`
    ConsciousBlock,
    HIDAction,
    ConsciousBase,
)

# logging.basicConfig(
#     # filename=filename,
#     level=logging.getLogger().getEffectiveLevel(),
#     # stream=sys.stderr,
#     force=True,
#     handlers=[myHandler, stdout_handler],
# )

# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
from recording_train_parse import getTrainingData

# logging.critical("")
import datetime

current_time = datetime.datetime.now().isoformat()
logger_print(f"logging starts: {current_time}".center(100, "="))
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


@pytest.fixture(scope="session")
def basePath():
    return "../recordings/2023-06-02T07_59_45.711256/"


def test_get_training_data(basePath: str):
    for trainingDataFrame in getTrainingData(basePath):
        logger_print("training data frame:", trainingDataFrame)


# test fetching training data.


def test_fetching_training_data(basePath: str):
    from conscious_struct import trainModelWithDataBasePath, TestEnqueue

    myQueue = TestEnqueue()
    # fake sequentialqueue.
    trainModelWithDataBasePath(basePath, myQueue)


import os
from pathlib import Path


@pytest.fixture(scope="session")
def vit_model_path():
    path = Path(
        os.path.abspath(relpath := "../../../model_cache/vit_b_16-c867db91.pth")
    )
    if not path.exists():
        raise Exception(
            f"Current directory: {os.curdir}\nModel weight does not exist: {path}"
        )
    # return "/Volumes/Toshiba XG3/model_cache/vit_b_16-c867db91.pth"
    return path


@pytest.fixture(scope="session")
def vit_model(vit_model_path: str):
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


@pytest.fixture(scope="session")
def model(vit_model: VisionTransformer):
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


@pytest.fixture(scope="session")
def loss_fn():
    from torch.nn import CrossEntropyLoss

    loss = CrossEntropyLoss(reduction="mean")
    yield loss
    del loss


@pytest.fixture(scope="session")
def optimizer(model: CustomModel):
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
def test_train_model_with_training_data(
    model: CustomModel, loss_fn, optimizer, basePath: str, random_seed: int
):
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

    with stopit.ThreadingTimeout(5):  # timeout exception suppressed!
        trainModelWithDataBasePath(
            basePath, myQueue, shuffle_for_test=True, random_seed=random_seed
        )
    logger_print("SESSION TIMEOUT NOW".center(60, "_"))


import einops
from conscious_struct import KeyPress, KeyRelease, MouseMove, MouseScroll, MouseClick


@pytest.mark.parametrize(
    "HIDActionObj",
    [
        MouseClick(x=993, y=659, button="Button.left", pressed=True),
        MouseMove(x=10, y=20),
        MouseScroll(x=10, y=10, dx=10, dy=-10),
        KeyPress(key="""'9'"""),
        KeyRelease(key="""'8'"""),
    ],
)
def test_eval_with_model(model: CustomModel, HIDActionObj):
    model.eval()
    # it is been observed by video recording script.
    max_x, max_y = 1280, 768
    HIDActionJson = HIDActionObj.to_list()
    # HIDActionJsonList = [HIDActionObj.to_list() for HIDActionObj in HIDActionObjList]
    actionData = HIDAction.from_action_json(
        HIDActionJson, max_x=max_x, max_y=max_y
    ).to_ndarray()
    # actionDataList = [HIDAction.from_action_json(
    #     HIDActionJson, max_x=max_x, max_y=max_y
    # ).to_ndarray() for HIDActionJson in HIDActionJsonList]
    randomImageData = np.random.random(
        (ConsciousBase.image_channels, ConsciousBase.image_dim, ConsciousBase.image_dim)
    )
    # imageData = einops.pack(randomImageData, "*")  # just what shape shall this be?
    imageData = einops.rearrange(randomImageData, "c h w -> (c h w)")
    # actionConsciousBlocks = [ConsciousBlock(
    #     data_type="HIDAction", special_token=None, action_data=actionData
    # ) for actionData in actionDataList]
    actionConsciousBlock = ConsciousBlock(
        data_type="HIDAction", special_token=None, action_data=actionData
    )
    imageConsciousBlock = ConsciousBlock(
        data_type="image", special_token=None, image_data=imageData
    )
    # cs = ConsciousFlow(consciousBlocks=[*actionConsciousBlocks, imageConsciousBlock])
    cf = ConsciousFlow(consciousBlocks=[ # not here but inside.
        actionConsciousBlock, 
        # imageConsciousBlock,
        ]
    )
    # cs = ConsciousFlow(consciousBlocks=[actionConsciousBlock, imageConsciousBlock])
    # must be 3d, not 2d.
    # print(actionConsciousBlock.to_tensor().shape) # torch.Size([154639]) ~ d
    # print(imageConsciousBlock.to_tensor().shape) # torch.Size([154639]) ~ d
    # print(cf.to_tensor().shape) # torch.Size([2, 154639]) ~ s d
    # breakpoint()
    result = model.forward(conscious_stream=einops.batchedcf.to_tensor()) # b s d
    logger_print(result)
    # do not load any weight yet. just use its random state.
    # do not execute anything in this test! just get the predicted things out.
