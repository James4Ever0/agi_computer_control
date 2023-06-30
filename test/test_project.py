
"""Creating IES topology type system.

Usage:
    type_system_v2.py [(-p | --plot_only) | --version]
    
Options:
    -l --level       Level of debug. 
"""

from docopt import docopt


options = docopt(__doc__, version="1.0")
import logging

level = options.get('--level', )
import sys
sys.path.append("../")

logging.basicConfig(level=level, )
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