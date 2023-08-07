# from collections import namedtuple
try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict
try:
    from typing import Literal
except:
    from typing import Literal
try:
    from typing import NamedTuple
except:
    from typing_extensions import NamedTuple
import numpy as np
from typing import Union, cast, overload
# import logging
from log_utils import logger

class HIDStruct(TypedDict):
    HIDEvents: list

class TrainingFrame(NamedTuple):
    datatype: Literal['hid','image']
    data: Union[HIDStruct, np.ndarray]

# we just need the basepath.
def getTrainingData(basePath: str):
    import os
    hid_timestamp_path = os.path.join(basePath,"hid_timestamps.json")
    video_timestamp_path = os.path.join(basePath,"video_timestamps.json")

    video_path = os.path.join(basePath,"video_record.mp4")
    hid_rec_path = os.path.join(basePath,"hid_record.jsonl")
    
    import json

    import cv2
    import jsonlines

    video_cap = cv2.VideoCapture(video_path)
    # breakpoint()
    # 318 frames? only got 266 timestamps!
    frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info("FRAME COUNT: %d", frame_count)

    def load_json(filename):
        with open(filename, "r") as f:
            return json.load(f)

    hid_timestamp = load_json(hid_timestamp_path)
    video_timestamp = load_json(video_timestamp_path)

    from typing import List, Union

    import numpy as np

    def getVideoFrameIndexSynced(
        x: Union[List[int], np.ndarray],
        y: Union[List[int], np.ndarray],
        EPS: float = 1e-10,
    ) -> List[int]:
        """

        Notes:
            All input arrays and output array are positive and increasing.

        Params:
            x: Actual video frame indexes.
            y: Index list to be synced against.

        Output:
            x_: Synced frame indexs. (len(x_) == len(y))
        """
        x_ = np.linspace(x[0], x[-1] + (1 - EPS), len(y))
        x_ = np.floor(x_).astype(int).tolist()
        return x_

    hidseq = np.zeros(shape=(2, len(hid_timestamp))) - 1
    hidseq[0] = np.array(range(len(hid_timestamp)))

    videoseq = np.zeros(shape=(2, len(video_timestamp))) - 1

    # videoseq[1] = np.array(range(len(video_timestamp)))
    index_list_to_be_synced_against = np.array(range(len(video_timestamp)))
    actual_video_frame_indexs = np.array(range(int(frame_count)))

    videoseq[1] = getVideoFrameIndexSynced(
        actual_video_frame_indexs, index_list_to_be_synced_against
    )

    seq = np.hstack((hidseq, videoseq))
    logger.info("SEQ SHAPE: %s", seq.shape)

    timeseq = np.array(hid_timestamp + video_timestamp)
    sorted_indexes = np.argsort(timeseq)

    sorted_seq = seq[:, sorted_indexes].T.astype(int)
    # print(sorted_seq)

    # now, attempt to parse them.
    hid_data_list = []
    with open(hid_rec_path, "r") as f:
        jsonl_reader = jsonlines.Reader(f)
        while True:
            try:
                hid_data = jsonl_reader.read()
                hid_data_list.append(hid_data)
            except:
                break

    # maybe you should "yield" data through these iterators.

    NO_CONTENT = -1

    suc, frame = video_cap.read()
    frame_index_cursor = 0

    for hid_index, frame_index in sorted_seq:
        logger.debug("HID INDEX: %d, FRAME INDEX: %d", hid_index, frame_index)
        assert not all(
            [e == NO_CONTENT for e in [hid_index, frame_index]]
        ), "at least one type of content is active"
        assert not all(
            [e != NO_CONTENT for e in [hid_index, frame_index]]
        ), "cannot have two types of active content sharing the same index"
        if hid_index != NO_CONTENT:
            hid_data = hid_data_list[hid_index]
            logger.debug("HID DATA: %s", hid_data)
            yield TrainingFrame(datatype='hid', data=cast(HIDStruct, hid_data))
        elif frame_index != NO_CONTENT:
            while frame_index_cursor != frame_index:
                suc, frame = video_cap.read()
                frame_index_cursor += 1
            assert (
                suc
            ), f"Video '{video_path}' failed to read frame #{frame_index} (index starting from zero)"
            logger.debug("FRAME SHAPE: %s", frame.shape)
            yield TrainingFrame(datatype='image', data=frame)
            # cv2.imshow("win", frame)
            # cv2.waitKey(1)
        else:
            raise Exception("Something impossible has happened.")

    # breakpoint()
    video_cap.release()
    # success, frame = video_cap.read()
    # print(frame.shape) # (768, 1280, 3)
