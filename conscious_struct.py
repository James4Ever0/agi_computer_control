# import pynput
# no such dependency when training.
import einops
import os
import numpy as np
import cv2
import ast
from pydantic import BaseModel, validator
from typing import Union, Mapping, List

# import logging
from log_utils import logger
from pydantic_numpy import NDArray
import torch

try:
    from typing import Literal
except:
    from typing_extensions import Literal  # this is a failsafe.


try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

##############
#  HID BASE  #
##############


class HIDActionTypes:
    keyboard_action_types: TypeAlias = Literal[
        "key_press",
        "key_release",
    ]
    mouse_action_types: TypeAlias = Literal[
        "mouse_move",
        "mouse_click",
        "mouse_scroll",
    ]
    action_types: TypeAlias = Literal[
        keyboard_action_types,
        mouse_action_types,
    ]
    mouse_buttons: TypeAlias = Literal[
        "Button.left",
        "Button.middle",
        "Button.right",
    ]
    keys: TypeAlias = Literal[
        """','""",
        """'.'""",
        """'/'""",
        """';'""",
        """\"'\"""",
        """'['""",
        """']'""",
        """'\\'""",
        """'='""",
        """'-'""",
        """'0'""",
        """'9'""",
        """'8'""",
        """'7'""",
        """'6'""",
        """'5'""",
        """'4'""",
        """'3'""",
        """'2'""",
        """'1'""",
        """'`'""",
        """'a'""",
        """'b'""",
        """'c'""",
        """'d'""",
        """'e'""",
        """'f'""",
        """'g'""",
        """'h'""",
        """'i'""",
        """'j'""",
        """'k'""",
        """'l'""",
        """'m'""",
        """'n'""",
        """'o'""",
        """'p'""",
        """'q'""",
        """'r'""",
        """'s'""",
        """'t'""",
        """'u'""",
        """'v'""",
        """'w'""",
        """'x'""",
        """'y'""",
        """'z'""",
        "Key.alt",
        "Key.alt",
        "Key.alt_r",
        "Key.alt_r",
        "Key.backspace",
        "Key.caps_lock",
        "Key.cmd",
        "Key.cmd",
        "Key.cmd_r",
        "Key.ctrl",
        "Key.ctrl",
        "Key.ctrl_r",
        "Key.delete",
        "Key.down",
        "Key.end",
        "Key.enter",
        "Key.esc",
        "Key.f1",
        "Key.f2",
        "Key.f3",
        "Key.f4",
        "Key.f5",
        "Key.f6",
        "Key.f7",
        "Key.f8",
        "Key.f9",
        "Key.f10",
        "Key.f11",
        "Key.f12",
        "Key.f13",
        "Key.f14",
        "Key.f15",
        "Key.f16",
        "Key.f17",
        "Key.f18",
        "Key.f19",
        "Key.f20",
        "Key.home",
        "Key.left",
        "Key.page_down",
        "Key.page_up",
        "Key.right",
        "Key.shift",
        "Key.shift",
        "Key.shift_r",
        "Key.space",
        "Key.tab",
        "Key.up",
        "Key.media_play_pause",
        "Key.media_volume_mute",
        "Key.media_volume_down",
        "Key.media_volume_up",
        "Key.media_previous",
        "Key.media_next",
    ]


class HIDActionBase:
    mouse_resolution: int = 1000
    keyboard_action_types = list(HIDActionTypes.keyboard_action_types.__args__)
    mouse_action_types = list(HIDActionTypes.mouse_action_types.__args__)
    action_types = list(HIDActionTypes.action_types.__args__)
    mouse_buttons = list(HIDActionTypes.mouse_buttons.__args__)
    keys = list(HIDActionTypes.keys.__args__)

    length = (
        len(action_types)
        + len(keys)
        + len(mouse_buttons)
        + 1  # mouse pressed
        + 4 * mouse_resolution
    )  # ,
    #             1)

    @staticmethod
    def unshift_keycode(keycode: str) -> Union[str, None]:
        unshift_keycodes = {
            "!": "1",
            "@": "2",
            "#": "3",
            "$": "4",
            "%": "5",
            "^": "6",
            "&": "7",
            "*": "8",
            "(": "9",
            ")": "0",
            "_": "-",
            "+": "=",
            "{": "[",
            "}": "]",
            "|": "\\",
            ":": ";",
            '"': "'",
            "<": ",",
            ">": ".",
            "?": "/",
            "~": "`",
        }
        ctrl_keycodes = {
            "\x01": "a",
            "\x02": "b",
            "\x03": "c",
            "\x04": "d",
            "\x05": "e",
            "\x06": "f",
            "\x07": "g",
            "\x08": "h",
            "\t": "i",
            "\n": "j",
            "\x0b": "k",
            "\x0c": "l",
            "\r": "m",
            "\x0e": "n",
            "\x0f": "o",
            "\x10": "p",
            "\x11": "q",
            "\x12": "r",
            "\x13": "s",
            "\x14": "t",
            "\x15": "u",
            "\x16": "v",
            "\x17": "w",
            "\x18": "x",
            "\x19": "y",
            "\x1a": "z",
            "<219>": "[",
            "<221>": "]",
            "<189>": "-",
            "<187>": "=",
            "<192>": "`",
            "<48>": "0",
            "<49>": "1",
            "<50>": "2",
            "<51>": "3",
            "<52>": "4",
            "<53>": "5",
            "<54>": "6",
            "<55>": "7",
            "<56>": "8",
            "<57>": "9",
            "<220>": "\\",
            "<186>": ";",
            "<222>": "'",
            "<188>": ",",
            "<190>": ".",
            "<191>": "/",
        }
        keycode = unshift_keycodes.get(keycode, ctrl_keycodes.get(keycode, keycode))
        # still, this is something out of concern.
        if keycode.startswith("<") and keycode.endswith(">"):
            logger.warning("Discarding unconvertable keycode: %s" % keycode)
            # keycode = pynput.keyboard.KeyCode(int(keycode[1:-1]))
            return
        return keycode

    @staticmethod
    def uncover_keycode(keycode: str) -> Union[str, None]:
        if not keycode.startswith("Key."):
            keycode_converted = HIDActionBase.unshift_keycode(
                keycode
                if keycode.startswith("<") and keycode.endswith(">")
                else ast.literal_eval(keycode)
            )
            return keycode_converted
            # this could be None.
            # when this is None, simply skip this code. do not end the conversion. skip it.
        else:
            return keycode


class HIDAction(BaseModel, HIDActionBase):
    # static method: from_action
    # static method: from_ndarray
    # instance method: to_ndarray
    # instance method: to_action
    max_x: int
    max_y: int
    action_type: Literal[
        "key_press",  # ["key_press", "'w'"]
        "key_release",  # ["key_release", "'r'"]
        "mouse_move",  # ["mouse_move", [176.7734375, 580.40625]], "timeStamp": 1680247557.125498}
        "mouse_click",  # ["mouse_click", [176.7734375, 580.40625, "Button.left", true]]
        "mouse_scroll",  # ["mouse_scroll", [938.76171875, 318.75, 0, 0]]
        #         None,  # end_of_action
    ]  # you need to specify this.
    key: Union[
        HIDActionTypes.keys,
        None,
    ] = None
    mouse_button: Union[HIDActionTypes.mouse_buttons, None] = None
    mouse_pressed: Union[bool, None] = None
    x: Union[float, None] = None
    y: Union[float, None] = None
    dx: Union[float, None] = None
    dy: Union[float, None] = None

    @validator("max_x", "max_y")
    def greater_than_zero(cls, v):
        assert type(v) == int
        assert v > 0
        return v

    @validator("action_type")
    def action_type_within_action_types(cls, v):
        if v:
            assert v in HIDActionBase.action_types
        return v

    @validator("key")
    def key_within_keys(cls, v):
        if v:
            assert v in HIDActionBase.keys
        return v

    @validator("mouse_button")
    def mouse_button_within_mouse_buttons(cls, v):
        if v:
            assert v in HIDActionBase.mouse_buttons
        return v

    @validator("mouse_pressed")
    def mouse_pressed_type_check(cls, v):
        if v:
            assert type(v) == bool
        return v

    @staticmethod
    def from_action_json(action_json: list, max_x: int, max_y: int):
        action_type = action_json[0]
        action_args = action_json[1]

        construct_args = dict(max_x=max_x, max_y=max_y, action_type=action_type)

        # BUG: convert single char keys to quoted format.
        # TODO: make sure ' ' is converted into Key.Space
        if action_type.startswith("key"):
            if len(action_args) == 1:
                if action_args != "'":
                    action_args = f"'{action_args}'"
                else:
                    action_args = f'"{action_args}"'
            if action_args == repr(" "):
                action_args = "Key.space"

        if action_type == "key_press":
            assert action_args in HIDActionBase.keys

            construct_args.update(dict(key=action_args))
        elif action_type == "key_release":
            assert action_args in HIDActionBase.keys

            construct_args.update(dict(key=action_args))
        elif action_type == "mouse_move":
            assert action_args[0] >= 0 and action_args[0] <= max_x
            assert action_args[1] >= 0 and action_args[1] <= max_y

            construct_args.update(dict(x=action_args[0], y=action_args[1]))
        elif action_type == "mouse_click":
            assert action_args[0] >= 0 and action_args[0] <= max_x
            assert action_args[1] >= 0 and action_args[1] <= max_y
            assert action_args[2] in HIDActionBase.mouse_buttons
            assert type(action_args[3]) == bool

            construct_args.update(
                dict(
                    x=action_args[0],
                    y=action_args[1],
                    mouse_button=action_args[2],
                    mouse_pressed=action_args[3],
                )
            )
        elif action_type == "mouse_scroll":
            assert action_args[0] >= 0 and action_args[0] <= max_x
            assert action_args[1] >= 0 and action_args[1] <= max_y
            assert action_args[2] >= -max_x and action_args[2] <= max_x
            assert action_args[3] >= -max_y and action_args[3] <= max_y

            construct_args.update(
                dict(
                    x=action_args[0],
                    y=action_args[1],
                    dx=action_args[2],
                    dy=action_args[3],
                )
            )
        else:
            raise Exception(
                "Unknown action type: %s\naction args: %s" % (action_type, action_args)
            )

        mHIDAction = HIDAction(**construct_args)
        return mHIDAction

    @staticmethod
    def from_ndarray(ndarray: np.ndarray, max_x: int, max_y: int):
        assert ndarray.shape == (HIDActionBase.length,)
        cursor = 0

        action_type_ndarray = ndarray[cursor : cursor + len(HIDActionBase.action_types)]
        cursor += len(HIDActionBase.action_types)
        action_type_index = np.argmax(action_type_ndarray)
        action_type = HIDActionBase.action_types[action_type_index]
        del action_type_ndarray
        del action_type_index

        construct_args = dict(max_x=max_x, max_y=max_y, action_type=action_type)

        if action_type:
            key_ndarray = ndarray[cursor : cursor + len(HIDActionBase.keys)]
            cursor += len(HIDActionBase.keys)
            key_index = np.argmax(key_ndarray)
            key = HIDActionBase.keys[key_index]
            del key_ndarray
            del key_index

            mouse_button_ndarray = ndarray[
                cursor : cursor + len(HIDActionBase.mouse_buttons)
            ]
            cursor += len(HIDActionBase.mouse_buttons)
            mouse_button_index = np.argmax(mouse_button_ndarray)
            mouse_button = HIDActionBase.mouse_buttons[mouse_button_index]
            del mouse_button_ndarray
            del mouse_button_index

            mouse_pressed_ndarray = ndarray[cursor : cursor + 1]
            cursor += 1
            mouse_pressed = bool(mouse_pressed_ndarray[0][0])
            del mouse_pressed_ndarray

            x_ndarray = ndarray[cursor : cursor + HIDActionBase.mouse_resolution]
            cursor += HIDActionBase.mouse_resolution
            x_index = np.argmax(x_ndarray)
            x = (x_index / HIDActionBase.mouse_resolution) * max_x
            del x_ndarray
            del x_index

            y_ndarray = ndarray[cursor : cursor + HIDActionBase.mouse_resolution]
            cursor += HIDActionBase.mouse_resolution
            y_index = np.argmax(y_ndarray)
            y = (y_index / HIDActionBase.mouse_resolution) * max_y
            del y_ndarray
            del y_index

            dx_ndarray = ndarray[cursor : cursor + HIDActionBase.mouse_resolution]
            cursor += HIDActionBase.mouse_resolution
            dx_index = np.argmax(dx_ndarray)
            dx = (dx_index / HIDActionBase.mouse_resolution) * 2 * max_x - max_x
            del dx_ndarray
            del dx_index

            dy_ndarray = ndarray[cursor : cursor + HIDActionBase.mouse_resolution]
            cursor += HIDActionBase.mouse_resolution
            dy_index = np.argmax(dy_ndarray)
            dy = (dy_index / HIDActionBase.mouse_resolution) * 2 * max_y - max_y
            del dy_ndarray
            del dy_index

            if action_type == "key_press":
                construct_args.update(dict(key=key))
            elif action_type == "key_release":
                construct_args.update(dict(key=key))
            elif action_type == "mouse_move":
                construct_args.update(dict(x=x, y=y))
            elif action_type == "mouse_click":
                construct_args.update(
                    dict(
                        x=x, y=y, mouse_button=mouse_button, mouse_pressed=mouse_pressed
                    )
                )
            elif action_type == "mouse_scroll":
                construct_args.update(dict(x=x, y=y, dx=dx, dy=dy))
        else:
            pass

        del cursor

        mHIDAction = HIDAction(**construct_args)
        return mHIDAction

    def round_within(self, number: Union[int, float], number_name: str) -> int:
        result = round(number)
        if result > self.mouse_resolution:
            logger.warning(f"Warning: {number_name} overflow")
            logger.warning(f"Value {result} greater than {self.mouse_resolution}")
            return self.mouse_resolution
        elif result < 0:
            logger.warning(f"Warning: {number_name} overflow")
            logger.warning(f"Value {result} smaller than 0")
            return 0
        return result

    def to_ndarray(
        self,
    ) -> np.ndarray:
        action_type_ndarray = np.zeros((len(self.action_types), 1))
        action_type_ndarray[self.action_types.index(self.action_type)] = 1

        key_ndarray = np.zeros((len(self.keys), 1))
        if self.key:
            key_ndarray[self.keys.index(self.key)] = 1

        mouse_button_ndarray = np.zeros((len(self.mouse_buttons), 1))
        if self.mouse_button:
            mouse_button_ndarray[self.mouse_buttons.index(self.mouse_button)] = 1

        mouse_pressed_array = np.zeros((1, 1))
        if self.mouse_pressed:
            mouse_pressed_array[0] = 1

        x_ndarray = np.zeros((self.mouse_resolution, 1))
        if self.x:
            x_ndarray[
                self.round_within(self.mouse_resolution * self.x / self.max_x, "X")
            ] = 1

        y_ndarray = np.zeros((self.mouse_resolution, 1))
        if self.y:
            y_ndarray[
                self.round_within(self.mouse_resolution * self.y / self.max_y, "Y")
            ] = 1

        dx_ndarray = np.zeros((self.mouse_resolution, 1))
        if self.dx:
            dx_ndarray[
                self.round_within(
                    self.mouse_resolution * (self.dx + self.max_x) / (2 * self.max_x),
                    "DX",
                )
            ] = 1

        dy_ndarray = np.zeros((self.mouse_resolution, 1))
        if self.dy:
            dy_ndarray[
                self.round_within(
                    self.mouse_resolution * (self.dy + self.max_y) / (2 * self.max_y),
                    "DY",
                )
            ] = 1

        ndarray = np.concatenate(
            [
                action_type_ndarray,
                key_ndarray,
                mouse_button_ndarray,
                mouse_pressed_array,
                x_ndarray,
                y_ndarray,
                dx_ndarray,
                dy_ndarray,
            ]
        )
        return ndarray

    def to_action_json(
        self,
    ) -> Union[list, None]:
        action_type = self.action_type
        if action_type:
            if action_type == "key_press":
                assert self.key in self.keys

                action_args = self.key
            elif action_type == "key_release":
                assert self.key in self.keys

                action_args = self.key
            elif action_type == "mouse_move":
                assert self.x >= 0 and self.x <= self.max_x
                assert self.y >= 0 and self.y <= self.max_y

                action_args = [self.x, self.y]
            elif action_type == "mouse_click":
                assert self.x >= 0 and self.x <= self.max_x
                assert self.y >= 0 and self.y <= self.max_y
                assert self.mouse_button in self.mouse_buttons
                assert type(self.mouse_pressed) == bool

                action_args = [self.x, self.y, self.mouse_button, self.mouse_pressed]
            elif action_type == "mouse_scroll":
                assert self.x >= 0 and self.x <= self.max_x
                assert self.y >= 0 and self.y <= self.max_y
                assert self.dx >= -self.max_x and self.dx <= self.max_x
                assert self.dy >= -self.max_y and self.dy <= self.max_y

                action_args = [self.x, self.y, self.dx, self.dy]
            else:
                raise Exception("Unknown action_type: %s" % action_type)
            action_json = [action_type, action_args]
        else:
            action_json = None
        return action_json


#########################
#  HID DATA VALIDATION  #
#########################

from pydantic import confloat


class KeyPress(BaseModel):
    _action_type = "key_press"

    key: HIDActionTypes.keys

    def to_list(self) -> List:
        return [self._action_type, self.key]

    @classmethod
    def from_list(cls, lst: List):
        action_type = lst[0]
        action_args = [lst[1]]
        assert len(action_args) == 1
        assert action_type == cls._action_type

        assert isinstance(action_args[0], HIDActionTypes.keys)

        return cls(key=action_args[0])


class KeyRelease(BaseModel):
    _action_type = "key_release"

    key: HIDActionTypes.keys

    def to_list(self) -> List:
        return [self._action_type, self.key]

    @classmethod
    def from_list(cls, lst: List):
        action_type = lst[0]
        action_args = [lst[1]]
        assert len(action_args) == 1
        assert action_type == cls._action_type

        assert isinstance(action_args[0], HIDActionTypes.keys)

        return cls(key=action_args[0])


class MouseClick(BaseModel):
    _action_type = "mouse_click"

    x: confloat(ge=0)
    y: confloat(ge=0)
    button: HIDActionTypes.mouse_buttons
    pressed: bool

    def to_list(self) -> List:
        return [self._action_type, [self.x, self.y, self.button, self.pressed]]

    @classmethod
    def from_list(cls, lst: List):
        action_type = lst[0]
        action_args = lst[1]
        assert len(action_args) == 4
        assert action_type == cls._action_type

        assert isinstance(action_args[0], confloat(ge=0))
        assert isinstance(action_args[1], confloat(ge=0))
        assert isinstance(action_args[2], HIDActionTypes.mouse_buttons)
        assert isinstance(action_args[3], bool)

        return cls(
            x=action_args[0],
            y=action_args[1],
            button=action_args[2],
            pressed=action_args[3],
        )


class MouseMove(BaseModel):
    _action_type = "mouse_move"

    x: confloat(ge=0)
    y: confloat(ge=0)

    def to_list(self) -> List:
        return [self._action_type, [self.x, self.y]]

    @classmethod
    def from_list(cls, lst: List):
        action_type = lst[0]
        action_args = lst[1]
        assert len(action_args) == 2
        assert action_type == cls._action_type

        assert isinstance(action_args[0], confloat(ge=0))
        assert isinstance(action_args[1], confloat(ge=0))

        return cls(x=action_args[0], y=action_args[1])


class MouseScroll(BaseModel):
    _action_type = "mouse_scroll"

    x: confloat(ge=0)
    y: confloat(ge=0)
    dx: float
    dy: float

    def to_list(self) -> List:
        return [self._action_type, [self.x, self.y, self.dx, self.dy]]

    @classmethod
    def from_list(cls, lst: List):
        action_type = lst[0]
        action_args = lst[1]
        assert len(action_args) == 4
        assert action_type == cls._action_type

        assert isinstance(action_args[0], confloat(ge=0))
        assert isinstance(action_args[1], confloat(ge=0))
        assert isinstance(action_args[2], float)
        assert isinstance(action_args[3], float)

        return cls(
            x=action_args[0], y=action_args[1], dx=action_args[2], dy=action_args[3]
        )


#################
# VIDEO CONTEXT #
#################


class VideoCaptureContextManager:
    def __init__(self, videoPath):
        self.videoPath = videoPath

    def __enter__(self):
        logger.info("Entering the context...")
        self.cap = cv2.VideoCapture(self.videoPath)
        return self.cap

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            self.cap.release()
        finally:
            import gc

            gc.collect()
            logger.info("Leaving the context...")
        #  print(exc_type, exc_value, exc_tb, sep="\n")


##################
# CONSCIOUS BASE #
##################


class ConsciousBase:
    data_types = ["image", "HIDAction"]
    special_tokens = ["image_newline", "image_end", "action_end", None]
    #     vector_size = 1+2+1000+4110 # visual things are pre-encoded. no raw image here!
    # vector size is "length" now

    image_dim = 224
    image_channels = 3

    data_type_length = len(data_types)
    special_token_length = len(special_tokens)
    image_length = image_dim * image_dim * image_channels  # obviously flattened.

    # FIX 1: plus to colon.
    split_sizes = [
        len(data_types),
        len(special_tokens),
        image_length,  # FIX 9: change to flattened image bits count
        HIDActionBase.length,  # 4110?
    ]
    length = sum(split_sizes)

    # you cannot easily revert this compression by argmax or something else.
    # so you need the decoder.


# can it be consciousnessless?


class ConsciousBlock(BaseModel, ConsciousBase):
    data_type: Union[Literal["image"], Literal["HIDAction"]]  # 2 bits, required
    special_token: Union[
        Literal["image_newline"],
        Literal["image_end"],
        Literal[
            "action_end"
        ],  # change some of these bits into -torch.inf, so you won't have paradox like results.
        None,
    ] = None  # 4 bits
    image_data: Union[
        None, NDArray
    ] = None  # what is the shape of this image data? assume to be [3,224,224] (c h w) flattened
    action_data: Union[None, NDArray] = None  # assume to be: (4110, )
    # [1,1000] -> [3,1000,1000] -> [3,224,224]
    #    einsum.repeat       conv2d

    # so, maybe you still need some ViT decode layer.

    @staticmethod
    def from_json(data: Mapping):
        mConsciousBlock = ConsciousBlock(**data)
        return mConsciousBlock

    def to_json(self) -> Mapping:
        mJson = self.dict()
        return mJson

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        # check its shape.
        assert tensor.shape == (ConsciousBlock.length,)
        split_sizes = ConsciousBase.split_sizes
        data_bits, special_bits, image_data, action_data = einops.unpack(
            tensor, [[s] for s in split_sizes], "*"
        )
        #         data_bits, special_bits, image_data, action_data = size_splits(tensor, split_sizes)

        data_type = ConsciousBase.data_types[int(torch.argmax(data_bits))]
        if data_type == "image":
            special_bits[2] = -torch.inf
            special_token = ConsciousBase.special_tokens[
                int(torch.argmax(special_bits))
            ]

            mConsciousBlock = ConsciousBlock(
                data_type=data_type, special_token=special_token, image_data=image_data
            )
        elif data_type == "HIDAction":
            special_bits[0:2] = -torch.inf
            special_token = ConsciousBase.special_tokens[
                int(torch.argmax(special_bits))
            ]

            mConsciousBlock = ConsciousBlock(
                data_type=data_type,
                special_token=special_token,
                action_data=action_data,
            )
        else:
            raise Exception("Unknown data_type:", data_type)
        return mConsciousBlock

    def to_tensor(self) -> torch.Tensor:
        mTensorBase = torch.zeros((ConsciousBase.length))
        # BUG 1: not enough values to unpack
        #         print(ConsciousBase.length)
        #         print(ConsciousBase.split_sizes)
        data_bits, special_bits, image_data, action_data = einops.unpack(
            mTensorBase, [[s] for s in ConsciousBase.split_sizes], "*"
        )
        #         data_bits, special_bits, image_data, action_data = size_splits(mTensorBase, ConsciousBase.split_sizes)
        data_bits[ConsciousBase.data_types.index(self.data_type)] = 1
        if self.data_type == "image":
            # BUG 2: comparing ndarray to None
            # FIX 2: change "!=" into "is not"
            assert self.image_data is not None
            logger.debug("Image data shape: %s", self.image_data.shape)
            logger.debug(
                "Expected data shape: %s",
                expected_data_shape := (ConsciousBase.image_length,),
            )
            assert self.image_data.shape == expected_data_shape
            assert self.special_token != "action_end"

            image_data = torch.Tensor(self.image_data)
            special_bits[ConsciousBase.special_tokens.index(self.special_token)] = 1

        elif self.data_type == "HIDAction":
            assert self.action_data is not None
            if len(self.action_data.shape) > 1:
                self.action_data = self.action_data.reshape((-1,))
            logger.debug("Action data shape: %s", self.action_data.shape)
            # BUG: actual: (4110, 1)
            # shall we reshape this.
            logger.debug(
                "Expected data shape: %s",
                expected_data_shape := (HIDActionBase.length,),
            )  # TODO: expected shape is (4110, )? how to make this typed?
            assert self.action_data.shape == expected_data_shape
            assert self.special_token not in ["image_newline", "image_end"]

            action_data = torch.Tensor(self.action_data)
            special_bits[ConsciousBase.special_tokens.index(self.special_token)] = 1

        mTensor, _ = einops.pack(
            (data_bits, special_bits, image_data, action_data), "*"
        )
        #         mTensor = torch.concat((data_bits, special_bits, image_data, action_data))
        del mTensorBase
        return mTensor


class ConsciousFlow(BaseModel, ConsciousBase):
    consciousBlocks: List[ConsciousBlock]

    @staticmethod
    def from_json(data: List[Mapping]):
        mList = [ConsciousBlock.from_json(j) for j in data]
        mConsciousFlow = ConsciousFlow(consciousBlocks=mList)
        return mConsciousFlow

    def to_json(self) -> List[Mapping]:
        mJson = [c.to_json() for c in self.consciousBlocks]
        return mJson

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        consciousBlockCount, vector_length = tensor.shape
        assert vector_length == ConsciousBase.length
        mConsciousBlocks = []
        for i in range(consciousBlockCount):
            arr = tensor[i, :]
            mConsciousBlock = ConsciousBlock.from_tensor(arr)
            mConsciousBlocks.append(mConsciousBlock)
        mConsciousFlow = ConsciousFlow(consciousBlocks=mConsciousBlocks)
        return mConsciousFlow

    def to_tensor(self) -> torch.Tensor:
        mTensor, _ = einops.pack([c.to_tensor() for c in self.consciousBlocks], "* d")
        #         mTensor = torch.Tensor([c.to_tensor() for c in self.consciousBlocks])
        return mTensor


#####################
# TRAINER & DATASET #
#####################


class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer  # shall be registered to model parameters.

    def step(self, batched_input, batched_output=None):
        # BUG 8: model forward keyword error
        # FIX 8: fixing keyword, adding default keyword argument
        model_output = self.model.forward(batched_input, target_output=batched_output)
        loss = self.loss_fn(model_output, batched_output)
        logger.debug("LOSS? %s", loss)

        # this loss is incorrect. shall use some argmax stuff.
        # to ensure that this thing is the thing that we want.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


from typing import Protocol


class Enqueue(Protocol):
    def enqueue(self, data, /):
        ...

    def clear(self):
        ...


class TestEnqueue(Enqueue):
    def __init__(self):
        ...
        # self.queue = []

    def enqueue(self, data):
        logger.debug("DATA QUEUE: %s", data)
        # may you print the data shape.
        if data.action_data is not None:
            logger.debug("ACTION DATA SHAPE: %s", data.action_data.shape)
        elif data.image_data is not None:
            logger.debug("IMAGE DATA SHAPE: %s", data.image_data.shape)
        logger.debug("")

    def clear(self):
        ...


class SequentialTrainingQueue:
    def __init__(self, context_length: int, batch_size: int, trainer: Trainer):
        self.context_length = context_length
        self.batch_size = batch_size

        self.trainer = trainer

        self.max_critical_length = self.context_length + self.batch_size
        self.min_critical_length = self.context_length + 1

        self.consciousVectors = []

    def enqueue(self, consciousBlock: ConsciousBlock, clear: bool = False):
        # change that into some tensor first.

        # BUG 3: consciousBlock has tensor output but not numpy ndarray
        # FIX 3: find and replace all "consciousBlock.to_nparray()" with "consciousBlock.to_tensor()"
        #         print(consciousBlock)
        #         print(type(consciousBlock))
        consciousVector = consciousBlock.to_tensor()
        logger.debug(
            "CONSCIOUS VECTOR [TYPE: %s SHAPE: %s]",
            type(consciousVector),
            consciousVector.shape,
        )
        self.consciousVectors.append(consciousVector)

        if not clear:
            # BUG 5: no "max_critical_point"
            # FIX 5: replace it with "max_critical_length"
            if len(self.consciousVectors) == self.max_critical_length:
                self.train()
        else:
            self.clear()

    def train(self, clear: bool = False):
        # TODO: use `ConsciousFlow` to replace logic here.
        # train the model and clear the queue.
        # size of queue before: self.context_length+self.batch_size (should be? at least geq to self.length+1)
        # size of queue after training: self.context_length

        if len(self.consciousVectors) >= self.min_critical_length:
            batch_size = len(self.consciousVectors) - self.context_length
            # BUG 6: missing self.
            # FIX 6: adding self.

            # BUG 7: torch.Tensor conversion error
            # FIX 7: change "torch.Tensor([])" into einops.pack

            #             print(self.consciousVectors)
            batched_input, _ = einops.pack(
                [
                    #             batched_input = torch.Tensor([
                    einops.pack(
                        self.consciousVectors[i : i + self.context_length], "* d"
                    )[0]
                    for i in range(batch_size)
                    #                     torch.Tensor(self.consciousVectors[i:i+self.context_length]) for i in range(batch_size)
                ],
                "* s d",
            )
            batched_output, _ = einops.pack(
                [
                    #             batched_output = torch.Tensor([
                    self.consciousVectors[self.context_length + i]
                    for i in range(batch_size)
                ],
                "* d",
            )
            self.trainer.step(batched_input, batched_output)

            if not clear:
                # now remove some elements.
                self.consciousVectors = self.consciousVectors[-self.context_length :]
            else:
                self.consciousVectors = []

    def clear(self):
        # check if anything left in queue. call at the end of queue.
        self.train(clear=True)


###############
# IMAGE UTILS #
###############


def resizeImage(im, desired_size):
    # im = cv2.imread(im_pth)
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


class SequentialEvalQueue:
    # loop: init data (visual+hid actions) -> predict till end or limit -> update visual and action data
    # visual data are inserted in a regular basis.
    # human action data/bot action data will be inserted in between visual data.
    # limit max machine predict token count per interval to 5. forcing it to end anyway.
    # what about the machine trying to spit out some visual prediction?
    # we just shadow it. (do not act! just dream. compare to current visual ground truth and perform gradient descent. maybe you can use that as continuous training basis? or redesign the model so it can choose to skip (by retrievable positional encoding) some blocks of visual data and perform descent only on selected area?)
    ...


################
# READING DATA #
################

desired_size = 224 * 4

# get perspective width/height with:
# pyautogui.size()
# must be on the same machine.

# with VideoCaptureContextManager(videoPath) as cap:
from recording_train_parse import getTrainingData
import json
import re
import parse
import random


# this process is actually training it.
def trainModelWithDataBasePath(
    basePath: str,
    sequentialTrainingQueue: Enqueue,
    shuffle_for_test: bool = False,
    random_seed: int = 43  # pass hypothesis param here!
    # sequentialTrainingQueue: SequentialTrainingQueue
):
    if shuffle_for_test:
        random.seed(random_seed)
    # read perspective width & height from basepath.
    fpath = os.path.join(basePath, "video_record_script.sh")
    with open(fpath, "r") as f:
        data = f.read()
        # data = json.load(f)
        parse_target = re.finditer(r"\b\d+x\d+\b", data).__next__().group()
        parsed_data = parse.parse(
            "{perspective_width:d}x{perspective_height:d}", parse_target
        )
        if parsed_data:
            perspective_width, perspective_height = (
                parsed_data["perspective_width"],
                parsed_data["perspective_height"],
            )
        else:
            raise Exception(f"Cannot parse perspective size from file: {fpath}")
    trainingDataGenerator = getTrainingData(basePath)

    if shuffle_for_test:
        myGenerator = list(trainingDataGenerator)
        random.shuffle(myGenerator)

    for trainingDataFrame in trainingDataGenerator:
        if trainingDataFrame.datatype == "hid":
            encoded_actions = []
            actions = trainingDataFrame.data["HIDEvents"]
            if shuffle_for_test:
                random.shuffle(actions)
            for action in actions:
                action_type, action_args = action
                if action_type in HIDActionBase.keyboard_action_types:
                    action_args = HIDActionBase.uncover_keycode(action_args)
                    if action_args is None:
                        logger.warning("Skipping: %s", action)
                        continue
                mHIDAction = HIDAction.from_action_json(
                    [action_type, action_args],
                    max_x=perspective_width,
                    max_y=perspective_height,
                )  # related to mouse coordinates.
                mHIDActionNDArray = mHIDAction.to_ndarray()
                logger.debug("HID NDArray shape: %s", mHIDActionNDArray.shape)
                encoded_actions.append(mHIDActionNDArray)

            for index, EA in enumerate(encoded_actions):
                st = None
                if index + 1 == len(encoded_actions):
                    st = "action_end"
                consciousBlock = ConsciousBlock(
                    data_type="HIDAction", special_token=st, action_data=EA
                )
                sequentialTrainingQueue.enqueue(consciousBlock)
        elif trainingDataFrame.datatype == "image":
            image = trainingDataFrame.data

            image_resized = resizeImage(image, desired_size)
            image_reshaped = einops.rearrange(image_resized, "h w c -> c h w")
            #             image_reshaped = np.rollaxis(image_resized, 2, 0) # (3, 896, 896)
            image_sliced = [
                image_reshaped[:, x * 224 : (x + 1) * 224, y * 224 : (y + 1) * 224]
                for x in range(4)
                for y in range(4)
            ]  # ) # c h w

            # IMAGE RESHAPED: (896, 3, 896)?
            # IMAGE RESHAPED: (896, 896, 3)
            #             print('IMAGE RESHAPED:', image_reshaped.shape)
            #             print('IMAGE SLICED:', image_sliced.shape)
            #     (16, 3, 224, 224)
            # hell?
            consciousBlocks = []
            for index, im in enumerate(image_sliced):
                im = einops.rearrange(im, "c h w -> (c h w)")
                st = None
                if index == 15:
                    st = "image_end"
                elif index != 0 and (index + 1) % 4 == 0:
                    st = "image_newline"

                # BUG 4: tuple
                # FIX 4: remove .to_tensor() method call
                consciousBlock = ConsciousBlock(
                    data_type="image", special_token=st, image_data=im
                )
                if shuffle_for_test:
                    consciousBlocks.append(consciousBlock)
                else:
                    #                 print(consciousBlock)
                    sequentialTrainingQueue.enqueue(consciousBlock)
            #             last_output = torch.zeros(1, output_size)
            if shuffle_for_test:
                for consciousBlock in consciousBlocks:
                    sequentialTrainingQueue.enqueue(consciousBlock)
            del image
            del image_reshaped
            del image_resized
        else:
            assert False, f"wrong datatype: {trainingDataFrame.datatype}"
    sequentialTrainingQueue.clear()


#########################################
#  CONSISTENCY WITH RECORDER AND ACTOR  #
#########################################

# import ctypes

# PROCESS_PER_MONITOR_DPI_AWARE = 2

# ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

#########
# MODEL #
#########

# notice: when in online mode only image will be backpropagated.
# like using some upside down mirror.


class CustomModel(torch.nn.Module):
    def __init__(self, vit_model, hidden_size_vit=1000, vit_block_size=228):
        #     def __init__(self, rwkv_model, vit_model, tokenizer, hidden_size_rwkv, hidden_size_vit, output_size, vit_times = 4, vit_block_size=228):
        super(CustomModel, self).__init__()
        #         self.rwkv_model = rwkv_model # processing language, generate actions.
        self.vit_model = vit_model
        self.vit_block_size = vit_block_size  # this is default.
        #         self.vit_times = vit_times

        # seq2seq alike.
        #         self.hidden_size = hidden_size_rwkv+hidden_size_vit
        self.hidden_size = hidden_size_vit

        self.HIDEncoder = torch.nn.Linear(HIDActionBase.length, 1000)
        self.HIDDecoder = torch.nn.Linear(
            1000, HIDActionBase.length
        )  # use torch.where or something
        # sparse matrix?

        # some magic going elsewhere.
        self.ViTDecoder = torch.nn.Conv2d(
            in_channels=9, out_channels=3, kernel_size=22, stride=4, dilation=5
        )  # n c h w

        # FIX 13: change 1000 to ConsciousBase.data_type_length+ConsciousBase.special_token_length+1000
        io_h_size = (
            ConsciousBase.data_type_length + ConsciousBase.special_token_length + 1000
        )
        self.rnn = torch.nn.LSTM(
            input_size=io_h_size, hidden_size=io_h_size, batch_first=True
        )

        # use tensor.

    def forward(
        self,
        conscious_stream: torch.Tensor,
        target_output: Union[None, torch.Tensor] = None,
    ):
        # if have target, and our datatype bits are wrong, we only optimize the datatype bits, making image/action bits identical
        # otherwise just calculate all bits from the model.

        # conscious_stream: [batch_size, (ConsciousBase.length) data_type+special_tokens+image_bits+action_bits]
        # you need another pydantic model for it.

        # BUG 9: input shape error
        # FIX 9: check input and output shape
        #         print(conscious_stream.shape, ConsciousBase.length, )
        #         if target_output is not None:
        #             print(target_output.shape)
        batch_size, seq_length, dim_block = conscious_stream.shape
        assert dim_block == ConsciousBase.length

        conscious_stream_reshaped = einops.rearrange(
            conscious_stream, "b s d -> (b s) d"
        )

        datatype_bits, special_bits, image_bits, action_bits = einops.unpack(
            conscious_stream_reshaped, [[s] for s in ConsciousBase.split_sizes], "b *"
        )
        #         datatype_bits, special_bits, image_bits, action_bits = size_splits(conscious_stream_reshaped, ConsciousBase.split_sizes,dim=1)
        #         desired_size = self.vit_block_size*self.vit_times

        datatypes = torch.argmax(datatype_bits, dim=1)
        image_indexs = datatypes == 0
        action_indexs = datatypes == 1

        special_tokens = torch.argmax(special_bits, dim=1)
        image_newline_indexs = special_tokens == 0
        image_end_indexs = special_tokens == 1
        action_end_indexs = special_tokens == 2
        nop_indexs = special_tokens == 3

        # 4+2+1000

        # you are gonna take actions.
        # prepare some zeros.

        # BUG 10: len() on int
        # FIX 10: find and fix incorrect len() calls
        batched_rnn_input = torch.zeros(
            (
                batch_size * seq_length,
                ConsciousBase.data_type_length
                + ConsciousBase.special_token_length
                + 1000,
            )
        )

        batched_rnn_input[image_indexs, 0] = 1
        batched_rnn_input[action_indexs, 1] = 1

        # BUG 11: indexs not defined
        # FIX 11: indexes -> indexs
        batched_rnn_input[image_newline_indexs, 2] = 1
        batched_rnn_input[image_end_indexs, 3] = 1
        batched_rnn_input[action_end_indexs, 4] = 1
        batched_rnn_input[nop_indexs, 5] = 1

        # process this.
        datatype_and_special_token_length = (
            ConsciousBase.data_type_length + ConsciousBase.special_token_length
        )

        # BUG 12: unannotated unknown axes in einops.rearrange
        # FIX 12: annotate these axes
        selected_image_bits = image_bits[image_indexs, :]
        transformed_image_bits = einops.rearrange(
            selected_image_bits,
            "b (c h w) -> b c h w",
            h=ConsciousBase.image_dim,
            w=ConsciousBase.image_dim,
            c=ConsciousBase.image_channels,
        )
        processed_image_bits = self.vit_model(transformed_image_bits)
        batched_rnn_input[
            image_indexs, datatype_and_special_token_length:
        ] = processed_image_bits

        selected_action_bits = action_bits[action_indexs, :]
        processed_action_bits = self.HIDEncoder(selected_action_bits)
        batched_rnn_input[
            action_indexs, datatype_and_special_token_length:
        ] = processed_action_bits

        batched_rnn_input_reshaped = einops.rearrange(
            batched_rnn_input, "(b s) d -> b s d", b=batch_size, s=seq_length
        )

        # BUG 13: mismatched shape for rnn
        _, (h1, c1) = self.rnn(batched_rnn_input_reshaped)

        # shape of h1: [batch_size, dim_block]

        if target_output is not None:
            (
                target_datatype_bits,
                target_special_bits,
                target_image_bits,
                target_action_bits,
            ) = einops.unpack(
                target_output, [[s] for s in ConsciousBase.split_sizes], "b *"
            )

        output_datatype_bits, output_special_bits, output_data_bits = einops.unpack(
            einops.rearrange(h1, "b s d -> (b s) d"),
            [
                [s]
                for s in [
                    ConsciousBase.data_type_length,
                    ConsciousBase.special_token_length,
                    1000,
                ]
            ],
            "b *",
        )
        #         output_datatype_bits, output_special_bits, output_data_bits = size_splits(h1, [ConsciousBase.data_type_length, ConsciousBase.special_token_length, 1000] ,dim=1)
        output_datatypes = torch.argmax(output_datatype_bits, dim=1)

        output_image_indexs = output_datatypes == 0
        output_action_indexs = output_datatypes == 1

        if target_output is not None:
            target_output_datatypes = torch.argmax(target_datatype_bits, dim=1)

            target_output_image_indexs = target_output_datatypes == 0
            target_output_action_indexs = target_output_datatypes == 1

            common_output_image_indexs = torch.logical_and(
                target_output_image_indexs, output_image_indexs
            )
            common_output_action_indexs = torch.logical_and(
                target_output_action_indexs, output_action_indexs
            )

            common_output_indexs = torch.logical_or(
                common_output_image_indexs, common_output_action_indexs
            )

            target_exclusive_output_image_indexs = torch.logical_and(
                target_output_image_indexs, torch.logical_not(output_image_indexs)
            )
            target_exclusive_output_action_indexs = torch.logical_and(
                target_output_action_indexs, torch.logical_not(output_action_indexs)
            )

            target_exclusive_output_indexs = torch.logical_or(
                target_exclusive_output_image_indexs,
                target_exclusive_output_action_indexs,
            )

        if target_output is not None:
            selected_output_image_bits = output_data_bits[common_output_image_indexs, :]
        else:
            selected_output_image_bits = output_data_bits[output_image_indexs, :]
        processed_output_image_bits = einops.repeat(
            selected_output_image_bits, "b d -> b d 9"
        )
        processed_output_image_bits = einops.einsum(
            processed_output_image_bits,
            processed_output_image_bits,
            "b h c, b w c -> b c h w",
        )
        processed_output_image_bits = self.ViTDecoder(processed_output_image_bits)
        processed_output_image_bits = einops.rearrange(
            processed_output_image_bits, "b c h w -> b (c h w)"
        )

        if target_output is not None:
            selected_output_action_bits = output_data_bits[
                common_output_action_indexs, :
            ]
        else:
            selected_output_action_bits = output_data_bits[output_action_indexs, :]
        processed_output_action_bits = self.HIDDecoder(selected_output_action_bits)

        # preparing blank output
        output_0 = torch.zeros((batch_size, ConsciousBase.length))

        (
            output_datatype_bits_0,
            output_special_token_bits_0,
            output_image_bits_0,
            output_action_bits_0,
        ) = einops.unpack(output_0, [[s] for s in ConsciousBase.split_sizes], "b *")
        #         output_datatype_bits_0, output_special_token_bits_0, output_image_bits_0, output_action_bits_0 = size_splits(output, ConsciousBase.split_sizes, dim=1)

        if target_output is not None:
            output_datatype_bits_0[
                target_exclusive_output_indexs, :
            ] = target_datatype_bits[target_exclusive_output_indexs, :]
            output_datatype_bits_0[common_output_indexs, :] = output_datatype_bits[
                common_output_indexs, :
            ]

            output_special_token_bits_0[
                target_exclusive_output_indexs, :
            ] = output_special_bits[target_exclusive_output_indexs, :]
            output_special_token_bits_0[common_output_indexs, :] = output_special_bits[
                common_output_indexs, :
            ]

            output_special_token_bits_0[common_output_image_indexs, 2] = 0
            output_special_token_bits_0[common_output_action_indexs, :2] = 0

            output_image_bits_0[
                target_exclusive_output_image_indexs, :
            ] = target_image_bits[target_exclusive_output_image_indexs, :]
            output_action_bits_0[
                target_exclusive_output_action_indexs, :
            ] = target_action_bits[target_exclusive_output_action_indexs, :]

            output_image_bits_0[
                common_output_image_indexs, :
            ] = processed_output_image_bits
            output_action_bits_0[
                common_output_action_indexs, :
            ] = processed_output_action_bits
        else:
            output_datatype_bits_0 = output_datatype_bits

            output_special_bits_0 = output_special_bits

            output_special_bits_0[output_image_indexs, 2] = 0
            output_special_bits_0[output_action_indexs, :2] = 0

            output_image_bits_0[output_image_indexs, :] = processed_output_image_bits
            output_action_bits_0[output_action_indexs, :] = processed_output_action_bits

        output, _ = einops.pack(
            (
                output_datatype_bits_0,
                output_special_token_bits_0,
                output_image_bits_0,
                output_action_bits_0,
            ),
            "b *",
        )
        #         output = torch.concat((output_datatype_bits_0, output_special_token_bits_0, output_image_bits_0, output_action_bits_0), dim=1)

        return output
