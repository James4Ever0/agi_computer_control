from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

class Mousemove(BaseModel):
    screenX: int
    screenY: int
    clientX: int
    clientY: int
    ctrlKey: bool
    shiftKey: bool
    altKey: bool
    metaKey: bool
    button: int
    buttons: int
    relatedTarget: Any
    pageX: int
    pageY: int
    x: int
    y: int
    offsetX: int
    offsetY: int
    movementX: int
    movementY: int
    fromElement: Any
    toElement: Dict[str, Any]
    layerX: int
    layerY: int


class Mousedown(BaseModel):
    screenX: int
    screenY: int
    clientX: int
    clientY: int
    ctrlKey: bool
    shiftKey: bool
    altKey: bool
    metaKey: bool
    button: int
    buttons: int
    relatedTarget: Any
    pageX: int
    pageY: int
    x: int
    y: int
    offsetX: int
    offsetY: int
    movementX: int
    movementY: int
    fromElement: Any
    toElement: Dict[str, Any]
    layerX: int
    layerY: int


class Mouseup(BaseModel):
    screenX: int
    screenY: int
    clientX: int
    clientY: int
    ctrlKey: bool
    shiftKey: bool
    altKey: bool
    metaKey: bool
    button: int
    buttons: int
    relatedTarget: Any
    pageX: int
    pageY: int
    x: int
    y: int
    offsetX: int
    offsetY: int
    movementX: int
    movementY: int
    fromElement: Any
    toElement: Dict[str, Any]
    layerX: int
    layerY: int


class Keydown(BaseModel):
    key: str
    code: str
    location: int
    ctrlKey: bool
    shiftKey: bool
    altKey: bool
    metaKey: bool
    repeat: bool
    isComposing: bool
    charCode: int
    keyCode: int
    DOM_KEY_LOCATION_STANDARD: int
    DOM_KEY_LOCATION_LEFT: int
    DOM_KEY_LOCATION_RIGHT: int
    DOM_KEY_LOCATION_NUMPAD: int


class Keyup(BaseModel):
    key: str
    code: str
    location: int
    ctrlKey: bool
    shiftKey: bool
    altKey: bool
    metaKey: bool
    repeat: bool
    isComposing: bool
    charCode: int
    keyCode: int
    DOM_KEY_LOCATION_STANDARD: int
    DOM_KEY_LOCATION_LEFT: int
    DOM_KEY_LOCATION_RIGHT: int
    DOM_KEY_LOCATION_NUMPAD: int


class Model(BaseModel):
    mousemove: Mousemove
    mousedown: Mousedown
    mouseup: Mouseup
    keydown: Keydown
    keyup: Keyup
