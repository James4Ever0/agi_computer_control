# you are expected to receive mouse/keyboard events
#
serverPort = 4471

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable, Any, Dict, Union, Literal

origins = ["*"]
app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=origins, allow_headers=origins
)
from pydantic import BaseModel


class MouseEventData(BaseModel):
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


class KeyboardEventData(BaseModel):
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


class EventPayload(BaseModel):
    eventType: str
    data: dict


class MouseEventPayload(BaseModel):
    eventType: Literal["mousedown", "mouseup", "mousemove"]
    data: MouseEventData


class KeyboardEventPayload(BaseModel):
    eventType: Literal["keydown", "keyup"]
    # eventType: Literal['keydown', 'keyup', 'keypress']
    data: KeyboardEventData


# tradeoff when using string types: you need to call update_forward_refs()
# import uuid

class EventModel(BaseModel):
    # eventType: str
    timestamp: float
    # timestamp: str
    # data: str
    client_id: str

class ScreenshotEvent(EventModel):
    screenshot_data: str
    # screenshot_data: str

class BrowserEvent(EventModel):
    # client_id: uuid.UUID
    # payload: Union[MouseEventPayload, KeyboardEventPayload]
    payload: Union[MouseEventPayload, KeyboardEventPayload, EventPayload]
    # payload: EventPayload


# obviously not right.
#    "data": {
#     "isTrusted": true
# }

from fastapi import Request, Response
from fastapi.routing import APIRoute

# from log_utils import terminal_column_size
terminal_column_size = 80
import json


class ValidationErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            # except RequestValidationError as exc:
            except Exception as e:
                is_json = False
                try:
                    body = await request.json()
                    body = json.dumps(body, indent=4, ensure_ascii=False)
                    is_json = True
                except:
                    body = await request.body()
                print(
                    "request{}".format("_json" if is_json else "")
                    .upper()
                    .center(terminal_column_size, "_"),
                    body,
                    sep="\n",
                )
                print(
                    "exception".upper().center(terminal_column_size, "_"), e, sep="\n"
                )
                # detail = {"errors": exc.errors(), "body": body.decode()}
                # raise HTTPException(status_code=422, detail=detail)
                raise e

        return custom_route_handler


app.router.route_class = ValidationErrorLoggingRoute

# sample_data_path = 'sample_event_data.json'
# sample_data = {}
# @app.get('/getIdentifier')
# def get_identifier(client_id:str):
#     return dict(client_id=client_id)

STATUS_OK_RESPONSE = {'status': 'ok'}

@app.post("/submitScreenshot")
def receiveScreenshotEvent(request_data: ScreenshotEvent):
    return STATUS_OK_RESPONSE

@app.post("/browserInputEvent")
# def receiveBrowserInputEvent(body:Dict[str, Any]):
# def receiveBrowserInputEvent(
#     eventType: str,
#     timestamp: str,
#     data: str,
# ):
def receiveBrowserInputEvent(request_data: BrowserEvent):
    # print("received body:", eventType, timestamp, data)
    print("received body:", request_data)
    # eventType = request_data.payload.eventType
    # data = request_data.payload.data
    # sample_data[eventType] = data
    # if len(sample_data.keys()) == 5:
    #     with open(sample_data_path, 'w+') as f:
    #         f.write(json.dumps(sample_data, ensure_ascii=False, indent=4))
    #     print("sample data saved to", sample_data_path)
    #     exit(0)
    return STATUS_OK_RESPONSE


if __name__ == "__main__":
    import uvicorn

    print("server address: http://localhost:%d" % serverPort)
    uvicorn.run(app, host="0.0.0.0", port=serverPort)
