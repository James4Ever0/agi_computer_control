# you are expected to receive mouse/keyboard events

serverPort = 4471

import fastapi
from fastapi.middleware.cors import CORSMiddleware

origins = ['*']
app = fastapi.FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods = origins, allow_headers = origins)
from pydantic import BaseModel


class BrowserEvent(BaseModel):
    eventType: str
    timestamp: str
    data: str
    # data: dict


# obviously not right.
#    "data": {
#     "isTrusted": true
# }

from fastapi import Request, Response
from fastapi.routing import APIRoute
from typing import Callable

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


# from typing import Any, Dict
@app.post("/browserInputEvent")
# def receiveBrowserInputEvent(body:Dict[str, Any]):
# def receiveBrowserInputEvent(
#     eventType: str,
#     timestamp: str,
#     data: str,
# ):
def receiveBrowserInputEvent(request_data:BrowserEvent):
    # print("received body:", eventType, timestamp, data)
    print("received body:", request_data)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print("server address: http://localhost:%d" % serverPort)
    uvicorn.run(app, host="0.0.0.0", port=serverPort)
