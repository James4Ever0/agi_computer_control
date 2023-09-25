import fastapi

app = fastapi.FastAPI()
import datetime
from beat_server_config import beat_server_address, beat_client_data

import pytz
# with respect to our dearly Py3.6
timezone_str = "Asia/Shanghai"
# timezone = pytz.timezone(timezone_str:='Asia/Shanghai')
timezone = pytz.timezone(timezone_str)

@app.get(beat_server_address['beat_url'])
def beat_request(uuid: str):
    strtime = datetime.datetime.now(tz=timezone).strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"received beat request from {uuid} at time {strtime}")
    return {beat_client_data['access_time_key']: strtime}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **beat_server_address)
