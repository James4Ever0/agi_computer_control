import fastapi

app = fastapi.FastAPI()
import datetime

@app.get('/beat_request')
def beat_request(uuid:str):
    strtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"received beat request from {uuid} at time {strtime}")
    return strtime

if __name__ == "__main__":
    port = 8981
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)