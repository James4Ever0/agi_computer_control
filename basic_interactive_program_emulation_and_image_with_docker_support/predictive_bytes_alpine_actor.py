from predictive_alpine_actor import PredictiveAlpineActor, run_actor_forever
from bytes_actor import BytesActor

class PredictiveAlpineBytesActor(BytesActor, PredictiveAlpineActor):
    ...
if __name__ == "__main__":
    run_actor_forever(PredictiveAlpineBytesActor)