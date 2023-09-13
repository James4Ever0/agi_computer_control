from bytes_actor import BytesActor
from predictive_alpine_actor import PredictiveAlpineActor, run_actor_forever


class PredictiveAlpineBytesActor(BytesActor, PredictiveAlpineActor):
    ...


if __name__ == "__main__":
    run_actor_forever(PredictiveAlpineBytesActor)
