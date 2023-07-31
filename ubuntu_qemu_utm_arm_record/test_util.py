from utils import TimestampedContext

with TimestampedContext("abc") as f:
    f.commit()
    shit = 1 / 0
