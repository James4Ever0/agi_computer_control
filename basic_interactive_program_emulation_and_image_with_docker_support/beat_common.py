beat_server_address = dict(
    host="0.0.0.0",
    port=(server_port := 8771),
    beat_url="/beat_request",
    info_url="/info",
    url_prefix=f"http://localhost:{server_port}",
)
beat_client_data = dict(
    url=(f"{beat_server_address['url_prefix']}{beat_server_address['beat_url']}"),
    info_url=f"{beat_server_address['url_prefix']}{beat_server_address['info_url']}",
    timeout=2,
    access_time_key="access_time",
)

from functools import lru_cache, wraps
from time import monotonic_ns

# use cacheout instead.
# ref: https://github.com/dgilland/cacheout
def timed_lru_cache(
    _func=None, *, seconds: int = 7000, maxsize: int = 128, typed: bool = False
):
    """Extension over existing lru_cache with timeout
    :param seconds: timeout value
    :param maxsize: maximum size of the cache
    :param typed: whether different keys for different types of cache keys
    """

    def wrapper_cache(f):
        # create a function wrapped with traditional lru_cache
        f = lru_cache(maxsize=maxsize, typed=typed)(f)
        # convert seconds to nanoseconds to set the expiry time in nanoseconds
        f.delta = seconds * 10**9
        f.expiration = monotonic_ns() + f.delta

        @wraps(f)  # wraps is used to access the decorated function attributes
        def wrapped_f(*args, **kwargs):
            if monotonic_ns() >= f.expiration:
                # if the current cache expired of the decorated function then
                # clear cache for that function and set a new cache value with new expiration time
                f.cache_clear()
                f.expiration = monotonic_ns() + f.delta
            return f(*args, **kwargs)

        wrapped_f.cache_info = f.cache_info
        wrapped_f.cache_clear = f.cache_clear
        return wrapped_f

    # To allow decorator to be used without arguments
    if _func is None:
        return wrapper_cache
    else:
        return wrapper_cache(_func)


import requests

# from frozendict import frozendict
# create a session object
session = requests.Session()  # effectively faster. really?


@timed_lru_cache(seconds=1, maxsize=1)
def heartbeat_base(uuid: str, action: str, pid: int, role: str):
    return heartbeat_base_nocache(uuid, action, pid, role)


def heartbeat_base_nocache(uuid: str, action: str, pid: int, role: str):
    params = dict(uuid=uuid, action=action, pid=pid, role=role)
    url = beat_client_data["url"]
    data = request_with_timeout_and_get_json_data(params, url)
    access_time = data[beat_client_data["access_time_key"]]
    return access_time


def query_info():
    return request_with_timeout_and_get_json_data(dict(), beat_client_data["info_url"])


def request_with_timeout_and_get_json_data(params: dict, url: str, success_code=200):
    r = session.get(url, params=params, timeout=beat_client_data["timeout"])
    assert r.status_code == success_code
    data = r.json()
    return data
