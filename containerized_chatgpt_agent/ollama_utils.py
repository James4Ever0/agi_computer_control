# TODO: post this patch as issue to litellm
import litellm.llms.ollama as ollama
import litellm
import copy
import os

max_token_count_env_key = "LITELLM_MAX_TOKEN_COUNT"
old_completion = copy.copy(litellm.completion)

from contextlib import contextmanager


@contextmanager
def set_max_token_count_env_context(max_tokens):
    orig_env = os.environ.get(max_token_count_env_key, None)
    if max_tokens:
        print("setting max tokens env:", max_tokens)
        os.environ[max_token_count_env_key] = str(max_tokens)
    else:
        print("not setting max token count")
    try:
        yield
    finally:
        if isinstance(orig_env, str):
            os.environ[max_token_count_env_key] = orig_env
        else:
            os.environ.pop(max_token_count_env_key, None)
        print("recovered max token count:", orig_env)


def new_completion(*args, **kwargs):
    max_tokens = kwargs.get("max_tokens", None)
    print("kwarg max tokens:", max_tokens)
    with set_max_token_count_env_context(max_tokens):
        ret = old_completion(*args, **kwargs)
    return ret


litellm.completion = new_completion


def get_max_token_from_environ():
    count = os.environ.get(max_token_count_env_key, None)
    print("getted count:", count)
    if count:
        count = int(count)
    else:
        count = float("inf")
    return count


old_get_ollama_response_stream = copy.copy(ollama.get_ollama_response_stream)

import progressbar


def new_get_ollama_response_stream(*args, **kwargs):
    old_generator = old_get_ollama_response_stream(*args, **kwargs)
    max_token_count = get_max_token_from_environ()
    total_count = 0

    # Create a new progress bar instance
    bar = progressbar.ProgressBar(max_value=max_token_count)

    for chunk in old_generator:
        piece = chunk["choices"][0]["delta"]["content"]
        piece_token_count = len(litellm.encoding.encode(piece))
        total_count += piece_token_count
        bar.update(min(total_count, max_token_count))
        # print(
        #     "received:",
        #     piece_token_count,
        #     "total:",
        #     total_count,
        #     "max:",
        #     max_token_count,
        # )
        yield chunk
        if total_count > max_token_count:
            break


ollama.get_ollama_response_stream = new_get_ollama_response_stream
