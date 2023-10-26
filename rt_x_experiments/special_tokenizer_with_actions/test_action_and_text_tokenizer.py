import tiktoken

# are you sure you can encode everything? what about bytes?
# why don't you just encode some bytes to the vocabulary?
# don't worry we will handle that.

cl100k_base = tiktoken.get_encoding("cl100k_base")

# you can separate action from content.
# we can compare the difference. eventually we will find out which is best and most performant.

# like: [keyboard_input_start] [char1] [char2] [keyboard_input_end]
# or: [keyboard_input] [char1] [keyboard_input] [char2]

# compared to: [keyboard a] [keyboard upper a] [mouse move x 10] [mouse move y 10]
# well these are some 'action' vocabularies worth learning.

# how many special tokens can we add?

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings

# builtin special tokens: {'<|endoftext|>': 100257, '<|fim_prefix|>': 100258, '<|fim_middle|>': 100259, '<|fim_suffix|>': 100260, '<|endofprompt|>': 100276}

# max token value: 100276

def make_special_tokens(token_name_list: list[str], start):
    result = {}
    for token_name in token_name_list:
        result[token_name] = start
        start += 1
    return result


build_special_token = lambda token_def: f"<|{token_def}|>"
custom_special_tokens = [build_special_token("my_new_token")]
for direction in 'x', 'y':
    for val_type in 'pos', 'neg':
        for i in range(1024):
            tk = build_special_token(f'{direction}_{val_type}_{i}')
            custom_special_tokens.append(tk)


enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=(mrs := cl100k_base._mergeable_ranks),
    special_tokens={
        **make_special_tokens(
            list(cl100k_base._special_tokens.keys()) + custom_special_tokens,
            start=max(mrs.values()) + 1,
        )
        # **cl100k_base._special_tokens,
        # "<|im_start|>": 100_264,
        # "<|im_end|>": 100_265,
        # "<|my_new_token|>": 200_000,
    },
)

# 100255 is the max token number in mergeable ranks
# you can add new tokens.
print(f"{cl100k_base.max_token_value:=}")
print(f"{cl100k_base.n_vocab:=}")
# cl100k_base.special_tokens_set
# breakpoint()
encode_bytes_target = bytes([x for x in range(255)])

# you could use unicode regex to encode bytes
# tweaking `_encode_single_piece`, `_encode_only_native_bpe` under tiktoken.core.Encoding


text_target = f"hello world{custom_special_tokens[0]}my name is andy{build_special_token('x_pos_1000')}"
tokens = enc.encode(
    text_target, disallowed_special=()
)  # this will pass but will not be converted into special tokens.

# can we train the model new tokens by using different encodings? this could introduce duality.
# you can inform the model about the encoding. so the model might not misbehave.

# tokens = enc.encode(text_target, allowed_special={custom_special_tokens[0]})
# tokens = enc.encode(text_target, allowed_special={custom_special_tokens[0]}, disallowed_special = ())
tokens_with_special = enc.encode(
    text_target, allowed_special="all", disallowed_special=()
)
print(tokens)  # no special token!
print(tokens_with_special)

# what if i allow the ai to emit multiple tokens a time?
# i will sort the "simutaneous" tokens and order by priority
# what about training? is that purely online? or shall we alter the training method?
# like: [a,b,c,d,e,f] -> [a,c,e], [b,d,f] -> sample by priority
# this is compression. this can speed up things. but not necessarily improve quality, unless you trade performance with quality.

# or you could augment the training data, like input = x[:-3], target = x[3:]
# either way, could possibly optimize the performance.