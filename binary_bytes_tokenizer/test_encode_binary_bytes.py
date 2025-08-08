# try to use tiktoken to tokenize the above thing
# tiktoken works

SPECIAL_TOKEN_ENDOFTEXT_LITERAL = "<|endoftext|>"


def test_encode_with_tiktoken(input_string: str):
    import tiktoken

    # TODO: implement tiktoken tokenizer JSON saver and loader
    # since mergeable_ranks containing bytes as dict keys, we need to improvise by either hex or base64 encoding
    # annotate the encoding used for mergeable rank keys. we may only need to annotate it once.

    # to construct custom tokenizer:
    # tiktoken.Encoding(name: 'str', *, pat_str: 'str', mergeable_ranks: 'dict[bytes, int]', special_tokens: 'dict[str, int]', explicit_n_vocab: 'int | None' = None)
    #  |      Creates an Encoding object.
    #  |      
    #  |      See openai_public.py for examples of how to construct an Encoding object.
    #  |      
    #  |      Args:
    #  |          name: The name of the encoding. It should be clear from the name of the encoding
    #  |              what behaviour to expect, in particular, encodings with different special tokens
    #  |              should have different names.
    #  |          pat_str: A regex pattern string that is used to split the input text.
    #  |          mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
    #  |              must correspond to merge priority.
    #  |          special_tokens: A dictionary mapping special token strings to their token values.
    #  |          explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
    #  |              that the number of mergeable tokens and special tokens is equal to this number.
    # Example:
    # cl100k_base = tiktoken.get_encoding("cl100k_base")

    # # In production, load the arguments directly instead of accessing private attributes
    # # See openai_public.py for examples of arguments for specific encodings
    # # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    # enc = tiktoken.Encoding(
    #     # If you're changing the set of special tokens, make sure to use a different name
    #     # It should be clear from the name what behaviour to expect.
    #     name="cl100k_im",
    #     pat_str=cl100k_base._pat_str,
    #     mergeable_ranks=cl100k_base._mergeable_ranks,
    #     special_tokens={
    #         **cl100k_base._special_tokens,
    #         "<|im_start|>": 100264,
    #         "<|im_end|>": 100265,
    #     }
    # )

    enc = tiktoken.get_encoding("o200k_base")
    print("Min mergeable rank: %s, Max mergeable rank: %s" % (min(enc._mergeable_ranks.values()), max(enc._mergeable_ranks.values())))
    # Min mergeable rank: 0, Max mergeable rank: 199997
    print("Special tokens:" , enc._special_tokens)
    # Special tokens: {'<|endoftext|>': 199999, '<|endofprompt|>': 200018}

    try:
        assert (
            enc.decode_bytes(enc._encode_bytes(b"\xff\xff")) == b"\xff\xff"
        )  # not working
    except:  # let's not dig into this 'str' vs 'bytes' myth right now, and favor 'str' over 'bytes'
        print("Bytes:", repr(b"\xff\xff"))
        print("Encode:", repr(enc._encode_bytes(b"\xff\xff")))
        print(
            "Encode -> Decode result:",
            repr(enc.decode_bytes(enc._encode_bytes(b"\xff\xff"))),
        )
    # default setting (allowed_special = (), disallowed_special = "all") of enc.encode() will prevent encoding special tokens.
    # cases for "<|endoftext|>"
    # allowed_special = "all" -> id of the special token
    # disallowed_special = () -> breakdown ids of the special token literal
    assert enc.decode(enc.encode(input_string)) == input_string

    print("Input: %s" % repr(input_string))

    encoded_tokens = enc.encode(input_string)
    print("Encoded tokens: ")
    print("", encoded_tokens)

    for it in encoded_tokens:
        print("Token: %s, Decoded: %s" % (it, repr(enc.decode([it]))))

    endoftext_token_ids = enc.encode(SPECIAL_TOKEN_ENDOFTEXT_LITERAL, allowed_special="all")
    print("Special token: %s, Encoded: %s, Decoded: %s" % (SPECIAL_TOKEN_ENDOFTEXT_LITERAL, endoftext_token_ids, repr(enc.decode(endoftext_token_ids))))

    # print("Attributes of OpenAI tiktoken tokenizer:")
    # print(dir(enc))
    # ['_core_bpe', '_encode_bytes', '_encode_only_native_bpe', '_encode_single_piece', '_mergeable_ranks', '_pat_str', '_special_tokens', 'decode', 'decode_batch', 'decode_bytes', 'decode_bytes_batch', 'decode_single_token_bytes', 'decode_tokens_bytes', 'decode_with_offsets', 'encode', 'encode_batch', 'encode_ordinary', 'encode_ordinary_batch', 'encode_single_token', 'encode_to_numpy', 'encode_with_unstable', 'eot_token', 'is_special_token', 'max_token_value', 'n_vocab', 'name', 'special_tokens_set', 'token_byte_values']

def test_encode_with_tokenizers(input_string: str):
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        "./gpt-oss-tokenizer/tokenizer.json" # 'gpt-oss' load from json passed assertion
    )  # "bert-base-uncased" assertion failed. not lossless.
    encoded_tokens = tokenizer.encode(input_string)
    # the output of tokenizer.encode() is an instance of Encoding
    # Encoding(num_tokens=1, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])

    print("Input: %s" % repr(input_string))

    print("Encoded tokens: ")
    print("", encoded_tokens)

    assert tokenizer.decode(ids=encoded_tokens.ids) == input_string
    for it in encoded_tokens.ids:
        decoded_token=tokenizer.decode(ids=[it])
        print('Token: %s, Decoded: %s' % (it, repr(decoded_token)))

    # list attributes of tokenizer object
    # print("Attributes of huggingface BPE tokenizer:")
    # print(dir(tokenizer))
    # ['add_special_tokens', 'add_tokens', 'decode', 'decode_batch', 'decoder', 'enable_padding', 'enable_truncation', 'encode', 'encode_batch', 'encode_batch_fast', 'encode_special_tokens', 'from_buffer', 'from_file', 'from_pretrained', 'from_str', 'get_added_tokens_decoder', 'get_vocab', 'get_vocab_size', 'id_to_token', 'model', 'no_padding', 'no_truncation', 'normalizer', 'num_special_tokens_to_add', 'padding', 'post_process', 'post_processor', 'pre_tokenizer', 'save', 'to_str', 'token_to_id', 'train', 'train_from_iterator', 'truncation']

    # interesting methods for "gpt-oss" tokenizer:
    # add_tokens(tokens: list[str]), tokens are added if not already in the vocab
    # add_special_tokens(tokens:)
    # encode(sequence, pair=None, is_pretokenized=False, add_special_tokens=True), add_special_tokens are for whether to convert literal special token representations into token ids
    # save(path: str, pretty=True)
    # decode(self, ids, skip_special_tokens=True)
    
    # to show help on these methods:
    # for it in ["decode", "add_tokens", "add_special_tokens", "encode", "save"]:
    #     print("Getting help on tokenizer.%s" % it)
    #     help(getattr(tokenizer, it))

    endoftext_token_ids = tokenizer.encode(SPECIAL_TOKEN_ENDOFTEXT_LITERAL)
    # endoftext_token_ids shall be list[str] of length = 1

    print("Special token: %s, Encoded: %s, Decoded: %s" % (SPECIAL_TOKEN_ENDOFTEXT_LITERAL, endoftext_token_ids.ids, repr(tokenizer.decode(endoftext_token_ids.ids, skip_special_tokens=False))))

def test():
    # example in asciinema cast recording
    input_string = ( # <|endoftext|> is a special token in gpt
        "\u001b[?2004h\u001b]0;root@50a6d006d95a: /\u0007root@50a6d006d95a:/# "
    )
    print("Testing with tiktoken")
    test_encode_with_tiktoken(input_string)
    print()
    print("Testing with tokenizers")
    test_encode_with_tokenizers(input_string)


if __name__ == "__main__":
    test()
