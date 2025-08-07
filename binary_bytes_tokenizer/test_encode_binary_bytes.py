# try to use tiktoken to tokenize the above thing
# tiktoken works


def test_encode_with_tiktoken(input_string: str):
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
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
    assert enc.decode(enc.encode(input_string)) == input_string

    print("Input: %s" % repr(input_string))

    encoded_tokens = enc.encode(input_string)
    print("Encoded tokens: ")
    print("", encoded_tokens)

    for it in encoded_tokens:
        print("Token: %s, Decoded: %s" % (it, repr(enc.decode([it]))))


def test_encode_with_tokenizers(input_string: str):
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        "./gpt-oss-tokenizer/tokenizer.json" # 'gpt-oss' load from json passed assertion
    )  # "bert-base-uncased" assertion failed. not lossless.
    encoded_tokens = tokenizer.encode(input_string)

    print("Input: %s" % repr(input_string))

    print("Encoded tokens: ")
    print("", encoded_tokens)

    assert tokenizer.decode(ids=encoded_tokens.ids) == input_string
    for it in encoded_tokens.ids:
        decoded_token=tokenizer.decode(ids=[it])
        print('Token: %s, Decoded: %s' % (it, repr(decoded_token)))


def test():
    # example in asciinema cast recording
    input_string = (
        "\u001b[?2004h\u001b]0;root@50a6d006d95a: /\u0007root@50a6d006d95a:/# "
    )
    print("Testing with tiktoken")
    test_encode_with_tiktoken(input_string)
    print()
    print("Testing with tokenizers")
    test_encode_with_tokenizers(input_string)


if __name__ == "__main__":
    test()
