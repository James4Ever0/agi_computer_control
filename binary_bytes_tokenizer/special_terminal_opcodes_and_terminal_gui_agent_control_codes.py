# we want to minimize new token creation and new special token adaption

# if our fine-tuned agent is only used for terminal/gui operation, we may skip special tokenization

# we may want to design a parametric markup language upon existing token system, or we want to create something new instead

# to formalize the process of special token handling, and to preserve the meaning of data, we can annotate the attribute of consecutive text pieces like:

# payload_type can be: "text", "token_ids"

# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "special_to_normal"} // allowed_special = (), disallowed_special = () (all special tokens will be decomposed into normal tokens)
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "normal_only"} // allowed_special = (), disallowed_special = "all" (all special tokens will cause exception)
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "special_with_normal"} // allowed_special = "all", disallowed_special = "all" (all special tokens will be mapped to their special token ids)

# or more favorably, specify the exact allowed_special and disallowed_special configuration
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_options": {"allowed_special": <allowed_special: Sequence[str] | Literal["all"]>, "disallowed_special": <disallowed_special: Sequence[str] | Literal["all"]>}}

# if some special token is not in allowed_special and in disallowed_special, that special token will make tokenizer.encode to raise exception.

# optionally, specify the file format and annotation details in the beginning of the file, just like asciicast v2
# {"file_format": "cybergod_tokenizer_exchange_format", "version": "1", "tokenizer_library": "openai-tiktoken", "tokenizer_config": {"tokenizer_load_method": <tokenizer_load_method: str>, "tokenizer_load_name": <tokenizer_load_name: str>, "additional_special_tokens": <additional_special_tokens: list[str]>, "additional_tokens": <additional_tokens: list[str]>}}