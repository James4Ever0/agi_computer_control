# we want to minimize new token creation and new special token adaption

# if our fine-tuned agent is only used for terminal/gui operation, we may skip special tokenization

# we may want to design a parametric markup language upon existing token system, or we want to create something new instead

# to formalize the process of special token handling, and to preserve the meaning of data, we can annotate the attribute of consecutive text pieces like:

# payload_type can be: "text", "token_ids", "bytes_base64", "bytes_hex"
# we may have binary payload, if using "bytes_*" types

# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "special_to_normal"} // allowed_special = (), disallowed_special = () (all special tokens will be decomposed into normal tokens)
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "normal_only"} // allowed_special = (), disallowed_special = "all" (all special tokens will cause exception)
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_preset": "special_with_normal"} // allowed_special = "all", disallowed_special = "all" (all special tokens will be mapped to their special token ids)
# // we may have additional presets like "special_only", "byte_level" (mapping every byte to a token)

# or more favorably, specify the exact allowed_special and disallowed_special configuration
# {"payload": <payload: str>, "payload_type": "text", "tokenizer_encode_options": {"allowed_special": <allowed_special: Sequence[str] | Literal["all"]>, "disallowed_special": <disallowed_special: Sequence[str] | Literal["all"]>}}

# if some special token is not in allowed_special and in disallowed_special, that special token will make tokenizer.encode to raise exception.

# optionally, specify the file format and annotation details in the beginning of the file, just like asciicast v2
# {"file_format": "cybergod_tokenizer_exchange_format", "version": "1", "tokenizer_library": "openai-tiktoken", "tokenizer_config": {"tokenizer_load_method": <tokenizer_load_method: str>, "tokenizer_load_name": <tokenizer_load_name: str>, "additional_special_tokens": <additional_special_tokens: list[str]>, "additional_tokens": <additional_tokens: list[str]>}}

osc_common_opcodes = [
    "\x1b]",  # OSC Control sequence prefix
    "\x1b]0;",  # OSC 0, Common usage: \x1b]0;<message>\x07
]

string_terminator_codes = [
    "\x07",  # BEL
    "\x1b\\",  # ST
]

# enumerate common ones, like "\x1b[A"
csi_common_opcodes = [
    "\x1b[",  # CSI Control sequence prefix
    "\x1b[A",  # Cursor up
    "\x1b[B",  # Cursor down
    "\x1b[C",  # Cursor forward (right)
    "\x1b[D",  # Cursor back (left)
    "\x1b[E",  # Move to next line (cursor down + column 0)
    "\x1b[F",  # Move to previous line (cursor up + column 0)
    "\x1b[H",  # Move cursor to home position (0,0)
    "\x1b[J",  # Clear screen from cursor down
    "\x1b[K",  # Clear line from cursor right
    "\x1b[1J",  # Clear screen from cursor up
    "\x1b[2J",  # Clear entire screen
    "\x1b[0K",  # Clear line from cursor right (same as \x1b[K)
    "\x1b[1K",  # Clear line from cursor left
    "\x1b[2K",  # Clear entire line
    "\x1b[s",  # Save cursor position
    "\x1b[u",  # Restore cursor position
    "\x1b[?25h",  # Show cursor
    "\x1b[?25l",  # Hide cursor
    "\x1b[0m",  # Reset text formatting
]
