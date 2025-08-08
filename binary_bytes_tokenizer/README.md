in terminal, some special operate codes are in binary, such as <ESC>

we first check if openai tiktoken and huggingface tokenizers supports binary tokenization

since unicode is a variable length encoding, and tokenizers may only use encoded string, as a result it might be troublesome to directly process binary data. we might want to change the code of tokenizers in order to adapt our needs.

if not, we will use some special escape sequence or write our own tokenizer.

secondly, we need to use a special asciicast-like format to retain the structure of our input data for tokenization, controlling training and inference.

a rather destructive method for adding more tokens while maintaining the overall token limit is to eradicate parts of rarely used tokens from the normal tokens. it is easy to do in openai tiktoken programmatically. meanwhile, one can directly edit the tokenizer.json file in huggingface tokenizers.

since openai tokenizer "tiktoken" use bytes in its "mergeable_ranks" or normal tokens, we have to convert string tokens into bytes.

all string in python can be encoded with utf-8 into bytes, but not all bytes can be decoded into string using utf-8. for example, bytes([0xff]) cannot be decoded into string using utf-8.

the terminal recorder asciinema can record string, but not bytes. guess the raw underlying data in tty

in terminal, if the input/output data are in string, and the terminal encoding is set to utf-8, then we can use bytes() to encode string into bytes, and use decode() to decode bytes into string.

notice, if you "cat" 0xff into terminal, it will be displayed/recorded as a square ("�", b'\xef\xbf\xbd', tofu), because 0xff is not a valid utf-8 character. the same applies to other bytes that cannot be decoded into string using utf-8. this means all non-valid utf-8 bytes will be displayed as tofu in terminal, losing their original values, just like [UNK] token in the BERT tokenizer.

terminal emulator is responsible for encoding/decoding string and bytes, and it is the terminal emulator that determines whether the bytes can be decoded into string using utf-8. if the terminal emulator cannot decode the bytes into string using utf-8, then the bytes will be displayed as tofu in terminal.

if the terminal emulator breaks because of non-valid utf-8 bytes, then one can reset the state of the terminal emulator by sending the escape sequence "\x1b[2J\x1b[H" to the terminal emulator. this will clear the terminal emulator and reset its state.

ansi control sequences are in bytes. for example, the sequence '\x9b' is not valid in utf-8, but it is a valid ansi single byte csi control sequence, and should not be encoded as b'\xc2\x9b'.

ansi control sequences are valid in utf-8

