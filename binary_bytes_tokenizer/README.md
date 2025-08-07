in terminal, some special operate codes are in binary, such as <ESC>

we first check if openai tiktoken and huggingface tokenizers supports binary tokenization

since unicode is a variable length encoding, and tokenizers may only use encoded string, as a result it might be troublesome to directly process binary data. we might want to change the code of tokenizers in order to adapt our needs.

if not, we will use some special escape sequence or write our own tokenizer.

secondly, we need to use a special asciicast-like format to retain the structure of our input data for tokenization, controlling training and inference.

a rather destructive method for adding more tokens while maintaining the overall token limit is to eradicate parts of rarely used tokens from the normal tokens. it is easy to do in openai tiktoken programmatically. meanwhile, one can directly edit the tokenizer.json file in huggingface tokenizers.