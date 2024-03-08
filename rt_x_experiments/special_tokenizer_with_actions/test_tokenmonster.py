import tokenmonster

# # Optionally set the tokenmonster directory, otherwise it will use ~/_tokenmonster
# tokenmonster.set_local_directory("/path/to/preferred")

# Load a vocabulary by name, filepath or URL
vocab = tokenmonster.load("english-24000-consistent-v1") # cannot download.

# Tokenize some text
text = "Some text to turn into token IDs."
tokens = vocab.tokenize(text)
print('tokens', tokens)
