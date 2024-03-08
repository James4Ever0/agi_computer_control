# words have orderings.
# but actions have not. usually stacked.
# how to resolve this issue?

# if you can decode the embedding using multiple embedding layers, what happens?
# one embedding -> multiple meanings
# for every embedding there is one silent token, allow to train separately.
# fourier transform to concat these different kinds of embeddings
# or you could train some decision embedding, to control the action type

import torch
emb = torch.nn.Embedding
emb2 = torch.nn.Embedding

# the bot told us that we need to process different token separately.

# you could have more choices on fft/ifft and more. you can first concat then use some dense layer to reduce its size.

# choice 1: emb1+emb2 (elementwise addition) -> transformer -> decode by two linear layers and compare original logits
# choice 2: fft(emb1)+fft(emb2) -> ifft -> transformer -> decode by two linear layers and compare original logits
# choice 3: emb1 -> tranformer1 -> linear1 -> logits1; emb2 -> tranformer1 -> linear2 -> logits2
# choice 4: concat emb1&emb2 -> transformer -> decode -> separate into two logits