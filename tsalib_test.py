from tsalib import dim_vars as dvs, size_assert
import tensorflow as tf
import torch

# declare dimension variables. e.g., full name 'Batch', shorthand 'b', length 32.
# Simply use the shorthand 'b' in rest of the code.
B, C, H, W = dvs('Batch(b):32 Channels(c):3 Height(h):256 Width(w):256') 
...
# create tensors using dimension variables (interpret dim vars as integers)
x: 'bchw' = torch.randn(B, C, H, W)
x: 'bchw' = tf.get_variable("x", shape=(B, C, H, W), initializer=tf.random_normal_initializer())

# perform tensor transformations, keep track of named shapes
x: 'b,c,h//2,w//2' = maxpool(x) 

# check assertions: compare dynamic shapes with declared shapes
# assertions are 'symbolic': don't change even if declared shapes change
assert x.size() == (B, C, H // 2, W // 2)
#or, check selected dimensions
size_assert(x.size(), (B,C,H//2,W//2), dims=[1,2,3])
mytensor = torch.randn(1, 2, 2, 3 , names=('N', 'C', 'H', 'W')) # still, not annotated. cannot check statically?