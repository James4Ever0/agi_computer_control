import larq  # keras/tensorflow

# ref: https://github.com/itayhubara/BinaryNet.pytorch
import tensorflow as tf
import random

dim = 10

# larq.layers.QuantDense(units=2)

# dim = 1024
# model = tf.keras.Sequential()
seqlen = 20

# (AB * BA) * (BA * AB)

# random_x_in = [[random.randint(0, 1) for _ in range(seqlen)]]
random_x_in = [[[random.randint(0, 1)] for _ in range(seqlen)]]
x_in = tf.convert_to_tensor(random_x_in, dtype=tf.float32)
# x_in = tf.random.uniform(shape=(1, dim), minval=0, maxval=2, dtype=tf.float32)
l_emb = tf.keras.layers.Embedding(2, dim)
l_out = larq.layers.QuantDense(  # this will regulate all values into integers
    units=2, # anything -> 2
    # units=dim,
    input_quantizer=larq.quantizers.SteSign(clip_value=1.0),
    kernel_quantizer=larq.quantizers.SteSign(clip_value=1.0),
    kernel_constraint=larq.constraints.WeightClip(clip_value=1.0),
    # input_shape=(42,), # still working? not in sequential.
    input_shape=(dim,),
)
# model.add(l_emb)
# model.add(l_out)
x_out = model(x_in)

print(x_in)
print()
print(x_out)  # not within 1 and 0
binary = tf.argmax(x_out, axis=2)
print()
print(binary)