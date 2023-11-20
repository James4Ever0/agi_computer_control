import larq  # keras/tensorflow

# ref: https://github.com/itayhubara/BinaryNet.pytorch
import tensorflow as tf
import random

dim = 10

# dim = 1024
random_x_in = [[random.randint(0, 1) for _ in range(dim)]]
x_in = tf.convert_to_tensor(random_x_in, dtype=tf.float32)
# x_in = tf.random.uniform(shape=(1, dim), minval=0, maxval=2, dtype=tf.float32)

x_out = larq.layers.QuantDense(  # this will regulate all values into integers
    units=dim,
    input_quantizer=larq.quantizers.SteSign(clip_value=1.0),
    kernel_quantizer=larq.quantizers.SteSign(clip_value=1.0),
    kernel_constraint=larq.constraints.WeightClip(clip_value=1.0),
)(x_in)

print(x_in)
print()
print(x_out)  # not within 1 and 0
