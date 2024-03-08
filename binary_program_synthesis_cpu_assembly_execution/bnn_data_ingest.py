import larq  # keras/tensorflow

# ref: https://github.com/itayhubara/BinaryNet.pytorch
import tensorflow as tf
import random

# tf.experimental.numpy.experimental_enable_numpy_behavior()


dim = 2
# dim = 10


def makeQuant(in_dim, out_dim):
    layer = larq.layers.QuantDense(  # this will regulate all values into integers
        units=out_dim,  # anything -> 2
        # units=dim,
        input_quantizer=larq.quantizers.SteSign(clip_value=1.0),
        kernel_quantizer=larq.quantizers.SteSign(clip_value=1.0),
        kernel_constraint=larq.constraints.WeightClip(clip_value=1.0),
        # input_shape=(42,), # still working? not in sequential.
        input_shape=(in_dim,),
    )
    return layer


# larq.layers.QuantDense(units=2)

# dim = 1024
# model = tf.keras.Sequential()
seqlen = 20
# (AB * BA) * (BA * AB)

# random_x_in = [[random.randint(0, 1) for _ in range(seqlen)]]
random_x_in = [[random.randint(0, 1) for _ in range(seqlen)]]
x_in = tf.convert_to_tensor(random_x_in, dtype=tf.float32)  # [1, 20]
# x_in = tf.random.uniform(shape=(1, dim), minval=0, maxval=2, dtype=tf.float32)
l_emb = tf.keras.layers.Embedding(2, dim)

t_emb = l_emb(x_in)

l_q = makeQuant(dim, dim)
l_k = makeQuant(dim, dim)
l_v = makeQuant(dim, dim)

t_q = l_q(t_emb)
t_k = l_k(t_emb)
t_v = l_v(t_emb)

t_att_pre = tf.matmul(t_k, t_q, transpose_a=True)

t_att = tf.nn.softmax(t_att_pre, axis=2) / (dim**0.5)
t_feat = tf.matmul(t_v, t_att)

l_out = makeQuant(dim, 2)
t_out = l_out(t_feat)

print(x_in)
print()
print(t_out)  # not within 1 and 0
binary = tf.argmax(t_out, axis=2)
print()
print(binary)
print(binary.shape, t_out.shape)
# breakpoint()
