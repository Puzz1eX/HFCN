import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros,Ones, glorot_normal,
                                                  glorot_uniform)
from tensorflow.python.keras import initializers
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Layer
import math

from .interaction import InteractingLayer
from .activation import activation_layer
from .utils import concat_func, reduce_sum, softmax, reduce_mean
import itertools

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

# capsule compatible batch_dot
def caps_batch_dot(x, y,transpose):
    x = K.expand_dims(x, 2)
    if transpose:
        y = K.permute_dimensions(y, (0,1,3,2))
    o = tf.matmul(x, y)
    return K.squeeze(o, 2)

# capsule compatible batch_dot
def caps_batch_outter_dot(x, y,transpose):
    x = K.expand_dims(x, 2)
    if transpose:
        y = K.permute_dimensions(y, (0,1,3,2))
    o = tf.matmul(x, y)
    return K.squeeze(o, 2)


def broadcastable_where(condition, x=None, y=None, *args, **kwargs):
    if x is None and y is None:
        return tf.where(condition, x, y, *args, **kwargs)
    else:
        _shape = tf.broadcast_dynamic_shape(tf.shape(condition), tf.shape(x))
        _broadcaster = tf.ones(_shape)
        return tf.where(
            condition & (_broadcaster > 0.0), 
            x * _broadcaster,
            y * _broadcaster,
            *args, **kwargs
        )
    

class CapsuleLayer(Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,layer_size=(32,32),
                 kernel_initializer='glorot_normal',seed=1024,
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.dropout_rate=0.7
        self.seed = seed

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = int(input_shape[1])
        self.input_dim_capsule = int(input_shape[2])
        
        
        # transmission matrix
        self.reweight_W = self.add_weight(shape=[self.input_dim_capsule,self.num_capsule,self.dim_capsule],
                                 initializer=glorot_uniform(seed=self.seed),
                                 name='reweight_W')
        
        self.num_fields=self.num_capsule
        self.kernel_mf = self.add_weight(
            name='kernel_mf',
            shape=(int(self.num_fields * (self.num_fields - 1) / 2), 1),
            initializer=tf.keras.initializers.Ones(),
            regularizer=None,
            trainable=True)

        self.kernel_fm = self.add_weight(
            name='kernel_fm',
            shape=(self.num_fields, 1),
            initializer=tf.keras.initializers.Constant(value=0.5),
            regularizer=None,
            trainable=True)
        
        # self-attention
        self.kernel_highint = self.add_weight(
            name='kernel_highint',
            shape=(self.num_fields, 1),
            initializer=tf.keras.initializers.Constant(value=0.5),
            regularizer=None,
            trainable=True)
        
        self.self_attention_factor=self.dim_capsule
        self.self_attention_layer=1
        self.head_num=2
        # embedding_size=self.self_attention_factor * self.head_num
        embedding_size=self.dim_capsule
        self.bias_mf = self.add_weight(name='bias_mf',
                                        shape=(embedding_size),
                                        initializer=Zeros())
        self.bias_fm = self.add_weight(name='bias_fm',
                                        shape=(embedding_size),
                                        initializer=Zeros())
        

        self.routing_init=self.add_weight(name="routing_init",
                                        shape=(self.num_capsule,self.input_num_capsule),
                                        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed,stddev=10))
        
        self.bias_highint = self.add_weight(name='bias_highint',
                                        shape=(self.self_attention_factor*self.head_num),
                                        initializer=Zeros())
        self.built = True
        # Be sure to call this somewhere!
        super(CapsuleLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        #inputs_hat.shape[None,input_num_capsule,num_capsule,dim_capsule]
        inputs_hat=tf.tensordot(inputs, self.reweight_W, axes=(-1, 0))
        inputs_hat=K.permute_dimensions(inputs_hat, (0,2,1,3))
        
        b = K.expand_dims(self.routing_init,0)
        b = K.tile(b,[K.shape(inputs_hat)[0],1,1])
        
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = softmax(b,axis=1)
            
            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule , dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(caps_batch_dot(c, inputs_hat,transpose=False))
            
            # outputs.shape =  [None, num_capsule, dim_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
            # b.shape=[batch_size, num_capsule, input_num_capsule]
            # b_add = caps_batch_dot(outputs, inputs_hat,transpose=True)
            # norm = (K.max(b,axis=1)-K.min(b_add,axis=1))/(b_add - K.min(b_add,axis=1))
            b += caps_batch_dot(outputs, inputs_hat,transpose=True) 
            
        # End: Routing algorithm -----------------------------------------------------------------------#

        c = softmax(b, axis=1)
        
        routing_score=K.expand_dims(c, -1)

        attention_output=routing_score*inputs_hat
        
        field_wise_embeds_list=[K.squeeze(embeds,1) for embeds in tf.split(attention_output,attention_output.shape[1],axis=1)]
        # HiFM module
        square_of_sum_list = [
            tf.square(reduce_sum(field_i_vectors, axis=1, keep_dims=True))
            for field_i_vectors in field_wise_embeds_list
        ]
        
        sum_of_square_list = [
            reduce_sum(field_i_vectors * field_i_vectors,
                       axis=1,
                       keep_dims=True)
            for field_i_vectors in field_wise_embeds_list
        ]

        field_fm = tf.concat([
            square_of_sum - sum_of_square for square_of_sum, sum_of_square in
            zip(square_of_sum_list, sum_of_square_list)
        ], 1)

        hi_fm = reduce_sum(field_fm, axis=1)
        hi_fm = reduce_sum(field_fm*self.kernel_fm , axis=1)
        hi_fm = tf.nn.bias_add(hi_fm, self.bias_fm)
        
        
        # mf
        field_wise_vectors=reduce_sum(attention_output,axis=2,keep_dims=False)
        
        left = []
        right = []
        

        for i, j in itertools.combinations(list(range(self.num_fields)), 2):
            left.append(i)
            right.append(j)

        embeddings_left = tf.gather(params=field_wise_vectors,
                                    indices=left,
                                    axis=1)
        embeddings_right = tf.gather(params=field_wise_vectors,
                                     indices=right,
                                     axis=1)
        
        embeddings_prod = embeddings_left * embeddings_right
        
        field_weighted_embedding = embeddings_prod * self.kernel_mf
        
        h_mf = reduce_sum(field_weighted_embedding, axis=1)
        h_mf = tf.nn.bias_add(h_mf, self.bias_mf)
        
        # self-attention
        for _ in range(self.self_attention_layer):
            field_wise_vectors=InteractingLayer(self.self_attention_factor, self.head_num, True)(field_wise_vectors)
        high_int=reduce_sum(field_wise_vectors*self.kernel_highint,axis=1)
        high_int = tf.nn.bias_add(high_int, self.bias_highint)

        
        return concat_func([hi_fm,h_mf,high_int]),routing_score

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
