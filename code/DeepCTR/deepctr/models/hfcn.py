import tensorflow as tf

from ..inputs import input_from_feature_columns, build_input_features, combined_dnn_input, get_linear_logit
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InteractingLayer
from ..layers.utils import concat_func, add_func

from ..layers.capsulelayers import CapsuleLayer

def HFCN(linear_feature_columns, dnn_feature_columns, att_layer_num=3, att_embedding_size=10, att_head_num=2,
            att_res=True,
            dnn_hidden_units=(256, 256), dnn_activation='relu', l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
            task='binary', ):
    

    if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
        raise ValueError("Either hidden_layer or att_layer_num must > 0")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    
    #获得embedding
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)
    #LR
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    
    #capsule network
    caps_input = concat_func(sparse_embedding_list, axis=1)

    fm_mf_out,routing_scores = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=1,
                             name='intaracting_caps')(caps_input)
    
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)

    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(concat_func([fm_mf_out, dnn_output]))

    final_logit = add_func([linear_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
