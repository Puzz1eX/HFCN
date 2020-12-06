import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
import tensorflow.keras.backend as K

import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping,LambdaCallback,Callback
from DeepCTR.deepctr.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from DeepCTR.deepctr.models import DeepFM,AFM,CaFiNet,xDeepFM,AutoInt,FLEN,DCN,FiBiNET
from DeepCTR.deepctr.models import HFCN,HFAA

import pickle
# 固定随机种子
SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


def auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


class SaveRoutingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, model,X_train,save_path):
        Callback.__init__(self)
        self.model = model
        self.x=X_train
        self.save_path=save_path
        self.routing_list=[]
    def on_step_end(self, step, logs={}):
        # print(self.model.layers[24].output)
        # self.routing_list.append(self.model.layers[24].output)
        pass
        
    def on_epoch_end(self, epoch, logs={}):
        print(K.eval(self.model.get_layer('intaracting_caps').output[1]))

        with open(self.save_path,'wb') as handle:
            pickle.dump(self.routing_list,handle)
        self.routing_list=[]

if __name__ == "__main__":
    data = pd.read_csv("movie_lens.csv")
    sparse_features = [
        "user_id","movie_id",
        "title","gender", "age", "occupation", "zip", ]
    field_info = dict(movie_id='movie', genres='movie', user_id='user_id', gender='user_cha',
                      age='user', occupation='user_cha', zip='user')
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=16)
                              for feat in sparse_features]
    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4, group_name=field_info[feat])
    #                           for feat in sparse_features]              

    use_weighted_sequence = False
    if use_weighted_sequence:
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
            key2index) + 1, embedding_dim=16), maxlen=max_len, combiner='mean',
                                                   weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
    else:
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
            key2index) + 1, embedding_dim=16), maxlen=max_len, combiner='mean',
                                                   weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["genres"] = genres_list
    model_input["genres_weight"] = np.random.randn(data.shape[0], max_len, 1)

    BATCH_SIZE=1024
    # 4.Define Model,compile and train
#     model = AFM(linear_feature_columns, dnn_feature_columns, task='binary',use_attention=False)
    model = AFM(linear_feature_columns, dnn_feature_columns, task='binary',use_attention=True,attention_factor=32)
    # model=FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
#     model=HFCN(linear_feature_columns, dnn_feature_columns,num_capsule=3, dim_capsule=4, routings=3, task='binary',dnn_hidden_units=(400,400,400))
#     model=HFAA(linear_feature_columns, dnn_feature_columns,num_feature=3, dim_feature=8, head_num=2,hi_att_factor_size=8, task='binary',dnn_hidden_units=())
    # model=xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary',cin_layer_size=(200,200,200),dnn_hidden_units=(400,400,400))
#     model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary',att_embedding_size=16,dnn_hidden_units=(400,400,400))
    # model = FLEN(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    #优化器
    # save_routing_callback = SaveRoutingCallback(model,model_input,"routing_list.pkl")

    earlystop=EarlyStopping(monitor='val_auc', patience=2, mode='max')
    optimizer=tf.keras.optimizers.Adam(lr=0.001,)
    model.compile(optimizer, "binary_crossentropy",metrics=['binary_crossentropy',auc] )
    history = model.fit(model_input, data[target].values,callbacks=[earlystop],
                        batch_size=BATCH_SIZE, epochs=40, verbose=1, validation_split=0.1, )
    # model.load_weights(model_par_cfg['checkpoint_path'], by_name=False)
    #取某一层的输出为输出新建为model，采用函数模型
#     caps_layer_model = tf.keras.models.Model(inputs=model.input,
#                                         outputs=model.get_layer('intaracting_caps').output)
#     #以这个model的预测值作为输出
#     routing_scores = caps_layer_model.predict(model_input,batch_size=BATCH_SIZE)
#     with open('routing_score.pkl','wb') as h:
#         pickle.dump(routing_scores,h)