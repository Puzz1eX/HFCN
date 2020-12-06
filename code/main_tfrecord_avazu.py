from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from DeepCTR.deepctr.models import xDeepFM,DeepFM,AFM,AutoInt,FLEN,FGCNN,PNN,DCN,FiBiNET,HFCN

from DeepCTR.deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

from tensorflow.python.keras.models import  save_model,load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping,LambdaCallback,Callback
from tensorflow.python.keras.optimizers import TFOptimizer

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import numpy as np
import random
import math
import os

from tqdm import tqdm
import pickle
import argparse

# 固定随机种子
SEED = 2020
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

# 把训练轮损失数据流到 JSON 格式的文件。文件的内容
import json
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, log_path):
        Callback.__init__(self)
        self.log = open(log_path, mode='at', buffering=1)
    def on_epoch_end(self, epoch, logs={}):
        log_dict={'epoch': str(epoch)}
        log_dict.update({k:str(v) for k,v in logs.items()})
        self.log.write(json.dumps(log_dict)+'\n')
    def on_train_end(self,logs):
        self.log.close()

def args_parse():
    parser = argparse.ArgumentParser(description='An implement of HFCN')
    parser.add_argument('--model_name', dest='model_name', type=str,default='HFCN',
                        help='define the model to use')
    
    # paths
    parser.add_argument('--train_path', dest='train_path', type=str,default="train_set.pkl",
                        help='train file path')
    parser.add_argument('--test_path', dest='test_path', type=str,default="train_set.pkl",
                        help='test file path')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str,default="./models/",
                        help='checkpoint root path')
    parser.add_argument('--log_path', dest='log_path', type=str,default="./logs/",
                        help='logs root path')
    # training config
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int,default=1024,
                        help='batch size')
    parser.add_argument('--epoch_num', dest='epoch_num', type=int,default=20,
                        help='max number of epochs')
    parser.add_argument('--is_train', dest='is_train', action='store_true',
                        help='whether in training phase')
    parser.add_argument('--use_checkpoint', dest='use_checkpoint', action='store_true',
                        help='whether use checkpoint for fine-tuning')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name',type=str, default='HFCN.hdf5',
                        help='the name of checkpoint')
    
    # model config
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int,default=16,
                        help='the dimension of feature embedding')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()

    
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    
    log_file_path=args.log_path+'log_{}.txt'.format(args.model_name)
    with open(log_file_path, 'at') as log:
        log.write(json.dumps(vars(args))+'\n')
    
    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip',
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]
    target = ['click']

    field_info = dict(C14='user', C15='user', C16='user', C17='user',
                      C18='user', C19='user', C20='user', C21='user', C1='user',
                      banner_pos='context', site_id='context',
                      site_domain='context', site_category='context',
                      app_id='item', app_domain='item', app_category='item',
                      device_model='user', device_type='user',
                      device_conn_type='context', hour='context',
                      device_id='user'
                      )
    # this file save the unique number of each feature field.
    with open("vocab_size_dict_avazu.pkl",'rb') as vocab_f:
        vocab_size_dict_avazu=pickle.load(vocab_f)
    
    # cause FLEN needs manner hierarchical fields
    if args.model_name=='FLEN':
        fixlen_feature_columns = [
            SparseFeat(name, vocabulary_size=vocab_size_dict_avazu[name], embedding_dim=args.embedding_dim, use_hash=False, dtype='int32',group_name=field_info[name])
            for name in sparse_features]
    else:
        fixlen_feature_columns = [
            SparseFeat(name, vocabulary_size=vocab_size_dict_avazu[name], embedding_dim=args.embedding_dim, use_hash=False, dtype='int32')
            for name in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    with open(args.train_path,'rb') as train_f:
        train_set=pickle.load(train_f)
    with open(args.test_path,'rb') as test_f:
        test_set=pickle.load(test_f)
    train_model_input = {name: train_set[name] for name in feature_names}
    test_model_input = {name: test_set[name] for name in feature_names}
    

    #定义模型
    model=None
    if args.model_name=='CIN':
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary',cin_layer_size=(200,200,200))
    elif args.model_name=='xDeepFM':
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary',cin_layer_size=(200,200,200),dnn_hidden_units=(400,400,400))
    elif args.model_name=='DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    elif args.model_name=='FM':
        model = AFM(linear_feature_columns, dnn_feature_columns, task='binary',use_attention=False)
    elif args.model_name=='AFM':
        model = AFM(linear_feature_columns, dnn_feature_columns, task='binary',use_attention=True,attention_factor=256)
    elif args.model_name=='AutoInt_shallow':
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary',att_embedding_size=32,dnn_hidden_units=())
    elif args.model_name=='AutoInt_dnn':
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary',att_embedding_size=32,dnn_hidden_units=(400,400,400))
    elif args.model_name=='FGCNN':
        model = FGCNN(linear_feature_columns, dnn_feature_columns, task='binary', conv_kernel_width=(9, 9, 9, 9), conv_filters=(38, 40, 42, 44),dnn_hidden_units=(4096,2048))
    elif args.model_name=='IPNN':
        model=PNN(dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    elif args.model_name=='DCN':
        model=DCN(linear_feature_columns,dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400),cross_num=3)
    elif args.model_name=='FLEN':
        model=FLEN(linear_feature_columns,dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    elif args.model_name=='FiBiNET':
        model=FiBiNET(linear_feature_columns,dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    elif args.model_name=='HFCN':
        model=HFCN(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=(400,400,400))
    
    if model is None:
        print('模型名称有误！')
        exit(0)
    # if resume
    if args.use_checkpoint and args.checkpoint_name is not None:
        model.load_weights(args.checkpoint_path+args.checkpoint_name, by_name=True)
        initial_epoch=int(args.checkpoint_path.split('-')[1])
    else:
        initial_epoch=0
    
    #优化器
    optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate,)
    
    #callbacks list
    json_logging_callback = LoggingCallback(log_file_path)
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    filepath=args.checkpoint_path+args.model_name+"-{epoch: 02d}-{val_auc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1,save_best_only=True,mode='max')
    earlystop=EarlyStopping(monitor='val_auc', patience=3, mode='max')
    
    
    model.compile(optimizer, "binary_crossentropy",metrics=['binary_crossentropy',auc], )
    print(model.summary())
    
    if args.is_train:
        history = model.fit(train_model_input,train_set[target].values,batch_size=args.batch_size,
        validation_data=(test_model_input,test_set[target].values),
        callbacks=[checkpoint,earlystop,json_logging_callback],
        epochs=args.epoch_num,initial_epoch=initial_epoch,
        verbose=1,workers=8,max_queue_size=args.batch_size*10,use_multiprocessing=True)
    

    if 'HFCN' in args.model_name:
        del train_model_input
        import gc
        gc.collect()
        #取某一层的输出为输出新建为model，采用函数模型
        caps_layer_model = tf.keras.models.Model(inputs=model.input,
                                            outputs=model.get_layer('intaracting_caps').output)
        #以这个model的预测值作为输出
        y,inner_product,routing_scores = caps_layer_model.predict(test_model_input,verbose=1,batch_size=4096)
        del y,test_model_input
        gc.collect()
        with open('routing_score_avazu.pkl','wb') as h:
            pickle.dump([inner_product,routing_scores],h)
