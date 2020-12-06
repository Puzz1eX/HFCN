# HFCN: Learning Hierarchical Fields via Capsule Networks for Click-Through Rate Prediction
The code is based on [DeepCTR](https://github.com/shenweichen/DeepCTR), which helps us to keep the common module (MLP, LR, prediction layer) in the same. All the experiments are tested on Tensorflow 2.2.
## request
tensorflow  
tqdm
## Dataset
We implemented our experiments on three pulic datasets:
* Criteo: [avaliable at](http://labs.criteo.com/2014/02/download-kaggle-display\\-advertising-challenge-dataset/)
* Avazu: [avaliable at](https://www.kaggle.com/c/avazu-ctr-prediction)
* MovieLens-1M: [avaliable at](https://grouplens.org/datasets/movielens/)
Please follow the data prepocess methods in [DeepCTR](https://github.com/shenweichen/DeepCTR).
Partically, we remove the infrequent features (appearing in less than threshold instances) and treat them as a single
feature “<unknown>”, where threshold is set to 20 for Criteo, Avazu data sets. Second, since numerical
features may have large variance and hurt machine learning algorithms, we normalize numerical values by transforming a value z to $log_2(z)$ if z > 2, which is proposed by the [winner of Criteo Competition](https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf).

## Training Example
Our model [./code/DeepCTR/deepctr/models/hfcn.py](https://github.com/Puzz1eX/HFCN/blob/main/code/DeepCTR/deepctr/models/hfcn.py)
Our layers [./code/DeepCTR/deepctr/layers/capsulelayers.py](https://github.com/Puzz1eX/HFCN/blob/main/code/DeepCTR/deepctr/layers/capsulelayers.py)
Train script:  
```sh
python code/main_avazu.py --is_train --train_path <your train file> --test_path <your test file>
```


