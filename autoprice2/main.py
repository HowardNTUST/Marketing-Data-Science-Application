#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:38:55 2018

@author: howard
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:31:41 2018

@author: howard
"""

import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
# from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
from PyQt5 import QtCore, QtGui
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

train_XL1, valid_XL1, train_yL1, valid_yL1 = train_test_split(sparse_merge, y, test_size = 0.1, random_state = 144) 
d_trainL1 = lgb.Dataset(train_XL1, label=train_yL1, max_bin=8192)
d_validL1 = lgb.Dataset(valid_XL1, label=valid_yL1, max_bin=8192)
watchlistL1 = [d_trainL1, d_validL1]

paramsL1 = {
    'learning_rate': 0.65,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 60,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.5,
    'nthread': 7
}
modelL1 = lgb.train(paramsL1, train_set=d_trainL1, num_boost_round=8000, valid_sets=watchlistL1, 
                    early_stopping_rounds=5000, verbose_eval=500) 
predsL1 = modelL1.predict(test_X)
#rmsleL1 = rmsle(predsL1, test_y)
rmsle(np.expm1(predsL1), np.expm1(test_y))

print ("LightGBM1 RMSLE = " + str(rmsle(np.expm1(predsL1), np.expm1(test_y))))
# LightGBM1 RMSLE = 0.4519021444030349

print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(predsL1), np.expm1(test_y)))   )
# lightgbm比起 Ridge regression的rmsle還要低上 0.015767138530773328

print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( round( (( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(predsL1), np.expm1(test_y))) / rmsle(np.expm1(predsR), np.expm1(test_y)))*100,2) )+ '%')
# lightgbm比起 Ridge regression的rmsle還要低上 3.37%



print('Start training...')
train_XL2, valid_XL2, train_yL2, valid_yL2 = train_test_split(train_X, train_y, test_size = 0.1, random_state = 144) 
d_trainL2 = lgb.Dataset(train_XL2, label=train_yL2, max_bin=8192)
d_validL2 = lgb.Dataset(valid_XL2, label=valid_yL2, max_bin=8192)
watchlistL2 = [d_trainL2, d_validL2]
paramsL2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 1,
        'nthread': 7
    }
modelL2 = lgb.train(paramsL2, train_set=d_trainL2, num_boost_round=5500, valid_sets=watchlistL2, \
early_stopping_rounds=1000, verbose_eval=500) 


print ("LightGBM1 RMSLE = " + str(rmsle(np.expm1(predsL2), np.expm1(test_y))))
#LightGBM1 RMSLE = 0.4636889152284461

print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(predsL2), np.expm1(test_y)))   )
#lightgbm比起 Ridge regression的rmsle還要低上 0.003980367705362142

print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( round( (( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(predsL2), np.expm1(test_y))) / rmsle(np.expm1(predsR), np.expm1(test_y)))*100,2) )+ '%')
#lightgbm比起 Ridge regression的rmsle還要低上 0.85%



# predsL1的Rmsle表現的好！ 所以給予多一點的權重，predsL2次之，predsR最差，而權重的加總 ＝ 1
preds = predsR*0.2 + predsL1*0.55 + predsL2*0.25

# 集成模型 2.0的結果是...
print ("集成模型 2.0 RMSLE = " + rmsle(np.expm1(preds), np.expm1(test_y)))
# 0.44282788615681024
print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(preds), np.expm1(test_y)))   )
#lightgbm比起 Ridge regression的rmsle還要低上 0.02484139677699798

print('lightgbm比起 Ridge regression的rmsle還要低上 ' + str( round( (( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(preds), np.expm1(test_y))) / rmsle(np.expm1(predsR), np.expm1(test_y)))*100,2) )+ '%')
#lightgbm比起 Ridge regression的rmsle還要低上 5.31%




# make pred and true matrix
def pred_true_df(pred =preds , test_y=test_y, log_selling_threshold = 2, selling_fee= .1):
    pred_true_df = pd.concat([pd.DataFrame(pred),pd.DataFrame(test_y.reset_index() )], axis = 1)
    del pred_true_df['index']
    pred_true_df.columns = ['Pred_price', 'true_price']
    pred_true_df['Pred_price'] = np.expm1(pred_true_df['Pred_price'] )
    pred_true_df['true_price'] = np.expm1(pred_true_df['true_price'] )
    pred_true_df['log_diff'] = np.log1p(np.abs(pred_true_df['Pred_price'] - pred_true_df['true_price']))
    pred_true_df_select = pred_true_df[pred_true_df['log_diff'] <= 2 ]
    
    return pred_true_df_select['Pred_price'].sum() *selling_fee




# 營收大比拼：集成2.0 vs Ridge regression

#集成2.0 
集成2_0_revenue = pred_true_df(pred =preds , test_y=test_y, log_selling_threshold = 2)
# 營收 157031.50337287926

# Ridge regression
Ridge_revenue = pred_true_df(pred =predsR , test_y=test_y, log_selling_threshold = 2)
# 營收 149124.05499286225

# 兩者營收差距
營收差距 = (集成2_0_revenue- Ridge_revenue) * 29.21 # 2018/4/5 匯價
# 差了230,976元新台幣阿

# 從Ridge regression轉換到集成模型2.0的RMSLE每1%到底差多少營收？！
營收差距 / ((( rmsle(np.expm1(predsR), np.expm1(test_y)) - rmsle(np.expm1(preds), np.expm1(test_y))) / rmsle(np.expm1(predsR), np.expm1(test_y))) *100)#
# 從Ridge regression轉換到集成模型2.0的 Rmsle 每1%就差了43,484元新台幣！


