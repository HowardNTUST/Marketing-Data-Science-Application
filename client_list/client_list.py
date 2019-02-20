#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:40:59 2019

@author: howard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:17:15 2019

@author: howard
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def detect_str_columns(data):
    '''
    1. 偵測有字串的欄位
    2. 挑選出來，準備encoding
    
    '''
    strlist = list(set(np.where((data.applymap(type)==str))[1].tolist()))
    return data.columns[strlist].tolist()

def get_dummies(dummy, dataset):
    ''''
    make variables dummies
    ref：http://blog.csdn.net/weiwei9363/article/details/78255210
    '''
    dummy_fields = list(dummy)
    for each in dummy_fields:
        dummies = pd.get_dummies( dataset.loc[:, each], prefix=each ) 
        dataset = pd.concat( [dataset, dummies], axis = 1 )
    
    fields_to_drop = dummy_fields
    dataset = dataset.drop( fields_to_drop, axis = 1 )
    return dataset

# 讀取marketing資料
data = pd.read_csv('marketing2.csv')


# 偵測有字串的欄位
str_columns = detect_str_columns(data.drop(columns = 'UID'))
dataset = get_dummies(str_columns, data)

# 確認全部都是數字 float, int, uint --> ML
dataset.info()


# 切分資料集
X =dataset.drop(columns=['買A商品'])
y =dataset['買A商品']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# 保留UID
train_uid = X_train['UID']
test_uid = X_test['UID']

# 3. -------------ML預測-------------

# 設定xgb 分類模型
del X_train['UID']
del  X_test['UID']
clf = XGBClassifier(n_estimators=300 ,random_state = 0, nthread = 8, learning_rate=0.5)
model_xgb = clf.fit(X_train, y_train, verbose=True,eval_set=[(X_train, y_train), (X_test, y_test)])

# 進行預測
y_pred = model_xgb.predict(X_test)
y_pred_prob = model_xgb.predict_proba(X_test)[:,1]

# 精準客戶名單
XGBClassifier_test_df=pd.DataFrame(y_test.values ,columns =['客戶對A商品【實際】購買狀態'])
XGBClassifier_test_df['客戶對A商品【預測】購買機率'] = y_pred_prob
test_uid = test_uid.reset_index().drop(columns = ['index'])
XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)



