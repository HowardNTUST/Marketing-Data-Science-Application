#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:22:09 2019

@author: howard
"""

# -------------前處理-------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from xgboost import XGBClassifier
# Importing the libraries
from util2 import get_dummies, client_list,detect_str_columns,model_testRF
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, accuracy_score,classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

# ----設定繪圖-------
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

# 讀取電商資料
data = pd.read_csv('contract.csv')


# ------------變數視覺化------------
# buy
print('不購買', round(data['buy'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('購買', round(data['buy'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

# 看看y變數的分佈
colors = ["#0101DF", "#DF0101"]
sns.countplot('buy', data=data, palette=colors)
plt.title('purchase decision \n (0:no vs 1:yes )', fontsize=14)


# ------------ 資料處理與轉換 - 變數轉換 ------------
str_columns = detect_str_columns(data)
dataset = get_dummies(str_columns, data)


# ------------ 資料處理與轉換 - 切分資料集 ------------

# 將X與y分割開來
X =dataset.drop(columns=['buy'])
y =dataset['buy']

# 切分資料集
'''
切分成80％的訓練資料集
    - X_train : X訓練變數
    - y_train : y訓練變數
    
切分成20％的測試資料集
    - X_test : X訓練變數
    - y_test : y訓練變數
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# 來看看各自的維度
X_train.shape
y_train.shape

X_test.shape
y_test.shape

# ------------ 資料處理與轉換 - 將UID拿出來 ------------
# 保留UID
train_uid = X_train['UID']
test_uid = X_test['UID']

# 設定xgb 分類模型
del X_train['UID']
del X_test['UID']


# -------------- modeling -------------------
'''
xgb_model
給出500棵樹進行預測: XGBClassifier(n_estimators=100 ,random_state = 0, nthread = 8, learning_rate=0.5)

'''


# 命名模型物件
xgb_model = XGBClassifier(n_estimators=100 ,random_state = 0, nthread = 8, learning_rate=0.5)

# 進行訓練
model_xgb = xgb_model.fit(X_train, y_train, verbose=True,
                          eval_set=[(X_train, y_train), (X_test, y_test)])

# 進行預測
xgb_pred = model_xgb.predict(X_test)

# 預測機率
xgb_pred_prob = model_xgb.predict_proba(X_test)


# confusion matrix plot
model_testRF(  XGBClassifier(n_estimators = 100, random_state = 0, nthread = 8, learning_rate=0.5), 
             X_train,y_train,X_test,y_test, plot_name = 'XGBClassifier')


xgb_conf =confusion_matrix(y_test, xgb_pred)
accuracy_score(y_test, xgb_pred)


# -------------- 製作顧客產品推薦名單 -------------------
client_list(model=model_xgb,y_test=y_test, X_test=X_test, test_uid=test_uid, name = 'test')



