#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:02:37 2018

@author: howard
"""

from mymodel import get_dummies, detect_str_columns,model,logistic_model
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# -------------- modeling - Random Forest--------------------

# 讀取電商資料
data = pd.read_csv('contract.csv')

# 偵測有字串的欄位
str_columns = detect_str_columns(data)
dataset = get_dummies(str_columns, data)

# 確認全部都是數字 float, int, uint --> ML
dataset.info()

# 切分資料集
X =dataset.drop(columns=['buy'])
y =dataset['buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# XGBClassifier for var imp
res_data, xgboost_y_test_df = model(
        clf = XGBClassifier(n_estimators = 300, random_state = 0, nthread = 8),
        X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,
        plot_name = 'xgboost'
        )

# logistic regression for var imp
result_df,y_test_df = logistic_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        plot_name = 'logistic_regression'
                   )

