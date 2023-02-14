#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# SECTION - 針對不同的產品，到底哪些才是高購買率的消費者？


# %%
# SECTION - 程式碼1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, r2_score
from ai_marketing_lib import model_profit_fun, profit_linechart
import matplotlib.pyplot as plt
from ai_marketing_lib import plot_confusion_matrix
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from ai_marketing_lib import detect_str_columns, transform_to_category

# 讀取marketing資料
data = pd.read_csv('marketing_ec_data_train.csv')
data.info()

# 輸出data.info() --> 將data.dtypes 與 data.count() 合併
data_type = pd.concat(
    [pd.DataFrame(data.dtypes), data.count()], axis=1).reset_index()
data_type.columns = ['欄位名稱', '資料型態', '非空值的資料筆數']
data_type.to_excel('01_資料形態.xlsx', index=False)
#!SECTION - 程式碼1

# %%
# SECTION - 程式碼2
# 偵測有字串的欄位
str_columns = detect_str_columns(data.drop(columns='UID'))
dataset = transform_to_category(str_columns, data)


# 確認全部都是數字 float, int, uint --> ML
dataset.info()

# 輸出【轉換後】data.info() --> 將data.dtypes 與 data.count() 合併
data_type = pd.concat([pd.DataFrame(dataset.dtypes),
                      dataset.count()], axis=1).reset_index()
data_type.columns = ['欄位名稱', '資料型態', '非空值的資料筆數']
data_type.to_excel('02_【轉換後】資料形態.xlsx', index=False)
#!SECTION - 程式碼2

# %%
# SECTION - 程式碼3
# 將測試資料也同樣讀取進來並執行資料轉換
# 讀取marketing資料
data_test = pd.read_csv('marketing_ec_data_testing.csv')
data_test.info()

# 偵測有字串的欄位
str_columns = detect_str_columns(data_test.drop(columns='UID'))
dataset_test = transform_to_category(str_columns, data_test)

# 確認全部都是數字 float, int, uint --> ML
dataset_test.info()

#!SECTION - 程式碼3


# %%
# SECTION - 程式碼4
# 繪圖：查看y變數的分佈圖
# Buy 比例
print('行銷顧客後卻不購買的比例', round(dataset['買A商品'].value_counts()
      [0]/len(data) * 100, 2), '%')
print('行銷顧客成功而購買的比例', round(dataset['買A商品'].value_counts()[
      1]/len(data) * 100, 2), '%')


# 看看 y變數的分佈圖
dataset['count'] = 1
data_count = dataset.groupby('買A商品', as_index=False)['count'].sum()
fig = px.bar(data_count, x="買A商品", y="count",
             color='count',
             title='purchase decision \n (0:no vs 1:yes )',
             )

plot(fig, filename='purchase_decision.html', auto_open=False)
#!SECTION - 程式碼4

# %%
# SECTION - 程式碼5
# 區分訓練資料集的X與y
X_train = dataset.drop(columns=['買A商品'])
y_train = dataset['買A商品']

X_test = dataset_test.drop(columns=['買A商品'])
y_test = dataset_test['買A商品']

# 保留UID
train_uid = X_train['UID']
test_uid = X_test['UID']

# 刪除UID
del X_train['UID']
del X_test['UID']

# 再次檢查X_train與X_test的資料型態
data_train_type = pd.concat(
    [pd.DataFrame(X_train.dtypes), X_train.count()], axis=1).reset_index()
data_train_type.columns = ['欄位名稱', '資料型態', '非空值的資料筆數']
data_train_type.to_excel('03_【訓練資料集_特徵變數】資料形態.xlsx', index=False)

data_test_type = pd.concat(
    [pd.DataFrame(X_test.dtypes), X_test.count()], axis=1).reset_index()
data_test_type.columns = ['欄位名稱', '資料型態', '非空值的資料筆數']
data_test_type.to_excel('04_【測試資料集_特徵變數】資料形態.xlsx', index=False)

# 刪除不需要的變數
if X_train.filter(regex='count').shape[1] > 0:
    X_train.drop(columns=X_train.filter(regex='count'), inplace=True)

print(X_train.columns)

# !SECTION - 程式碼5

# %%
# SECTION - 程式碼6
# 訓練機器學習模型

# 命名模型物件
clf = XGBClassifier(n_estimators=500,
                    tree_method="hist",
                    enable_categorical=True,
                    random_state=0, nthread=8,
                    learning_rate=0.05,
                    )

# 進行訓練
model_xgb = clf.fit(X_train, y_train, verbose=True,
                    eval_metric='aucpr', eval_set=[(X_test, y_test)])

# 預測測試資料集的顧客購買與否
y_pred = model_xgb.predict(X_test)

# 預測測試資料集的顧客購買機率
y_pred_prob = model_xgb.predict_proba(X_test)[:, 1]

# 建構精準顧客名單
consumer_acc_list = pd.DataFrame(y_test.values, columns=['顧客對A商品【實際】購買狀態'])
consumer_acc_list['顧客對A商品【預測】購買機率'] = y_pred_prob

# 6. 將UID加回去顧客精準行銷名單
test_uid = test_uid.reset_index().drop(columns=['index'])
consumer_acc_list = pd.concat([test_uid, consumer_acc_list], axis=1)

# 7. 將精準顧客名單購買機率由大到小排序
consumer_acc_list = consumer_acc_list.sort_values(
    by='顧客對A商品【預測】購買機率', ascending=False)

consumer_acc_list = consumer_acc_list[[
    'UID', '顧客對A商品【預測】購買機率', '顧客對A商品【實際】購買狀態',]]

consumer_acc_list['顧客對A商品【預測】購買機率'] = round(
    consumer_acc_list['顧客對A商品【預測】購買機率'], 3)

# 8. 將精準顧客名單存成excel檔
consumer_acc_list.to_excel('05_顧客精準行銷清單.xlsx', index=False)

# !SECTION - 程式碼6

# %%
# SECTION - 程式碼7
# 資料科學角度評估模型表現

# 準確度
print(accuracy_score(y_test, y_pred))
# !SECTION - 程式碼7

# %%
# SECTION - 程式碼8
# 混淆矩陣評估模型表現

# 首先評估測試資料集真正購買A商品顧客佔所有資料的比例
marketing_all_consumer = round(y_test.value_counts()[1]/len(y_test), 4)
print('真正購買A商品顧客佔所有資料的比例', marketing_all_consumer * 100, '%')
# !SECTION - 程式碼8

# %%
# SECTION - 程式碼9
# 先做出混淆矩陣

model_conf = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(model_conf,
                      classes=['No', 'Buy'],
                      title='Confusion matrix',
                      cmap=plt.cm.Reds
                      )

# 行銷成功的比例可以達到多少呢？
ps = precision_score(y_test, y_pred)
print('行銷成功的比例', round(ps, 4) * 100, '%')

# 相較原先海撒全部顧客行銷方法來說，用了模型之後，提升了多少成交比呢？
lift = round(ps / marketing_all_consumer, 2)
print('提升了', lift, '倍')
# !SECTION - 程式碼9


# %%
# SECTION - 程式碼10
# 行銷活動利潤評估

XGB_all_df, XGB_model_profit_df, XGB_y_test_df = model_profit_fun(
    clf=XGBClassifier(n_estimators=500,
                      tree_method="hist",
                      enable_categorical=True,
                      random_state=0, nthread=8,
                      learning_rate=0.05,
                      ),  # sklearn的模型
    
    X_train=X_train,  # 訓練資料集 X：training set (x)
    y_train=y_train,  # 訓練資料集 Y：training set (buy)
    X_test=X_test,  # 測試資料集 X : testing set (x)
    y_test=y_test,  # 測試資料集 Y : testing set (buy)
    test_uid = test_uid,  # 測試資料集 UID
    sales_price=2500,  # 價格
    marketing_expense=185.32,  # 行銷費用或銷貨成本
    product_cost=1215.2,  # 產品成本
    plot_name='XGB_A商品_')  # 產出結果的名稱，會存到資料夾

# !SECTION - 程式碼10

# %%
# SECTION - 程式碼11
# 畫出XGB利潤折線圖
profit_linechart(
    y_test_df=XGB_y_test_df, # 顧客精準行銷清單
    sales_price=2500,  # 價格
    marketing_expense=185.32,  # 行銷費用
    product_cost=1215.2,
    plot_name='XGB_product A_')

# !SECTION - 程式碼11

