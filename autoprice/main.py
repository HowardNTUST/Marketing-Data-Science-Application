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
#%matplotlib inline

train = pd.read_table('train.tsv', engine='c')

print('價格數序性統計\n' + str(round(train.price.describe())))

# 選擇比較熱門的品牌數值當作分析依據，其他以missing value代替之
dataset = train
brandnum = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing']
NUM_BRANDS = len(brandnum) - len(brandnum[brandnum<=4]) #中位數評斷

# 選擇熱門的種類當作參考，其他以missing value代替之
NUM_CATEGORIES = 1000

# 製作項目名稱詞袋時，所有被算入的字詞最少要有被提到過10次以上，否則不列入計算
NAME_MIN_DF = 10

# 製作description詞向量時，最大的feature以重要的50000字為主
MAX_FEATURES_ITEM_DESCRIPTION = 50000

# 以RMSLE當作損失函數
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# 將category分得更系的切分函數
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

# 部份變數有遺失值，填寫成NA函數
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

# 挑選出表現成績好的品牌與分類，當作分析依據
def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'

# 挑選將category變成類別函數
def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
    
start_time = time.time()

# 統計test data的欄數
nrow_test = train.shape[0] #-dftt.shape[0]

# 將小於1美元的（對於平台業者沒價值的商品）商品移除
dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)

nrow_train = train.shape[0] #-dftt.shape[0]

#y = train["price"]
merge: pd.DataFrame = pd.concat([train, dftt])
y = np.log1p(merge["price"])
del merge['price']
#merge: pd.DataFrame = pd.concat([train, dftt, test])

del train
gc.collect()

# 將category_name切成三塊，再將原本category_name移除，這樣分析的就更仔細拉
merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)
print('{} 完成切分時間'.format(time.time() - start_time))

# 處理NA
handle_missing_inplace(merge)
print('[{}] 處理遺失值的時間.'.format(time.time() - start_time))

# 將popular品牌挑選出來，把明不見經傳的品牌或僅有1次出現的變成遺失值，避免造成過度擬和(overfitting)之狀況
cutting(merge)
print('[{}] 處理熱門品牌時間.'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] 轉換成名目變數'.format(time.time() - start_time))

# 如果字數出現次數小於10，便不進行字數的計算（vector space model）
cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

# 將Category轉變成計數形式（vector space model）
cv = CountVectorizer()
X_category1 = cv.fit_transform(merge['general_cat'])
X_category2 = cv.fit_transform(merge['subcat_1'])
X_category3 = cv.fit_transform(merge['subcat_2'])
print('[{}] categories詞袋完成.'.format(time.time() - start_time))



# 在item_description以tf-idf法做
tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words='english')

X_description = tv.fit_transform(merge['item_description'])
print('[{}] TFIDF於 item_description花費時間.'.format(time.time() - start_time))

# 做brand name, item_condition_id, shipping變數 的 one hot encoding
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] one hot encoding於 brand_name花費時間 .'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] 將item_condition_id 及 shipping 轉變成類別變數的時間.'.format(time.time() - start_time))

# 將所有非結構化變數轉變成稀疏-結構化矩陣，以便分析
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
print('[{}] 將所有非結構化變數轉變成稀疏-結構化矩陣'.format(time.time() - start_time))



train_X, test_X, train_y, test_y = train_test_split(sparse_merge, y, test_size = 0.1, random_state = 144) 

model = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
  normalize=False, random_state=101, solver='auto', tol=0.01)

model.fit(train_X, train_y)
print('[{}] Train ridge completed'.format(time.time() - start_time))
predsR = model.predict(X=test_X)
print('[{}] Predict ridge completed'.format(time.time() - start_time))

rmsleR = rmsle(predsR, test_y)
rmsle(np.expm1(predsR), np.expm1(test_y))
np.sqrt(mean_squared_error(np.expm1(predsR), np.expm1(test_y)))
print(rmsle(np.expm1(predsR), np.expm1(test_y)))
#  0.4676692829338082

pred_true_df = pd.concat([pd.DataFrame(predsR),pd.DataFrame(test_y.reset_index() )], axis = 1)
del pred_true_df['index']
pred_true_df.columns = ['Pred_price', 'true_price']
pred_true_df['Pred_price'] = np.expm1(pred_true_df['Pred_price'] )
pred_true_df['true_price'] = np.expm1(pred_true_df['true_price'] )

pred_true_df.head(10)



print( '真實值比預測值高的有' + str(len(pred_true_df[pred_true_df['true_price']>pred_true_df['Pred_price'] ])) + '個')
# 真實值比預測值高的有71263個

print( '預測值高比真實值比高的有' + str(len(pred_true_df[pred_true_df['true_price']<pred_true_df['Pred_price'] ])) + '個')
# 預測值高比真實值比高的有76904個

(pred_true_df['Pred_price'] -  pred_true_df['true_price']).mean()
# 平均價格 -3.3035621763466594
