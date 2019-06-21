#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:56:17 2019

@author: howard
"""

import itertools

from sklearn.metrics import confusion_matrix, auc, accuracy_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

# ----設定繪圖-------
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sns 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(title+'.png', dpi=300)
# make demographics variables dummy 
def model_testRF(clf, X_train,y_train,X_test,y_test,
                    plot_name = 'logistic_regression'):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
    # 畫conf matrix
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    # -------single model summary--------
    

    print( "################ summary ################ ")
    
    print(confusion_matrix(y_test, y_pred))
#    print("____________________{}分類報告____________________".format(plot_name))
#    print(classification_report(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))


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


# function
def client_list(model, y_test, X_test, test_uid, name):
    '''
    model:要放入的模型
    y_test：要驗證的資料集，如果沒有，請設定None
    X_test:要預測的資料集
    test_uid:這資料集的UID
    
    '''
    
    # prediction
    xgb_pred_prob = model.predict_proba(X_test)
    xgb_pred = model.predict(X_test)
    
    # 產出【顧客產品推薦名單】
    XGBClassifier_test_df=pd.DataFrame(y_test.values ,columns =['客戶對A商品【實際】購買狀態'])
    XGBClassifier_test_df['客戶對A商品【預測】購買機率'] = xgb_pred_prob[:,1]
    test_uid = test_uid.reset_index().drop(columns = ['index'])
    XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)
    XGBClassifier_test_df = XGBClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)
        
    
    XGBClassifier_test_df.to_excel(name+'顧客產品推薦名單.xlsx')
    return XGBClassifier_test_df,xgb_pred
