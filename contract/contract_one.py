#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:22:09 2018

@author: howard
"""


# Importing the libraries
#from util import get_dummies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.font_manager import FontProperties 
import seaborn as sns 
myfont=FontProperties(fname='Microsoft JhengHei',size=14) 
sns.set(font=myfont.get_family()) 
sns.set_style("darkgrid",{"font.sans-serif":['Microsoft JhengHei']}) 


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

# read data
data = pd.read_csv('contract.csv')

# The 購買與否es are heavily skewed we need to solve this issue later.
print('不購買', round(data['購買與否'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('購買', round(data['購買與否'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

# 看看y變數的分佈
colors = ["#0101DF", "#DF0101"]
sns.countplot('購買與否', data=data, palette=colors)
plt.title('購買與否 Distributions \n (0:不購買|| 1:購買 )', fontsize=14)


# get_dummies
dataset = get_dummies(['使用者地區', '性別'],data)

# split training and testing set
X =dataset.drop(columns=['購買與否'])
y =dataset['購買與否']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 



#-------------- modeling-------------------
def model(clf, plot_name) :
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
#    print("ROC_AUC_score : %.6f" % (roc_auc_score(y_test, y_pred)))
    #Confusion Matrix
    print(confusion_matrix(y_test, y_pred))
    print("____________________{}分類報告____________________".format(plot_name))
    print(classification_report(y_test, y_pred))
    
    precision, recall, threshold = precision_recall_curve(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    #Plot Curves
    closest_zero = np.argmin(np.abs(threshold))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = [10,5])
    
    #Precision-Recall Curve
    ax1.plot(precision, recall)
    ax1.plot(closest_zero_p, closest_zero_r, 'o', markersize = 8, fillstyle = 'none', c = 'r')
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Recall')
    ax1.set_title(plot_name+" Precision-Recall Curve")

    #ROC Curve
    ax2.plot(fpr, tpr, label= 'ROC curve (AUC Score = {:0.3f}; Test Acc = {:0.3f})'.format((roc_auc), model.score(X_test, y_test)))
    ax2.plot([0, 1], [0, 1], c = 'r', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(plot_name+' ROC curve')
    ax2.legend(loc = 'lower right')
    plt.tight_layout()
    
    fig.savefig(plot_name+'.png', dpi=300)    
    




# all marketing
((405)/2000)*(405/(405)*(3616)) + (1-(405)/2000)*((1595/(1595))*-500)
#       p0   p1
#a0 [[0   1595]
#a1 [ 0  405]]


model(RandomForestClassifier(random_state = 0,n_estimators = 1000), 'Random Forest')
((212+193)/2000)*(212/(193+212)*(3616)) + (1-(212+193)/2000)*((67/(1528+67))*-500)

#Training Accuracy = 1.000
#Test Accuracy = 0.870
#ROC_AUC_score : 0.740725
#       p0   p1
#a0 [[1528   67]
#a1 [ 193  212]]
#____________________購買與否IFICATION REPORT____________________
#              precision    recall  f1-score   support
#
#           0       0.89      0.96      0.92      1595
#           1       0.76      0.52      0.62       405
#
#   micro avg       0.87      0.87      0.87      2000
#   macro avg       0.82      0.74      0.77      2000
#weighted avg       0.86      0.87      0.86      2000

model(XGBClassifier(n_estimators=300 ,random_state = 0,nthread = 8), 'xgboost')
((213+192)/2000)*(213/(213+192)*(3616)) + (1-(213+192)/2000)*((73/(1522+73))*-500)
#Training Accuracy = 0.885
#Test Accuracy = 0.868
#ROC_AUC_score : 0.740079
#       p0   p1
#a0 [[1522   73]
#a1 [ 192  213]]
#____________________購買與否IFICATION REPORT____________________
#              precision    recall  f1-score   support
#
#           0       0.89      0.95      0.92      1595
#           1       0.74      0.53      0.62       405
#
#   micro avg       0.87      0.87      0.87      2000
#   macro avg       0.82      0.74      0.77      2000
#weighted avg       0.86      0.87      0.86      2000

model(LGBMClassifier(n_estimators=800,random_state = 0),'LGBMClassifier')
((206+199)/2000)*(206/(206+199)*(3616)) + (1-((206+199)/2000))*((108/(1487+108))*-500)
#Training Accuracy = 1.000
#Test Accuracy = 0.847
#ROC_AUC_score : 0.720465
#[[1487  108]
# [ 199  206]]
#____________________購買與否IFICATION REPORT____________________
#              precision    recall  f1-score   support
#
#           0       0.88      0.93      0.91      1595
#           1       0.66      0.51      0.57       405
#
#   micro avg       0.85      0.85      0.85      2000
#   macro avg       0.77      0.72      0.74      2000
#weighted avg       0.84      0.85      0.84      2000

