#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 23:59:16 2018

@author: howard
"""

# roc curve and auc on imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np

# Importing the libraries
from util import get_dummies
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.font_manager import FontProperties 
import seaborn as sns 
myfont=FontProperties(fname='Microsoft JhengHei',size=14) 
sns.set(font=myfont.get_family()) 
sns.set_style("darkgrid",{"font.sans-serif":['Microsoft JhengHei']}) 

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.font_manager import FontProperties 
import seaborn as sns 
myfont=FontProperties(fname='Microsoft JhengHei',size=14) 
sns.set(font=myfont.get_family()) 
sns.set_style("darkgrid",{"font.sans-serif":['Microsoft JhengHei']}) 


# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.95,0.01], random_state=298)
# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)

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
    print("____________________購買與否IFICATION REPORT____________________")
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
    
model(KNeighborsClassifier(n_neighbors=3), 'KNeighborsClassifier')
