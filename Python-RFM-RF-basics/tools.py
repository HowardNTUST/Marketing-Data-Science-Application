# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:32:23 2020

@author: Ivan
"""
import pandas as pd
import platform
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
colrogroup = ['#427f8f','#4a8fa1','#559db0','#66a7b8','#77b1c0','#89bbc8','#9ac5d0','#bdd9e0','#cee3e8','#e0edf0']

def checkPlatform():
    # 判斷是甚麼作業系統
    theOS = list(platform.uname())[0]
    if theOS == 'Windows':
        theOS = '\\'
        ecode = 'utf-8-sig'
    else:
        theOS = '/'
        ecode = 'utf-8'
        
    return theOS, ecode

def cut_dataframe(df, cut = 6, cutColumn = '你想切割的欄位'):
    label = []
    tolerance = (df[cutColumn].max()-df[cutColumn].min())/cut #先計算切割的公差
    for i in range( cut-1): #開始創造切割label，最後一個值不要跑出來，因為要是最大值，用加的怕會有小數點誤差
        label.append(str(int(df[cutColumn].min()+tolerance*i)) + ' - ' + str(int(df[cutColumn].min()+tolerance*(i+1))))
    # 加上最後一位
    label.append(str(int(df[cutColumn].min()+tolerance*(i+1))) + ' - ' + str(int(df[cutColumn].max())))
    
    return pd.cut(df[cutColumn] , cut, labels=label), label #切割後的分類內容


