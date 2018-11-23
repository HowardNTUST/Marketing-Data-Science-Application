#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:33:20 2018

@author: slave1
"""

from sklearn.metrics import confusion_matrix, auc, accuracy_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools

# ----設定繪圖-------
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.font_manager import FontProperties 
import seaborn as sns 
myfont=FontProperties(fname='Microsoft JhengHei',size=14) 
sns.set(font=myfont.get_family()) 
sns.set_style("darkgrid",{"font.sans-serif":['Microsoft JhengHei']}) 

######################## util #####################################


def logistic_model(X_train,y_train,X_test,y_test,
                    plot_name = 'logistic_regression'
                   ):
    
    X_train_log = X_train.copy()
    X_train_log['intercept'] = 1
    logistic = sm.Logit(y_train,X_train_log)
    
    # fit the model
    result = logistic.fit()
    result_df = results_summary_to_dataframe(result, plot_name = plot_name)
    
    
    X_test_log = X_test.copy()
    X_test_log['intercept'] = 1
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = result.predict(X_test_log)
    y_test_df['pred_yn']= np.where(y_test_df[plot_name+'_pred']>=0.5, 1,0)
    
    
    conf_logist = confusion_matrix(y_test_df['buy'], y_test_df['pred_yn'])
    
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    
    # -------single model summary--------
    
    print( "################ summary ################ ")
    
    print(confusion_matrix(y_test_df['buy'], y_test_df['pred_yn']))
#    print("____________________{}分類報告____________________".format(plot_name))
#    print(classification_report(y_test_df['buy'], y_test_df['pred_yn']))
    print(accuracy_score(y_test_df['buy'], y_test_df['pred_yn']))      
    
          
    # importance
    print( '------------------ 應注意之變數 ------------------' )
    print('、'.join(result_df['變數'].tolist()))
    print('\n'.join(result_df['意涵'].tolist()))
    
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
#    print(result_df)
    result_df.to_excel(plot_name+'重要變數表.xlsx')
    return result_df,y_test_df


def logistic_conf(
        X_train,
        y_train,
        X_test,
        y_test,
        plot_name
                   ):
        
    X_train_log = X_train.copy()
    X_train_log['intercept'] = 1
    logistic = sm.Logit(y_train,X_train_log)
    
    # fit the model
    result = logistic.fit()
#    result_df = results_summary_to_dataframe(result, plot_name = plot_name)
        
        
        
    X_test_log = X_test.copy()
    X_test_log['intercept'] = 1
    y_test_df=pd.DataFrame(y_test)
    y_test_df['pred'] = result.predict(X_test_log)
    y_test_df['pred_yn']= np.where(y_test_df['pred']>=0.5, 1,0)
    
    
    conf_logist = confusion_matrix(y_test_df['buy'], y_test_df['pred_yn'])
    
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    
    print( "################ summary ################ ")
    print('Confusion matrix')
    print(confusion_matrix(y_test_df['buy'], y_test_df['pred_yn']))
#    print("____________________{}分類報告____________________".format(plot_name))
#    print(classification_report(y_test_df['buy'], y_test_df['pred_yn']))
    print("Test Accuracy = {:.3f}".format(accuracy_score(y_test_df['buy'], y_test_df['pred_yn'])))
    
    
    return conf_logist 




def detect_str_columns(data):
    '''
    1. 偵測有字串的欄位
    2. 挑選出來，準備encoding
    
    '''
    strlist = list(set(np.where((data.applymap(type)==str))[1].tolist()))
    return data.columns[strlist].tolist()


def results_summary_to_dataframe(results,plot_name):
    '''This takes the result of an statsmodel results table and transforms it into a dataframe'''
#    print(results.summary())
    pvals = round(results.pvalues, 3)
    coeff = round(results.params, 6)
#    conf_lower = round(results.conf_int()[0], 3)
#    conf_higher = round(results.conf_int()[1], 3)

    results_df = pd.DataFrame({
                               "參數":coeff,
                               "p_values":pvals,
#                               "conf_lower":conf_lower,
#                               "conf_higher":conf_higher
                                })
    
    
#    results_df = results_df[results_df['p_values']<0.05]
    results_df = results_df.reset_index()
    results_df.columns = ['變數', '參數', 'p_values']
    #Reordering...
#    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    
    feat_imp = results_df.參數
    feat = results_df['變數'].tolist()
    res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=True)
#    res_data.plot('Features', 'Importance', kind='bar', title='Feature Importances')
    plt.figure(figsize=(10,10))
    plt.bar(res_data['Features'], res_data['Importance'])
    plt.title('Importance')
#    plt.subplots_adjust(left=0.7, right=0.8, top=0.9, bottom=0.7)
    plt.ylabel(plot_name+' Feature Importance Score')
    plt.savefig(plot_name+'.png', dpi=300)
    plt.show()
    
    results_df['轉換後參數'] = round(np.exp(results_df['參數']), 3)
    results_df = results_df.sort_values(['轉換後參數'],ascending = False)
    results_df['程度排名'] =range(1,len(results_df)+1 ) 
    results_df['意涵'] = '每增加 「' + results_df['變數'] +  '」 的1個單位 ，等同增加' + results_df['轉換後參數'].astype(str) + '的可能購買倍數'
    
    return results_df


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

def model(clf, X_train,y_train,X_test,y_test,sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression') :
    model = clf.fit(X_train, y_train)
    
#    a = X_test.iloc[15:16, ::].copy()
#    model.predict_proba(a)[:,1]
#    a['registry_to_use_time'] = a['registry_to_use_time'] - 1 
#    a['Prod_output_num'] = a['Prod_output_num'] -400
#    model.predict_proba(a)[:,1]
#    
#    a = X_test.iloc[19:20, ::].copy()
#    model.predict_proba(a)[:,1]
#    a['registry_to_use_time'] = a['registry_to_use_time'] 
#    a['Prod_output_num'] = a['Prod_output_num'] -178
#    a['click_on_prod'] = a['click_on_prod'] - 4
#    model.predict_proba(a)[:,1]
    
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
    feat_imp = model.feature_importances_
    feat = X_train.columns.tolist()
    res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=True)
    res_data.plot('Features', 'Importance', kind='barh', title='變數重要性長條圖',stacked=True, figsize = (15,10))
    plt.ylabel(plot_name+' Feature Importance Score')
    plt.savefig(plot_name+'.png', dpi=300)
    plt.show()
    
    
    res_data = res_data.sort_values('Importance',ascending=False)
    
    # > 50%
    res_data = res_data[res_data['Importance']>res_data['Importance'].describe()['50%']]
    

    print( "################ summary ################ ")
    plot_confusion_matrix(conf_logist, ['no','buy'],
                  title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
    
    
    # importance
    print( '------------------ 應注意之變數 ------------------' )
                    
    print(res_data)
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
    res_data.to_excel(plot_name+'重要變數表.xlsx')
    
    
    return res_data, y_test_df
    
