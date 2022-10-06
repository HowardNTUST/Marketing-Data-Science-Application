#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:33:20 2018

@author: slave1
"""

from sklearn.metrics import confusion_matrix, auc, accuracy_score
import pandas as pd
import numpy as np
#import statsmodels.api as sm
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ----設定繪圖-------
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sns 

######################## util #####################################

#def docx_table(df,doc):
#    
#    # add a table to the end and create a reference variable
#    # extra row is so we can add the header row
##    doc.add_heading('測試文件', 1)
#    t = doc.add_table(df.shape[0]+1, df.shape[1])
#    
#    # add the header rows.
#    for j in range(df.shape[-1]):
#        t.cell(0,j).text = df.columns[j]
#    
#    # add the rest of the data frame
#    for i in range(df.shape[0]):
#        for j in range(df.shape[-1]):
#            t.cell(i+1,j).text = str(df.values[i,j])

def RFM_plot_grid(df2,frequency_label,recency_label,label):
    df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values') 
    df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
    df3 = df3.dropna()
    
    fig, axes = plt.subplots(6, 6,figsize=(25,15))
    counti = 0
    for i in frequency_label[::-1]:
        count = 6
        for j in recency_label:
            count -= 1 
            if df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)].shape[0] != 0:
                sns.barplot(x="types", y="values", data=df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)], capsize=.2,ax=axes[counti, count]) #.set_title("best customers")
        
        counti += 1 
    fig.savefig('RFM_plot_grid_'+label+'.png', dpi=300)

def RFM_stackedplot(df2, frequency_label,recency_label,label):
    
    df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate',label], var_name='types', value_name='values')
    df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
    df3 = df3.dropna()
    
    fig, axes = plt.subplots(6, 6, figsize=(25, 15))
#    plt.figlegend( [ax.legend()], 'label1', label = 'lower center', ncol=5, labelspacing=0.1 )
    
    counti = 0
    for i in frequency_label[::-1]:
        count = 6
        for j in recency_label:
            count -= 1 
            if df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)].shape[0] != 0:
                df4 = df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)]
                df4 = df4.groupby(['types', label]).agg({'values': 'sum'})
                df4 = df4.groupby(['types', label]).sum()
                df4 =df4.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
                df4 = df4.add_suffix('').reset_index() #to df
                df4=df4.pivot(label, 'types', 'values')
                
                #draw
                ax = df4.plot.bar(stacked=True,width=0.7, legend = False, ax =axes[counti, count] ,rot=0)
                ax.legend( loc= 1, fontsize =8)                
                # sns.barplot(x="types", y="values", data=df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)], capsize=.2,ax=axes[counti, count]) #.set_title("best customers")
        counti += 1 
    
    fig.savefig('RFM_stackedplot_'+label+'.png', dpi=300)
    
    return ax, fig



def RFM_RF_all(orders, set_date ='2018-04-11' ):
        
    # 情況一：日期有錯，請改成2018年
    orders['orderdate'] =  orders['orderdate'].str.replace('2017', '2018')
    
    # 情況二：將gender改變成male與female
    orders['gender'] = orders['gender'].str.replace('男性', 'male')
    
    # 問題：將女性改female
    orders['gender'] = orders['gender'].str.replace('女性', 'female')
    
    # 設定今天的日期： 2018/4/11
    '從今天來看過往的銷售狀況'
    from dateutil import parser
    assess_date = parser.parse(set_date)
    
    type(assess_date)
    type('2018-04-11')
    
    # columns
    orders.columns
    
    # 計算每個人在banana、water與milk的消費數量
    orders['values'] = 1
    
    df2 = orders.pivot_table(index=['clientId','gender','orderdate'],
                              columns='product',aggfunc=len, values='values').fillna(0).reset_index()
    
    # frequency
    '指消費者在一定期間內購買該產品的次數'
    df2['frequency'] = 1
    frequency = df2.groupby("clientId", as_index = False)['frequency'].sum()
    del df2['frequency']
    
    # recency
    '指消費者至今再次購買與上次購買產品的時間差'
    df2['orderdate'] = pd.to_datetime(df2['orderdate'])
    recent_recency = df2.groupby("clientId", as_index = False)['orderdate'].max()
    recent_recency['recency'] =(assess_date - recent_recency['orderdate'] ).astype(str)
    
    # 問題：如何將recent_recency['recency']的days去除，並轉換為 int? 
    recent_recency['recency'] = recent_recency['recency'].str.replace('days', '').str.replace('0', '').str.replace(':', '').str.replace('.', '').str.replace(' ', '')
    #recent_recency['recency'] = recent_recency['recency'].str.replace('days.*', '', regex=True)
    recent_recency['recency'] = pd.to_numeric(recent_recency['recency']).fillna(0)
    
    # merge
    df2 = recent_recency.merge(df2, on = ['clientId', 'orderdate'])
    
    # frequency merge
    # 問題：如何將df2與frequency合併再一起？
    df2 =df2.merge(frequency, on = ['clientId'])
    
    # 切割 recency
    recency_label =  ['0-7 day', '8-15 day', '16-22 day', '23-30 day', '31-55 day', '>55 day']
    recency_cut  = [-1, 7, 15, 22, 30, 55, df2['recency'].max()]
    df2['recency_cate'] = pd.cut( df2['recency'] ,recency_cut, labels =recency_label)
    
    # 切割 frequency
    # 問題：請切成頻率  [0, 1, 2, 3, 4, 5, >5]； labels 改成 ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']
    frequency_label =  ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']
    frequency_cut  = [0, 1, 2, 3, 4, 5, df2['frequency'].max()]
    df2['frequency_cate'] = pd.cut( df2['frequency'] , frequency_cut , labels =frequency_label )
    
    
    # RF交叉分析
    df2['frequency_cate']= pd.Categorical(df2['frequency_cate'], categories=frequency_label[::-1])
    df2['recency_cate']= pd.Categorical(df2['recency_cate'], categories=recency_label[::-1])
    
    
    ####### RFM 分析 ####### 
    RF_table = pd.crosstab(df2['frequency_cate'],
                           df2['recency_cate'])
    
    RF_table.to_csv('RF_table.csv', encoding = 'cp950')
    # 以個別消費者來說，這四種顧客分別是個體的誰？
    import numpy as np
    df2['customer'] = np.where( (df2['frequency'] >=frequency_cut[4]) & (df2['recency']<=recency_cut[3]), 'best customers',
       
                         np.where( (df2['frequency'] >=frequency_cut[4]) & ( df2['recency']>recency_cut[3]), 'former best customers',
                                  
                                  np.where( (df2['frequency'] < frequency_cut[4]) & ( df2['recency']>recency_cut[3]), 'onetime customers',
                                           
                                           'new customers'  )))
    
    
    # 製圖
    
    # 四種顧客圖
    #df3 = pd.melt(df2[['clientId',"milk","water", "banana", 'customer',
    #                 'recency_cate','frequency_cate','gender' ]], id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values')
    
    #df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values')
    
    # 顧客產品推薦圖 - 長條圖
    import util
    util.RFM_plot_grid(df2[df2['gender']=='male'], frequency_label,recency_label,'male')
    
    # 問題：female
    util.RFM_plot_grid(df2[df2['gender']=='female'],frequency_label,recency_label,'female')
    
    # 顧客產品推薦圖 - 堆疊圖
    util.RFM_stackedplot(df2, frequency_label,recency_label,'gender')
    
    
    
    
    # 活躍指數
    act = orders.pivot_table(index=['clientId','gender','orderdate'],
                              columns='product',aggfunc=len, values='values').fillna(0).reset_index()
    
    act['orderdate'] = pd.to_datetime(act['orderdate'])
    aa= act.groupby('clientId', as_index= False)['orderdate'].diff().reset_index()
    aa =aa[['orderdate']]
    aa.columns = ['diff']
    
    aa = pd.concat([act, aa], axis = 1)
    aa = aa.dropna()
    aa['diff'] =aa['diff'].astype('timedelta64[D]')
    
    average = aa.groupby('clientId', as_index= False)['diff'].mean()
    aa['day_diff_org'] =  (assess_date - aa['orderdate']).astype('timedelta64[D]')
    aa['day_diff'] = (aa['day_diff_org'].max() - aa['day_diff_org']) 
    aa['weighted'] = aa['day_diff']  * aa['diff']
    
    weighted_all = aa.groupby('clientId', as_index= False)['weighted'].sum()
    weighted_para = aa.groupby('clientId', as_index= False)['day_diff'].sum()
    all  =weighted_all.merge(weighted_para ,on ='clientId')
    all = all.merge(average, on = 'clientId')
    all['cai'] = (all['diff'] - all['weighted'] /all['day_diff']) / all['diff'] 
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all['cai'].values.reshape(-1, 1))
    all['cai2'] = scaler.transform(all['cai'].values.reshape(-1, 1))
    
    all = all.merge(df2, on = 'clientId')
    all.to_csv('顧客分析資料.csv', encoding = 'cp950')


def logistic_model(X_train,y_train,X_test,y_test,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
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
    
    # model_profit 
    model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
    
    model_profit_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,1],conf_logist[1,1], conf_logist[::,1].sum(), '-'],
            '小計' : [sales_price* conf_logist[1,1], product_cost* conf_logist[1,1], marketing_expense * conf_logist[::,1].sum(),model_profit  ],
            })
    
    
    # all_profit 
    all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
    
    all_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
            '小計' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
            })
    
    
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
    
    
    # profit comparison
    if model_profit - all_profit > 0 :
        print( '------------------模型相對權市場行銷來說【賺錢】------------------' )
        print( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
        print( '比較全市場行銷來說，淨利減少' + str( round(model_profit / all_profit, 3) ) + '倍')
        
    else:
        print( '------------------模型相對權市場行銷來說【賠錢】------------------' )
        print( '模型比全市場行銷損失 $' + str(model_profit - all_profit) )
        print( '比較全市場行銷來說，淨利增加' + str( round(model_profit / all_profit, 3) ) + '倍')
    
    print( '------------------全市場行銷利潤矩陣------------------' )
    print(all_df)
    all_df
    all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv', encoding = 'cp950')
    print('全市場行銷利潤矩陣.xlsx saved')
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv', encoding = 'cp950')
    print('模型行銷利潤矩陣.xlsx saved')
    
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
#    print(result_df)
    result_df.to_csv(plot_name+'重要變數表.csv', encoding = 'cp950')
    return all_df, model_profit_df, result_df,y_test_df
    

def logistic_importance(
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
    result_df = results_summary_to_dataframe(result, plot_name = plot_name)
    return result_df 


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
    
    
    results_df = results_df[results_df['p_values']<0.05]
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

import itertools

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
    tick_marks  =np.array([tick_marks[0]-0.5, tick_marks[1]+0.5])
    plt.yticks(tick_marks, classes, rotation=1)

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

# transform_fb_date
def transform_fb_date(data = None, columns = 'created_time', splitby = '-'):
    
    data[columns] = data[columns].str.replace('T','-')
    data[columns] = data[columns].str.replace(':','-')
    data[columns] = data[columns].str.split('+').str[0]
    
    col_num =  data[columns][0].count('-')+1
    date = data[columns].str.split('-', col_num, expand=True)
    date.columns = ['year', 'month', 'day', 'hour', 'min', 'sec']
    
    combine= pd.concat([data, date ], axis = 1) #keys=[],names=['gg', 'example'],
    #combine.columns = [columns + '_cut', columns]
    
    # drop original col
    fields_to_drop = [columns]
    combine = combine.drop( fields_to_drop, axis = 1 )
    return combine


#
#from docx import Document
#from docx.shared import Cm

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


#def model_testRF(clf, X_train,y_train,X_test,y_test,sales_price = 3500,
#                    marketing_expense = 300,
#                    product_cost = 1750,
#                    plot_name = 'logistic_regression') :
#    model = clf.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    y_pred_prob = model.predict_proba(X_test)[:,1]
#    
#    y_test_df=pd.DataFrame(y_test)
#    y_test_df[plot_name+'_pred'] = y_pred_prob
#    
#    #Confusion Matrix
#    conf_logist = confusion_matrix(y_test, y_pred)
#    
#    # 畫conf matrix
#    plot_confusion_matrix(conf_logist, ['no','buy'],
#                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
#    
#    # -------single model summary--------
#    
#
#    print( "################ summary ################ ")
#    
#    print(confusion_matrix(y_test, y_pred))
##    print("____________________{}分類報告____________________".format(plot_name))
##    print(classification_report(y_test, y_pred))
#    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
#    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))

def model(clf, X_train,y_train,X_test,y_test,sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression') :
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
    
    feat_imp = model.feature_importances_
    feat = X_train.columns.tolist()
    res_data = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=True)
    res_data.plot('Features', 'Importance', kind='barh', title='Feature Importances',stacked=True, figsize = (15,10))
    plt.ylabel(plot_name+' Feature Importance Score')
    plt.savefig(plot_name+'.png', dpi=300)
    plt.show()
    
    
    res_data = res_data.sort_values('Importance',ascending=False)
    
    # > 50%
    res_data = res_data[res_data['Importance']>res_data['Importance'].describe()['50%']]
    
    
    
    # model_profit 
    model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
    
    model_profit_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,1],conf_logist[1,1], conf_logist[::,1].sum(), '-'],
            '小計' : [sales_price* conf_logist[1,1], product_cost* conf_logist[1,1], marketing_expense * conf_logist[::,1].sum(),model_profit  ],
            })
    
    
    # all_profit 
    all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
    
    all_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
            '小計' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
            })

    # -------single model summary--------
    
    # 創造docx
#    doc = Document()
    
    # save the doc

    print( "################ summary ################ ")
#    doc.add_heading("################ summary ################ ", 0)
    
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
#    doc.add_paragraph("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
#    doc.add_paragraph("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))    
    
    
    # importance
    print( '------------------ 應注意之變數 ------------------' )
#    doc.add_heading('------------------ 應注意之變數 ------------------', 1)
                    
    print(res_data)
    print('\n'.join(res_data["Features"].tolist()))
#    docx_table(res_data,doc )
#    doc.add_paragraph('\n'.join(res_data["Features"].tolist()))
    
    # profit comparison
    if model_profit - all_profit > 0 :
        print( '------------------模型相對權市場行銷來說【賺錢】------------------' )
        print( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
        
#        doc.add_heading('------------------模型相對權市場行銷來說【賺錢】------------------',1)
#        doc.add_paragraph( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
        
        if all_profit<0:
            all_profit2 = model_profit - all_profit
#            print( '比較全市場行銷來說，行銷費用少' + str( round(model_profit / all_profit2, 3) ) + '倍')
        else:
            print( '比較全市場行銷來說，淨利增加' + str( round(model_profit / all_profit, 3) ) + '倍')
        
    else:
        print( '------------------模型相對權市場行銷來說【賠錢】------------------' )
        print( '模型比全市場行銷損失 $' + str(model_profit - all_profit) )
        print( '比較全市場行銷來說，淨利減少' + str( round(model_profit / all_profit, 3) ) + '倍')
#        doc.add_heading('------------------模型相對權市場行銷來說【賺錢】------------------',1)
#        doc.add_paragraph( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
        
        
        
        
    print( '------------------全市場行銷利潤矩陣------------------' )
    print(all_df)
    all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv', encoding = 'cp950')
    print('全市場行銷利潤矩陣.xlsx saved')
    
#    doc.add_heading('------------------全市場行銷利潤矩陣------------------',1)
#    docx_table(all_df,doc )
        
    
    
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv', encoding = 'cp950')
    print('模型行銷利潤矩陣.xlsx saved')
    
#    doc.add_heading( '------------------' +plot_name+ '模型行銷利潤矩陣------------------',1)
#    docx_table(model_profit_df,doc )
        
    
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
#    print(result_df)
    res_data.to_csv(plot_name+'重要變數表.csv', encoding = 'cp950')
    
    
#    doc.add_heading(  '------------------' +plot_name+ '重要變數表------------------',1)
#    docx_table(res_data,doc )
    
    # pic
#    doc.add_picture(plot_name+'.png')
#    doc.add_picture(plot_name+"Confusion Matrix plot.png")
#
#    doc.save(plot_name+'.docx')
#    
    return all_df, model_profit_df, res_data, y_test_df
    

def model_profit_fun(clf, X_train,y_train,X_test,y_test,sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression') :
    
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    y_test_df.columns = ['buy', plot_name+'_pred']
    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
    # 畫conf matrix
    plot_confusion_matrix(conf_logist, ['no','buy'],
                          title=plot_name+"Confusion Matrix plot", cmap=plt.cm.Reds)#, cmap=plt.cm.Reds
    
    
    # model_profit 
    model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
    
    model_profit_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,1],conf_logist[1,1], conf_logist[::,1].sum(), '-'],
            '小計' : [sales_price* conf_logist[1,1], product_cost* conf_logist[1,1], marketing_expense * conf_logist[::,1].sum(),model_profit  ],
            })
    
    
    # all_profit 
    all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
    
    all_df = pd.DataFrame({
            '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
            '金額' : [sales_price,product_cost, marketing_expense, '-'],
            '目標對象' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
            '小計' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
            })

    # -------single model summary--------
    
    # 創造docx
#    doc = Document()
    
    # save the doc

    print( "################ summary ################ ")
    
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
    
    # profit comparison
    if model_profit - all_profit > 0 :
        print( '------------------模型相對權市場行銷來說【賺錢】------------------' )
        print( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
        
#        
#        if all_profit<0:
#            all_profit2 = model_profit - all_profit
#        else:
#            print( '比較全市場行銷來說，淨利增加' + str( round(model_profit / all_profit, 3) ) + '倍')
        
    else:
        print( '------------------模型相對權市場行銷來說【賠錢】------------------' )
        print( '模型比全市場行銷損失 $' + str(model_profit - all_profit) )
        print( '比較全市場行銷來說，淨利減少' + str( round(model_profit / all_profit, 3) ) + '倍')
        
        
        
        
    print( '------------------全市場行銷利潤矩陣------------------' )
    print(all_df)
    all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv', encoding = 'cp950')
    print('全市場行銷利潤矩陣.xlsx saved')
    
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv', encoding = 'cp950')
    print('模型行銷利潤矩陣.xlsx saved')
    
    # 儲存模型
    pickle.dump(model, open(plot_name+"_model.dat", "wb"))
    
    # 顧客產品推薦名單
    y_test_df = y_test_df.sort_values([plot_name+'_pred'], ascending = False)
    y_test_df.to_csv(plot_name+'顧客產品推薦名單.csv', encoding = 'cp950')
    
    return all_df, model_profit_df,  y_test_df
    
def client_list_all(model_name, dataname, name):
        
    # 載入模型
    with open(model_name, "rb") as f:
        model_xgb = pickle.load(f)
    
    # 讀取資料
    newdata = pd.read_csv(dataname)
    
    # 保留UID
    newdata_uid = newdata['UID']
    
    # 設定xgb 分類模型
    del newdata['UID']
    
    # 產出【顧客產品推薦名單】
    c_list = client_list(model=model_xgb, y_test=None, X_test=newdata, test_uid=newdata_uid,name=name)
    return c_list

# function
import platform
import os
# 顧客產品推薦名單 function
def client_list(model, y_test, X_test, test_uid, name):
    '''
    model:要放入的模型
    y_test：要驗證的資料集，如果沒有，請設定None
    X_test:要預測的資料集
    test_uid:這資料集的UID
    
    '''
    
    # prediction
    xgb_pred_prob = model.predict_proba(X_test)
    
    # 產出【顧客產品推薦名單】
    if isinstance(y_test, pd.Series) :
        XGBClassifier_test_df=pd.DataFrame(y_test.values ,columns =['客戶對A商品【實際】購買狀態'])
        XGBClassifier_test_df['客戶對A商品【預測】購買機率'] = xgb_pred_prob[:,1]
        test_uid = test_uid.reset_index().drop(columns = ['index'])
        XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)
        
    else:
        XGBClassifier_test_df= pd.DataFrame( xgb_pred_prob[:,1],columns =['客戶對A商品【預測】購買機率'])
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values(['客戶對A商品【預測】購買機率'], ascending = False)
        test_uid = test_uid.reset_index().drop(columns = ['index'])
        XGBClassifier_test_df = pd.concat([test_uid,XGBClassifier_test_df], axis = 1)
        XGBClassifier_test_df = XGBClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)
    
    if platform.system() =='Windows':
        XGBClassifier_test_df.to_csv(name+'_顧客產品推薦名單.csv',
                                     encoding='UTF-8-sig')
    else:
        XGBClassifier_test_df.to_csv(name+'_顧客產品推薦名單.csv')
        
    return XGBClassifier_test_df 



def model_profit_newdata_fun(clf, X_train,y_train,X_test,y_test,
                             train = False,
                             model_name='a.dat',
                             test_uid='a',
                             percent=.5,
                             sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression') :
    
    if train:
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
        
        
        # model_profit 
        model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
        
        model_profit_df = pd.DataFrame({
                '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
                '金額' : [sales_price,product_cost, marketing_expense, '-'],
                '目標對象' : [conf_logist[1,1],conf_logist[1,1], conf_logist[::,1].sum(), '-'],
                '小計' : [sales_price* conf_logist[1,1], product_cost* conf_logist[1,1], marketing_expense * conf_logist[::,1].sum(),model_profit  ],
                })
        
        
        # all_profit 
        all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
        
        all_df = pd.DataFrame({
                '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
                '金額' : [sales_price,product_cost, marketing_expense, '-'],
                '目標對象' : [conf_logist[1,::].sum(), conf_logist[1,::].sum(), conf_logist.sum(), '-'],
                '小計' : [sales_price*conf_logist[1,::].sum(), product_cost* conf_logist[1,::].sum(), marketing_expense *  conf_logist.sum(),all_profit  ],
                })
    
            
        print( '------------------全市場行銷利潤矩陣------------------' )
        print(all_df)
        all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv', encoding = 'cp950')
        print('全市場行銷利潤矩陣.xlsx saved')
    
        # -------single model summary--------
        
        # 創造docx
    #    doc = Document()
        
        # save the doc
    
        print( "################ summary ################ ")
        
        print(confusion_matrix(y_test, y_pred))
        print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
        print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
        
        # profit comparison
        if model_profit - all_profit > 0 :
            print( '------------------模型相對權市場行銷來說【賺錢】------------------' )
            print( '模型比全市場行銷賺 $' + str(model_profit - all_profit) )
            
    #        
    #        if all_profit<0:
    #            all_profit2 = model_profit - all_profit
    #        else:
    #            print( '比較全市場行銷來說，淨利增加' + str( round(model_profit / all_profit, 3) ) + '倍')
            
        else:
            print( '------------------模型相對權市場行銷來說【賠錢】------------------' )
            print( '模型比全市場行銷損失 $' + str(model_profit - all_profit) )
            print( '比較全市場行銷來說，淨利減少' + str( round(model_profit / all_profit, 3) ) + '倍')
            
        print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
        print(model_profit_df)
        model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv', encoding = 'cp950')
        print('模型行銷利潤矩陣.xlsx saved')
        return all_df, model_profit_df,  y_test_df
    
    else:
         # 載入模型
        import pickle
        with open(model_name, "rb") as f:
            model_xgb = pickle.load(f)
            
            
        XGBClassifier_test_df, xgb_pred = client_list(model_xgb, y_test, X_test, test_uid,plot_name)
        cou=np.unique(xgb_pred, return_counts=True)
        
        
        # model_profit 
        model_profit = sales_price * cou[1][1]* percent -  cou[1][1]* marketing_expense - product_cost * cou[1][1]* percent
        
        model_profit_df = pd.DataFrame({
                '項目' : ['單品價格', '單品營業成本', '單品行銷費用', '利潤'],
                '金額' : [sales_price,product_cost, marketing_expense, '-'],
                '目標對象' : [cou[1][1]* percent, cou[1][1]* percent,  cou[1][1], '-'],
                '小計' : [sales_price * cou[1][1]* percent, product_cost * cou[1][1]* percent,  cou[1][1]* marketing_expense,model_profit  ],
                })
        
    
    
        print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
        print(model_profit_df)
        model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv', encoding = 'cp950')
        print('模型行銷利潤矩陣.xlsx saved')
        return model_profit_df,  XGBClassifier_test_df
        
    

# part 獲利閥值長條圖
def profit_linechart(y_test_df,
                     
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression' ):
    
    
    profit_line = []
    for i in np.arange(0,1,0.01):
        
        # set threshold
        y_test_df['pred_yn']= np.where(y_test_df[y_test_df.columns[1]]>=i, 1,0)    
        
        
        conf_logist = confusion_matrix(y_test_df['buy'], y_test_df['pred_yn'])
        
        # model_profit 
        model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
        
        
        # all_profit 
        all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
        
        # 將所有threshold append在一起
        profit_line.append([ i,model_profit, all_profit])
    
    profit_line = pd.DataFrame(profit_line, columns= ['閥值', plot_name, '全市場'])
    
    # draw
    X_max = profit_line[profit_line[plot_name] ==profit_line[plot_name].max()]['閥值']
    Y_max =  profit_line[plot_name].max()
    
    profit_line = profit_line.rename( columns= {'閥值': 'threshold', '全市場' : 'all_market'})
    profit_line.plot.line(x='threshold', y=[plot_name,     'all_market'],figsize=(15,10))
    plt.scatter(X_max,Y_max, c='red', marker='o',alpha=0.5)
    plt.text(X_max-0.005, Y_max+10000, plot_name+' best profit$ ' + str(Y_max ) + ', threshold='+  str(X_max.values[0]))
    plt.ylabel('expected profit')
    plt.savefig(plot_name+'_預期獲利最佳化模型與閥值折線圖.png', dpi=300)
    plt.show()
    
#    document = Document()
#    document.add_heading(plot_name+'_預期獲利最佳化模型與閥值折線圖.png', 0)
#    document.add_paragraph('台灣第一個行銷資料科學(MDS)知識部落\n\n本粉絲專頁在探討資料科學之基礎概念、趨勢、新工具和實作，讓粉絲們瞭解資料科學的行銷運用,並開啟厚植數據分析能力之契機')
#    document.add_picture(plot_name+'_預期獲利最佳化模型與閥值折線圖.png')
#    document.save(plot_name+'_預期獲利最佳化模型與閥值折線圖')




def cut_off_calu(y_test_df,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression'):
    profit_line = []
    for i in np.arange(0,1,0.01):
        
        # set threshold
        y_test_df[plot_name+'pred_yn']= np.where(y_test_df[plot_name]>=i, 1,0)    
        
        
        conf_logist = confusion_matrix(y_test_df['buy'], y_test_df[plot_name+'pred_yn'])
        
        # model_profit 
        model_profit = sales_price * conf_logist[1,1] - conf_logist[::,1].sum() * marketing_expense - product_cost * conf_logist[1,1]
        
        
        # all_profit 
        all_profit = sales_price*conf_logist[1,::].sum() - product_cost* conf_logist[1,::].sum()- marketing_expense *  conf_logist.sum()
        
        # 將所有threshold append在一起
        profit_line.append([ i,model_profit, all_profit])
    
    profit_line = pd.DataFrame(profit_line, columns= ['閥值', plot_name, '全市場'])
    return profit_line 
    
# part 獲利閥值長條圖
def profit_linechart_all(y_test_df ,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800
                    ):
    
    allcon  = []
    for i in y_test_df:
            
        profit_line = cut_off_calu(i,
                        sales_price = sales_price,
                        marketing_expense = marketing_expense,
                        product_cost = product_cost,
                        plot_name = i.columns[1])
        allcon.append(profit_line)
        
    
    from functools import reduce
    allcondf= reduce(lambda x, y: pd.merge(x, y, on = ['閥值', '全市場']), allcon)
        
    # draw
    
    
        
    allcondf = allcondf.rename( columns= {'閥值': 'threshold', '全市場' : 'all_market'})
#    profit_line.plot.line(x='threshold', y=[plot_name,     'all_market'],figsize=(15,10))
    
    allcondf.plot.line(x='threshold', y=allcondf.drop(columns='threshold').columns.tolist(), figsize=(15,10))
    
    for plot_name in allcondf.drop(columns='threshold').columns.tolist():
        
        # 因為全市場沒有最高點
        if not plot_name=='all_market':
            X_max = allcondf[allcondf[plot_name] ==allcondf[plot_name].max()]['threshold']
            Y_max =  allcondf[plot_name].max()
            plt.scatter(X_max.values[0],Y_max, c='red', marker='o',alpha=0.5)
            
            if 'Random' in plot_name:
                
                plt.text(X_max.values[0]-0.005, Y_max+Y_max * 0.02 , plot_name.replace('_pred','')+' best profit $ ' + str(Y_max ) + ', threshold='+  str(X_max.values[0]))
    
            elif 'xgb' in plot_name:
                plt.text(X_max.values[0]-0.005, Y_max+Y_max * 0.02, plot_name.replace('_pred','')+' best profit $ ' + str(Y_max ) + ', threshold='+  str(X_max.values[0]))
            
            else:
                plt.text(X_max.values[0]-0.005, Y_max + Y_max * 0.02 , plot_name.replace('_pred','')+' best profit $ ' + str(Y_max ) + ', threshold='+  str(X_max.values[0]))
    
            
    
    plt.ylabel('expected profit')
    plt.savefig('預期獲利最佳化模型與閥值折線圖'+'.png', dpi=300)
    plt.show()


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *

def classification_plot(classifier, X_train, y_train):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Random Forest Classification (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Stay_time')
    plt.legend()
    plt.show()
    

def plot_model_active(X_train, y_train, classifier, name):


    X_set, y_set = X_train, y_train
    
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
            
    pred_one = classifier.predict_proba(X_train)[::,1]
    
    try:
            
        trace0 = go.Scatter3d(
            x = X_set[y_set == 0, 0],
            y = X_set[y_set == 0, 1],
            z = pred_one[y_set ==0],
            name = 'Above',
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'red',
                line = dict(
                    width = 2,
                    color = 'rgb(0, 0, 0)'
                )
            )
        )
        
        trace1 = go.Scatter3d(
            x = X_set[y_set == 1, 0],
            y = X_set[y_set == 1, 1],
            z = pred_one[y_set ==1] ,#y_set[y_set ==1],
            name = 'Above',
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'green',
                line = dict(
                    width = 2,
                    color = 'rgb(0, 0, 0)'
                )
            )
        )
              
        data1 =   go.Surface(
            x= X1,            y= X2,
            z= classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape).tolist(),
            colorbar=dict(
                thickness=25,
                thicknessmode='pixels',
                len=0.9,
                lenmode='fraction',
                outlinewidth=0
            )
        )
            
        data = [trace0,trace1,data1]
        
            
        layout = go.Layout(
                title=name,
                 scene=Scene(
                    xaxis=XAxis(title='年齡'),
                    yaxis=YAxis(title='停留時間'),
                    zaxis=ZAxis(title='機率')
                ),
                autosize=False,
                width=700,
                height=700,
                margin=dict(
                    l=65,
                    r=50,
                    b=65,
                    t=90
                )
            )
        
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=name)
        
    except:
        print('plotly套件沒有安裝！')
        