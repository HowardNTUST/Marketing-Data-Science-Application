#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:33:20 2018

@author: slave1
"""
# import sys, importlib
# importlib.reload(sys.modules['util'])
from sklearn.metrics import confusion_matrix, auc, accuracy_score
import pandas as pd
import numpy as np
# import statsmodels.api as sm
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ----設定繪圖-------
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 

import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



def transform_to_category(str_columns, data):
    for c in str_columns:
        data[c] = data[c].astype('category')
    return data


# part 獲利閥值長條圖


def profit_linechart_ready_for_web(y_test_df ,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800,
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
    return allcondf
        


def profit_linechart_all_web(y_test_df ,
                    sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1800,
                    plot_name_main= '正常'
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
        
    
        
    allcondf = allcondf.rename( columns= {'閥值': 'threshold', '全市場' : 'all_market'})
    
    # draw
    fig = go.Figure()


    # Add traces
    model_line = [i for i in allcondf.columns if 'threshold' not in i]
    for i in model_line:

        # plot lines 
        fig.add_trace(go.Scatter(x=allcondf['threshold'], y=allcondf[i],
                            mode='lines',
                            name=i))

        if not i=='all_market':
            # plot text
            X_max = allcondf[allcondf[i] == allcondf[i].max()]['threshold'].values[0]
            Y_max =  allcondf[i].max()

            # 處理呈現字詞
            classifier_text = i.replace('_pred', '')
        
            # 加入字詞 
            fig.add_annotation(
                x=X_max,
                y=Y_max,
                text=classifier_text + "最好利潤值 = $" + str(Y_max)+ '; '+'閾值 = ' + str(X_max*100) + '%' ) # change here

            # 加入利潤最好的圖示點
            fig.add_trace(go.Scatter(
                            x = [X_max], 
                            y = [Y_max],
                            mode='markers',
                            # name=i
                            )
                            )

            # fig.update_annotations(dict(
            #             xref="x",
            #             yref="y",
            #             showarrow=True,
            #             arrowhead=3,
            # ))
    fig.update_layout(
        title="預期獲利最佳化模型折線圖",
        xaxis_title="閾值",
        yaxis_title="預期利潤",
        legend_title="模型",
        )


    iplot(fig)
    plot(fig, filename='【' + plot_name_main  + '】 - ' +'預期獲利最佳化模型折線圖.html')



    # fig = plt.figure(figsize=(15,10))
    # plt.rcParams['figure.facecolor'] = 'white'
    
    # for i in allcondf.drop(columns='threshold').columns.tolist():
    #     plt.plot(allcondf['threshold'],allcondf[i] )
    
    # for plot_name in allcondf.drop(columns='threshold').columns.tolist():
        
    #     # 因為全市場沒有最高點
    #     if not plot_name=='all_market':
    #         X_max = allcondf[allcondf[plot_name] ==allcondf[plot_name].max()]['threshold']
    #         Y_max =  allcondf[plot_name].max()
    #         plt.scatter(X_max.values[0],Y_max, c='red', marker='o',alpha=0.5)
            
    #         if 'Random' in plot_name:
                
    #             plt.text(X_max.values[0]-0.005, Y_max+Y_max * 0.02 , plot_name.replace('_pred','')+' 最好的利潤 $ ' + str(Y_max ) + ', 機率閾值='+  str(X_max.values[0]),size= 25)
    
    #         elif 'xgb' in plot_name:
    #             plt.text(X_max.values[0]-0.005, Y_max+Y_max * 0.02, plot_name.replace('_pred','')+' 最好的利潤 $ ' + str(Y_max ) + ', 機率閾值='+  str(X_max.values[0]),size= 25)
            
    #         else:
    #             plt.text(X_max.values[0]-0.005, Y_max + Y_max * 0.02 , plot_name.replace('_pred','')+' 最好的利潤 $ ' + str(Y_max ) + ', 機率閾值='+  str(X_max.values[0]),size= 25)
    
            
    # plt.ylabel('預期獲利',size = 20)
    # plt.xlabel('購買機率',size = 20)
    # plt.title('【' + plot_name_main + '】 - ' + '獲利最佳化折線圖',size = 35)
    # plt.savefig('預期獲利最佳化模型與閥值折線圖'+'.png', dpi=300)
    # plt.show()
    
    # plotly_fig = tls.mpl_to_plotly(fig)
    # plot(plotly_fig, filename='【' + plot_name_main  + '】 - ' +'預期獲利最佳化模型折線圖.html')
    # print('完成~！ 【' + plot_name_main + '】 - ' + '獲利最佳化折線圖')


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

def move_file(dectect_name, folder_name):
    '''
    dectect_name:
        
    folder_name:
        
    '''    
    # 抓出為【正常模型】的所有檔案名稱
    import os 
    save = []
    for i in os.listdir():
        if dectect_name in i:
            save.append(i)
    
    # save=[i for i in os.listdir() if plot_name2 in i]
    
    # make folder
    ff = [i for i in save if not '.' in i ]
    ff = [i for i in ff if  '（' in i ]
    
    
            
    try:
        os.makedirs(folder_name)
        folder_namenew= folder_name
    
    except:
        
        try:
            os.makedirs(folder_name + '（' +str(0)+'）')
            folder_namenew= folder_name + '（' +str(0)+'）'
        except: 
            
            for i in range(0, 10):
                iinn = [j for j in ff if folder_name + '（' +str(i)+'）'  in j]
                if len(iinn) == 0:
                    os.makedirs(folder_name + '（' +str(i)+'）')
                    folder_namenew =folder_name + '（' +str(i)+'）'
                    break
                
                # break
        
    
    
    # move files to that created folder
    import shutil
    save = [i for i in save if '.' in i ]
    for m in save:
        shutil.move(m, folder_namenew)

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
    all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('全市場行銷利潤矩陣.csv saved')
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('模型行銷利潤矩陣.csv saved')
    
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
#    print(result_df)
    result_df.to_csv(plot_name+'重要變數表.csv',encoding = 'utf-8-sig')
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
    tick_marks  =np.array([tick_marks[0], tick_marks[1]])
    plt.yticks(tick_marks, classes, rotation=1)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=8)
    plt.xlabel('Predicted label',fontsize=8)
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
    all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('全市場行銷利潤矩陣.csv saved')
    
#    doc.add_heading('------------------全市場行銷利潤矩陣------------------',1)
#    docx_table(all_df,doc )
        
    
    
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('模型行銷利潤矩陣.csv saved')
    
#    doc.add_heading( '------------------' +plot_name+ '模型行銷利潤矩陣------------------',1)
#    docx_table(model_profit_df,doc )
        
    
    
    print( '------------------' +plot_name+ '重要變數表------------------' )
#    print(result_df)
    res_data.to_csv(plot_name+'重要變數表.csv',encoding = 'utf-8-sig')
    
    
#    doc.add_heading(  '------------------' +plot_name+ '重要變數表------------------',1)
#    docx_table(res_data,doc )
    
    # pic
#    doc.add_picture(plot_name+'.png')
#    doc.add_picture(plot_name+"Confusion Matrix plot.png")
#
#    doc.save(plot_name+'.docx')
#    
    return all_df, model_profit_df, res_data, y_test_df
    

def model_profit_fun(clf, X_train,y_train,X_test,y_test,test_uid,
                     sales_price = 3500,
                    marketing_expense = 300,
                    product_cost = 1750,
                    plot_name = 'logistic_regression') :
    
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    y_test_df=pd.DataFrame(y_test)
    y_test_df[plot_name+'_pred'] = y_pred_prob
    
    y_test_df.columns = ['buy', plot_name+'_pred']
    y_test_df = pd.concat([test_uid, y_test_df], axis=1)

    
    #Confusion Matrix
    conf_logist = confusion_matrix(y_test, y_pred)
    
    # 畫conf matrix
    plot_confusion_matrix(conf_logist, ['No','Buy'],
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

    print( "################ 摘要 ################ ")
    
    print(confusion_matrix(y_test, y_pred))
    print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
    print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
    
    # profit comparison
    if model_profit - all_profit > 0 :
        print( '------------------模型相對全市場行銷來說【賺錢】------------------' )
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
    all_df.to_excel(plot_name+'全市場行銷利潤矩陣.xlsx')
    # to_csv(plot_name+'全市場行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('全市場行銷利潤矩陣.xlsx saved')
    
    
    print( '------------------' +plot_name+ '模型行銷利潤矩陣------------------' )
    print(model_profit_df)
    model_profit_df.to_excel(plot_name+'模型行銷利潤矩陣.xlsx')
    # to_csv(plot_name+'模型行銷利潤矩陣.csv',encoding = 'utf-8-sig')
    print('模型行銷利潤矩陣.xlsx saved')
    
    # 顧客產品推薦名單
    y_test_df = y_test_df.sort_values([plot_name+'_pred'], ascending = False)
    # y_test_df.to_csv(plot_name+'顧客產品推薦名單.csv',encoding = 'utf-8-sig')
    
    return all_df, model_profit_df,  y_test_df
    


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
    if y_test !=None:
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
    
    XGBClassifier_test_df.to_csv(name+'顧客產品推薦名單.csv',encoding = 'utf-8-sig')
    return XGBClassifier_test_df,xgb_pred


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
        all_df.to_csv(plot_name+'全市場行銷利潤矩陣.csv',encoding = 'utf-8-sig')
        print('全市場行銷利潤矩陣.csv saved')
    
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
        model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv',encoding = 'utf-8-sig')
        print('模型行銷利潤矩陣.csv saved')
        return all_df, model_profit_df,  y_test_df
    
    else:
         # 載入模型
        import pickle
        with open(model_name, "rb") as f:
            model_xgb = pickle.load(f)
            
            
        XGBClassifier_test_df, xgb_pred = client_list(model_xgb, y_test, X_test, test_uid,plot_name)
        cou=np.unique(xgb_pred, return_counts=True)
#        XGBClassifier_test_df['pred_final'] = xgb_pred
        
        
        try:
                
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
            model_profit_df.to_csv(plot_name+'模型行銷利潤矩陣.csv',encoding = 'utf-8-sig')
            print('模型行銷利潤矩陣.csv saved')
        except:
            model_profit_df = 'no'
        
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
        y_test_df['pred_yn']= np.where(y_test_df[y_test_df.columns[2]]>=i, 1,0)    
        
        
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
    plt.scatter(X_max.values[0],Y_max, c='red', marker='o',alpha=0.5)
    plt.text(X_max.values[0]-0.005, Y_max+10000, plot_name+' best profit$ ' + str(Y_max ) + ', threshold='+  str(X_max.values[0]))
    plt.ylabel('expected profit')
    plt.savefig(plot_name+'_預期獲利最佳化模型折線圖.png', dpi=300)
    plt.show()
    return X_max, Y_max
    
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
    
    