# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:27:34 2019

@author: howar
"""


import pandas as pd

def RFM_cal(orders,start_date, assess_date):
    orders['values'] = 1
    
    df2 = orders.pivot_table(index=['clientId','gender','orderdate'],
                              columns='product',aggfunc=len, values='values').fillna(0).reset_index()
    
    df2['orderdate'] = pd.to_datetime(df2['orderdate'])
    
    df2 = df2[ (df2['orderdate'] <= assess_date) &  (df2['orderdate'] >= start_date)]
    
    # frequency
    '指消費者在一定期間內購買該產品的次數'
    df2['frequency'] = 1
    frequency = df2.groupby("clientId", as_index = False)['frequency'].sum()
    del df2['frequency']
    
    # recency
    '指消費者至今再次購買與上次購買產品的時間差'
    df2['orderdate'] = pd.to_datetime(df2['orderdate'])
    recent_recency = df2.groupby("clientId", as_index = False)['orderdate'].max()
    recent_recency['recency'] =( assess_date - recent_recency['orderdate'] ).astype(str)
    
    # 問題：如何將recent_recency['recency']的days去除，並轉換為 int? 
    recent_recency['recency'] = recent_recency['recency'].str.replace('days.*', '', regex = True).astype(int)

    return df2,recent_recency, frequency


# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:27:34 2019

@author: howar
"""
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def customer_segmentation(df2,frequency_cut,recency_cut,product ):
        
    
    df2['customer'] = np.where( (df2['frequency'] >=frequency_cut[4]) & (df2['recency']<=recency_cut[3]), 'best customers',
       
                         np.where( (df2['frequency'] >=frequency_cut[4]) & ( df2['recency']>recency_cut[3]), 'former best customers',
                                  
                                  np.where( (df2['frequency'] < frequency_cut[4]) & ( df2['recency']>recency_cut[3]), 'onetime customers',
                                           
                                           'new customers'  )))
    
    recom_list = []
    for i in range(len(df2[product])):
        aa = df2[product].iloc[i,::].rank().sort_values(ascending = False)
        recom_list.append('、'.join(aa.index))
        
    df2['recommend_product'] =recom_list
    df2.to_csv('df_seg.csv', encoding = 'cp950')
    
    return df2


    
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
    recent_recency['recency'] =( assess_date - recent_recency['orderdate'] ).astype(str)
    
    # 問題：如何將recent_recency['recency']的days去除，並轉換為 int? 
    recent_recency['recency'] = recent_recency['recency'].str.replace('days.*', '', regex = True).astype(int)

    return df2,recent_recency, frequency

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
    
    df3 = pd.melt(df2.drop(columns = ['orderdate','recency','frequency']), id_vars=['clientId','customer','近因','頻率',label], var_name='types', value_name='values')
    df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
    df3 = df3.dropna()
    
    fig, axes = plt.subplots(6, 6, figsize=(25, 15))
#    plt.figlegend( [ax.legend()], 'label1', label = 'lower center', ncol=5, labelspacing=0.1 )
    
    counti = 0
    for i in frequency_label:
        count = 6
        for j in recency_label[::-1]:
            count -= 1 
            if df3[(df3['近因']==j) & (df3['頻率']==i)].shape[0] != 0:
                df4 = df3[(df3['近因']==j) & (df3['頻率']==i)]
                df4 = df4.groupby(['types', label]).agg({'values': 'sum'})
                df4 = df4.groupby(['types', label]).sum()
                df4 =df4.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))
                df4 = df4.add_suffix('').reset_index() #to df
                df4=df4.pivot(label, 'types', 'values')
                
                #draw
                ax = df4.plot.bar(stacked=True,width=0.7, legend = False, ax =axes[counti, count] ,rot=0)
                # ax.legend().set_visible(False)
                ax.legend( loc = 1, fontsize =5)                
                # sns.barplot(x="types", y="values", data=df3[(df3['近因']==j) & (df3['頻率']==i)], capsize=.2,ax=axes[counti, count]) #.set_title("best customers")
        counti += 1 
        
        
        
    # fig.legend( fontsize =8)                
    
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
    
    # 顧客產品推薦圖 - 長條圖
    RFM_plot_grid(df2[df2['gender']=='male'], frequency_label,recency_label,'male')
    
    # 問題：female
    RFM_plot_grid(df2[df2['gender']=='female'],frequency_label,recency_label,'female')
    
    # 顧客產品推薦圖 - 堆疊圖
    RFM_stackedplot(df2, frequency_label,recency_label,'gender')
    
    df2.to_csv('顧客分析資料.csv', encoding = 'cp950')
