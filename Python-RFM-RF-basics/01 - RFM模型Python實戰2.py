# -*- coding: utf-8 -*-
"""
Created on Fri Mar  13 00:47:18 2019

@author: Howard

"""

import pandas as pd
import seaborn as sns
import RFM

# 動態繪圖套件
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import plotly.express as px
import tools
import seaborn as sns

theOS, ecode = tools.checkPlatform()
''' 
資料讀取 
    注意編碼問題，尤其是windows作業系統出來的檔案，都有獨特的編碼格式。
'''
orders= pd.read_csv( 'orders2.csv', encoding=ecode)


#空值該列全部刪除
orders.dropna(inplace = True)

####### RFM 模型實戰演練 ####### 

####### 2.資料處理 ####### 

# 設定今天的日期： 2018/4/11
'從今天來看過往的銷售狀況'
from dateutil import parser
assess_date = parser.parse('2018-04-11')
start_date= parser.parse('2018-01-01')

type(assess_date)
type('2018-04-11')

# 計算每個人在banana、water與milk的消費數量與時間
df2, recent_recency, frequency = RFM.RFM_cal(orders,start_date,assess_date)

# merge recency
df2 = recent_recency.merge(df2, on = ['clientId', 'orderdate'])

# frequency merge
df2 =df2.merge(frequency, on = ['clientId'])


# 最近一次（天）的消費與顧客數量分佈圖
df2['顧客數量'] = 1
recency_table = df2.groupby('recency', as_index = False)['顧客數量'].sum()

fig = px.bar(recency_table, y='顧客數量', x='recency', text ='顧客數量' )
fig.update_traces(texttemplate='%{text:}人', textposition='outside')
fig.update_layout(
    title="最近一次（天）的消費與顧客數量分佈圖",
    xaxis_title="距離上次購買的天數",
    yaxis_title="顧客數量",
    font=dict(
        size=18,
    )
)

plot(fig, filename='最近一次（天）的消費與顧客數量分佈圖.html')


# 消費頻率與顧客數量分佈圖
df2['顧客數量'] = 1
frequency_table = df2.groupby('frequency', as_index = False)['顧客數量'].sum()

fig = px.bar(frequency_table, y='顧客數量', x='frequency', text ='顧客數量' )
fig.update_traces(texttemplate='%{text:}人', textposition='outside')
fig.update_layout(
    title="消費頻率與顧客數量分佈圖",
    xaxis_title="消費頻率",
    yaxis_title="顧客數量",
    font=dict(
        size=18,
    )
)

plot(fig, filename='消費頻率與顧客數量分佈圖.html')



# # ----------左偏示意----------# 

# # 最近一次（天）的消費與顧客數量分佈圖
# df2['顧客數量'] = 1
# recency_table = df2.groupby('recency', as_index = False)['顧客數量'].sum()
# recency_table['recency'] = recency_table['recency'].max() - recency_table['recency']

# fig = px.bar(recency_table, y='顧客數量', x='recency', text ='顧客數量' )
# fig.update_traces(texttemplate='%{text:}人', textposition='outside')
# fig.update_layout(
#     title="最近一次（天）的消費與顧客數量分佈圖（左偏分佈示意圖）",
#     xaxis_title="距離上次購買的天數",
#     yaxis_title="顧客數量",
#     font=dict(
#         size=18,
#     )
# )

# plot(fig, filename='最近一次（天）的消費與顧客數量分佈圖_左偏.html')


# # 消費頻率與顧客數量分佈圖
# df2['顧客數量'] = 1
# frequency_table = df2.groupby('frequency', as_index = False)['顧客數量'].sum()
# frequency_table['frequency'] = frequency_table['frequency'].max() - frequency_table['frequency']

# fig = px.bar(frequency_table, y='顧客數量', x='frequency', text ='顧客數量' )
# fig.update_traces(texttemplate='%{text:}人', textposition='outside')
# fig.update_layout(
#     title="消費頻率與顧客數量分佈圖（左偏分佈示意圖）",
#     xaxis_title="消費頻率",
#     yaxis_title="顧客數量",
#     font=dict(
#         size=18,
#     )
# )

# plot(fig, filename='消費頻率與顧客數量分佈圖_左偏.html')




# clustering 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 6, affinity='euclidean', linkage='ward')
recency_table['cluster'] = cluster.fit_predict(recency_table)
frequency_table['cluster'] = cluster.fit_predict(frequency_table)


# 【分群後】最近一次（天）的消費與顧客數量分佈圖
recency_table['cluster'] = recency_table['cluster'].astype(str)

fig = px.bar(recency_table, y='顧客數量', x='recency', text ='顧客數量' ,
             color =recency_table['cluster']  )
fig.update_traces(texttemplate='%{text:}人', textposition='outside')
fig.update_layout(
    title="【分群後】最近一次（天）的消費與顧客數量分佈圖",
    xaxis_title="距離上次購買的天數",
    yaxis_title="顧客數量",
    font=dict(
        size=18,
    )
)

plot(fig, filename='【分群後】最近一次（天）的消費與顧客數量分佈圖.html')



# 【分群後】消費頻率與顧客數量分佈圖
frequency_table['cluster'] = frequency_table['cluster'].astype(str)

fig = px.bar(frequency_table, y='顧客數量', x='frequency', text ='顧客數量' ,
             color=frequency_table['cluster'])
# ['red','blue','blue','blue','blue','blue','blue','blue','blue']
fig.update_traces(texttemplate='%{text:}人', textposition='outside')
fig.update_layout(
    title="【分群後】消費頻率與顧客數量分佈圖",
    xaxis_title="消費頻率",
    yaxis_title="顧客數量",
    font=dict(
        size=18,
    )
)

plot(fig, filename='【分群後】消費頻率與顧客數量分佈圖.html')




# 切割 recency
recency_label =  ['0-17 day', '18-27 day', '28-40 day', '41-56 day', '57-68 day', '>69 day']
recency_cut  = [-1, 17, 27, 40, 56, 68, df2['recency'].max()]
df2['recency_cate'] = pd.cut( 
        df2['recency'] , #目標欄位
        recency_cut, #切割條件
        labels =recency_label) #切割後的分類內容

# 切割 frequency
frequency_label =  ['1 freq', '2-4 freq', '5 freq', '6 freq', '7 freq', '>8 freq']
frequency_cut  = [-1, 2, 4, 5, 6, 7, df2['frequency'].max()]
df2['frequency_cate'] = pd.cut( 
        df2['frequency'] , #目標欄位
        frequency_cut,  #切割條件
        labels =frequency_label) #切割後的分類內容



####### RFM 分析 ####### 
# RF交叉分析
RF_table = pd.crosstab(df2['frequency_cate'].astype(str),
                       df2['recency_cate'].astype(str))

# 重新排序
RF_table['freq'] = RF_table.index
# RF_table = RF_table.sort_values('freq',ascending = False)

collist = ['freq'] + recency_label
RF_table = RF_table[collist]


# 以個別消費者來說，這四種顧客分別是個體的誰？
# 分類並標示出新客戶、常貴客、先前客
df2['customer'] = np.where( (df2['frequency'] >=frequency_cut[4]) & (df2['recency']<=recency_cut[3]), '常貴客',
   
                     np.where( (df2['frequency'] >=frequency_cut[4]) & ( df2['recency']>recency_cut[3]), '沉睡客',
                              
                              np.where( (df2['frequency'] < frequency_cut[4]) & ( df2['recency']>recency_cut[3]), '流失客',
                                       
                                       '新顧客'  )))

#--- 繪圖
df2 = df2.rename(columns = {'recency_cate':'近因'})
df2 = df2.rename(columns = {'frequency_cate':'頻率'})

g = sns.FacetGrid(df2, # 來源資料表
                  col="近因", # X資料來源欄位
                  row="頻率" ,  # Y資料來源欄位
                  col_order= recency_label,  # X資料順序
                  row_order= frequency_label, # Y資料順序
                  palette='Set1',  #畫布色調
                  margin_titles=True,
                  hue='customer'
                  )
#小圖表部分
g = g.map_dataframe(sns.barplot, y ='frequency')
g = g.set_axis_labels('近因','頻率').add_legend()
# g.savefig("RFplot.png")




######### 製圖 #######

# 顧客產品推薦圖 - 長條圖
g = sns.FacetGrid(df2, # 來源資料表
                  col="近因", # X資料來源欄位
                  row="頻率" ,  # Y資料來源欄位
                    col_order= recency_label,  # X資料順序
                    row_order= frequency_label, # Y資料順序
                    sharex=False,
            sharey=False,
                    size=2.2, aspect=1.6,
                  palette='Set1',  #畫布色調
                    margin_titles=True,
                    hue='customer'
                  )
#小圖表部分
g = g.map(sns.barplot, 'gender' ,'frequency')
g = g.add_legend()




# 顧客產品推薦圖 - 堆疊圖
del df2['顧客數量']
RFM.RFM_stackedplot(df2, frequency_label,recency_label,'gender')


# 商品個別推薦

product =  orders['product'].unique().tolist()

recom_list = []
for i in range(len(df2[product])):
    aa = df2[product].iloc[i,::].rank().sort_values(ascending = False)
    recom_list.append('、'.join(aa.index))
    
df2['recommend_product'] =recom_list


