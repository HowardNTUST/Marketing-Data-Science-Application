

# 載入library
library(dplyr)
library(reshape2)
library(ggplot2)
library(stringr)

# 讀取資料
orders = read.csv('orders.csv')
cac = read.csv('cac.csv') %>% .[-1]

# 計算clv
gr.margin <- data.frame(product=c('瓶裝水', '牛奶麵包', '高麗菜'), grossmarg=c(10, 50, 10))
# calculating customer lifetime value
orders <- merge(orders, gr.margin, by='product')

clv <- orders %>%
  group_by(clientId) %>%
  summarise(clv=sum(grossmarg)) %>%
  ungroup()

####gist2############


# 假設 2017 - 4 -11 為報告日期
today <- as.Date('2017-04-11', format='%Y-%m-%d')

# processing data
orders <- dcast(orders, orderId + clientId + gender + orderdate ~ product, value.var='product', fun.aggregate=length)
orders$orderdate = as.Date(orders$orderdate)

orders <- orders %>%
  group_by(clientId) %>%
  mutate(frequency=n(),
         recency=as.numeric(today-orderdate)) %>%
  filter(orderdate==max(orderdate)) %>%
  filter(orderId==max(orderId)) %>%
  ungroup()

# 切割頻率
orders.segm <- orders %>%
  mutate(buy_freq=ifelse(between(frequency, 1, 1), '1',
                         ifelse(between(frequency, 2, 2), '2',
                                ifelse(between(frequency, 3, 3), '3',
                                       ifelse(between(frequency, 4, 4), '4',
                                              ifelse(between(frequency, 5, 5), '5', '>5')))))) %>%
  
  
  # 切割近因畫出邊界
  mutate(segm.rec=ifelse(between(recency, 0, 7), '0-7 天',
                         ifelse(between(recency, 8, 15), '8-15 天',
                                ifelse(between(recency, 16, 22), '16-22 天',
                                       ifelse(between(recency, 23, 30), '23-30 天',
                                              ifelse(between(recency, 31, 55), '31-55 天', '>55 天')))))) %>%
  # 把商品放入變數中
  mutate(cart=paste(ifelse(瓶裝水!=0, '、瓶裝水', ''),
                    ifelse(牛奶麵包!=0, '、牛奶麵包', ''),
                    ifelse(高麗菜!=0, '、高麗菜', ''), sep='')) %>%
  arrange(clientId)

# '瓶裝水','牛奶麵包','高麗菜'
# 定義邊界的順序
orders.segm$buy_freq <- factor(orders.segm$buy_freq, levels=c('>5', '5', '4', '3', '2', '1'))
orders.segm$segm.rec <- factor(orders.segm$segm.rec, levels=c('>55 天', '31-55 天', '23-30 天', '16-22 天', '8-15 天', '0-7 天'))
orders.segm$cart = str_split_fixed(orders.segm$cart, '、', 2)[,2]


# 將 CAC與CLV結合進去
orders.segm <- merge(orders.segm, cac, by='clientId')
orders.segm <- merge(orders.segm, clv, by='clientId')

lcg.clv <- orders.segm %>%
  group_by(segm.rec, buy_freq) %>%
  summarise(quantity=n(),
            # calculating cumulative CAC and CLV
            cac=sum(cac),
            clv=sum(clv)) %>%
  ungroup() %>%
  # calculating CAC and CLV per client
  mutate(cac_mean=round(cac/quantity, 2),
         clv_mean=round(clv/quantity, 2))

# clv/cac ratio
lcg.clv$ratio = round(lcg.clv$clv/lcg.clv$cac, 2)

lcg.clv <- melt(lcg.clv, id.vars=c('segm.rec', 'buy_freq', 'quantity'))


howard_theme <- function(base_size = 12, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      axis.text.x = element_text(size=20, angle = 65, vjust = 1, hjust=1),
      axis.text.y = element_text(size=20),
      axis.title = element_text(size = 20),
      panel.grid.major = element_line(color = "grey"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "aliceblue"),
      strip.background = element_rect(fill = "navy", color = "navy", size = 1),
      strip.text = element_text(face = "bold", size = 10, color = "white"),
      legend.position = "right",
      legend.justification = "bottom",
      legend.background = element_blank(),
      legend.text=element_text(size=15),
      panel.border = element_rect(color = "grey", fill = NA, size = 0.05),
      title = element_text(size = 15),
      plot.caption=element_text(size = 10)
    )
}


ggplot(lcg.clv[lcg.clv$variable %in% c('clv', 'cac'), ], aes(x=variable, y=value, fill=variable)) +
  theme_bw() +
  theme(panel.grid = element_blank())+
  geom_bar(stat='identity', alpha=0.6, aes(width=quantity/max(quantity))) +
  geom_text(aes(y=value, label=value), size=4) +
  facet_grid( buy_freq~ segm.rec) +
  ggtitle("顧客終生價值（CLV）與成本（CAC）的加總比較") + howard_theme()+
  xlab("CLV與CAC之比較") + 
  ylab("＄錢＄")


ggplot(lcg.clv[lcg.clv$variable %in% c('clv_mean', 'cac_mean'), ], aes(x=variable, y=value, fill=variable)) +
  theme_bw() +
  theme(panel.grid = element_blank())+
  geom_bar(stat='identity', alpha=0.6, aes(width=quantity/max(quantity))) +
  geom_text(aes(y=value, label=value), size=4) +
  facet_grid(buy_freq ~ segm.rec) +
  ggtitle("顧客終生價值（CLV）與成本（CAC）- 顧客平均價值")+
  howard_theme()+
  xlab("CLV與CAC之比較") + 
  ylab("＄錢＄")

ggplot(lcg.clv[lcg.clv$variable %in% c('ratio'), ], aes(x=variable, y=value, fill=variable)) +
  theme_bw() +
  theme(panel.grid = element_blank())+
  geom_bar(stat='identity', alpha=0.6, aes(width=quantity/max(quantity),fill = value > 1) ) +
  geom_text(aes(y=value, label=value), size=4) +
  facet_grid(buy_freq ~ segm.rec) +
  ggtitle("顧客獲利率比較圖") + howard_theme()+
  guides(fill=FALSE)+
  xlab("比例變數") + 
  ylab("顧客獲利率")

# 平均顧客獲利率
# mean(lcg.clv[lcg.clv$variable %in% c('ratio'), ]$value)
# 
# 
# a = orders.segm[orders.segm$buy_freq == '>5' & orders.segm$segm.rec == '0-7 天',]
# 
# 19*(mean(a$frequency)/mean(a$recency))*2.13 + 12* 0.77* (2.44-1)
# 
# 
# 17*1.73 * (3.13-1)
# (14+5)* 1.82* (3.13-1.5) + (17-5)* 0.77* (2.44-1)
# 
# 19*(mean(a$frequency)/mean(a$recency))*2.13 +12* 0.77* (2.44-1)
# 
# org = 14* 1.82* (3.13-1) + 17 * 0.77* (2.44-1)
# change = 19*(mean(a$frequency)/mean(a$recency))*(3.13-1.18) +12* 0.77* (2.44-1)
# org = 140* 1.82* (3.13-1) + 170 * 0.77* (2.44-1)
# change = 190*(mean(a$frequency)/mean(a$recency))*(3.13-1.18) +120* 0.77* (2.44-1)
# org - change






# 讀取資料
orders = read.csv('orders.csv')
# 假設 2017 - 4 -11 為報告日期
today <- as.Date('2017-04-11', format='%Y-%m-%d')

# calculating customer lifetime value
orders <- merge(orders, gr.margin, by='product')
orders$orderdate = as.Date(orders$orderdate)

customers <- orders %>%
  # combining products and summarising gross margin
  group_by(orderId, clientId, orderdate) %>%
  summarise(grossmarg=sum(grossmarg)) %>%
  ungroup() %>%
  # calculating frequency, recency, average time lapses between purchases and defining cohorts
  group_by(clientId) %>%
  mutate(frequency=n(),
         recency=as.numeric(today-max(orderdate)),
         av.gap=round(as.numeric(max(orderdate)-min(orderdate))/frequency, 0),
         cohort=format(min(orderdate), format='%Y-%m')) %>%
  ungroup() %>%
  # calculating CLV to date
  group_by(clientId, cohort, frequency, recency, av.gap) %>%
  summarise(clv=sum(grossmarg)) %>%
  arrange(clientId) %>%
  ungroup()

# 切割頻率
customer_orders.segm <- customers %>%
  mutate(buy_freq=ifelse(between(frequency, 1, 1), '1',
                         ifelse(between(frequency, 2, 2), '2',
                                ifelse(between(frequency, 3, 3), '3',
                                       ifelse(between(frequency, 4, 4), '4',
                                              ifelse(between(frequency, 5, 5), '5', '>5')))))) %>%
  
  
  # 切割近因畫出邊界
  mutate(segm.rec=ifelse(between(recency, 0, 7), '0-7 天',
                         ifelse(between(recency, 8, 15), '8-15 天',
                                ifelse(between(recency, 16, 22), '16-22 天',
                                       ifelse(between(recency, 23, 30), '23-30 天',
                                              ifelse(between(recency, 31, 55), '31-55 天', '>55 天')))))) 




# 定義邊界的順序
customer_orders.segm$buy_freq <- factor(customer_orders.segm$buy_freq, levels=c('>5', '5', '4', '3', '2', '1'))
customer_orders.segm$segm.rec <- factor(customer_orders.segm$segm.rec, levels=c('>55 天', '31-55 天', '23-30 天', '16-22 天', '8-15 天', '0-7 天'))
customer_orders.segm <- merge(customer_orders.segm, cac, by='clientId')

lcg.coh <- customer_orders.segm %>%
  group_by(cohort, segm.rec, buy_freq) %>%
  # calculating cumulative values
  summarise(quantity=n(),
            cac=sum(cac),
            clv=sum(clv),
            av.gap=sum(av.gap)) %>%
  ungroup() %>%
  # calculating average values
  mutate(av.cac=round(cac/quantity, 2),
         av.clv=round(clv/quantity, 2),
         av.gap=round(av.gap/quantity, 2),
         diff=av.clv-av.cac)

ggplot(lcg.coh, aes(x=cohort, fill=cohort)) +
  theme_bw() +
  theme(panel.grid = element_blank())+
  geom_bar(aes(y=diff), stat='identity', alpha=0.5) +
  geom_text(aes(y=diff, label=round(diff,0)), size=4) +
  facet_grid(buy_freq ~ segm.rec) +
  theme(axis.text.x=element_text(angle=90, hjust=.5, vjust=.5, face="plain")) +
  ggtitle("時間序列的消費金額分析 - CLV與CAC之差異平均價值")+
  xlab("時間序列") + 
  ylab("CLV與CAC之差異平均價值")+
  howard_theme()+
  guides(fill=FALSE)

# 
# a = lcg.coh[lcg.coh$cohort=='2017-02' ,  ]
# sum(a$diff)
# 
# b = lcg.coh[lcg.coh$cohort=='2017-02' ,  ]
# sum(b$diff)
# 