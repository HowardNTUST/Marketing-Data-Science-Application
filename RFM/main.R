
# 載入library
library(dplyr)
library(reshape2)
library(ggplot2)
library(stringr)
# start from here
orders = read.csv('orders.csv')
orders$orderdate = as.Date(orders$orderdate, origin="2017-01-01")
# 報告日期
today <- as.Date('2017-04-11', format='%Y-%m-%d')

# 資料處理
orders <- dcast(orders, orderId + clientId + gender + orderdate ~ product, value.var='product', fun.aggregate=length)

orders <- orders %>%
  group_by(clientId) %>%
  mutate(frequency=n(),
         recency=as.numeric(today-orderdate)) %>%
  filter(orderdate==max(orderdate)) %>%
  filter(orderId==max(orderId)) %>%
  ungroup()



#繪製出RF分佈圖

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

#消費頻率與訂單數量分佈圖
ggplot(orders, aes(x=frequency)) +
  #theme_bw() +
  scale_x_continuous(breaks=c(1:10)) +
  geom_histogram(alpha=0.6, binwidth=1) +
  ggtitle("消費頻率與訂單數量分佈圖")+
  xlab("消費頻率") + 
  ylab("訂單數量") +howard_theme()+
  theme(plot.title = element_text(color="red", size=30),
        axis.title.x = element_text(color="blue", size=20),
        axis.title.y = element_text(color="#993333", size=20))

#R頻率分佈圖
ggplot(orders, aes(x=recency)) +
  theme_bw() +
  geom_histogram(alpha=0.6, binwidth=1) +
  scale_x_continuous(breaks=c(0:91))+
  ggtitle("最近一次（天）的消費與購買量分佈圖")+
  xlab("距離上次購買的天數") + 
  ylab("訂單數量") +#howard_theme()+
  theme(plot.title = element_text(color="red", size=30),
        axis.title.x = element_text(color="blue", size=20),
        axis.title.y = element_text(color="#993333", size=20),
        panel.background = element_rect(fill = "aliceblue"),
        strip.background = element_rect(fill = "navy", color = "navy", size = 1),
        strip.text = element_text(face = "bold", size = 10, color = "white"))


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

lcg <- orders.segm %>%
  group_by(segm.rec, buy_freq) %>%
  summarise(quantity=n()) %>%
  mutate(client='顧客人數') %>%
  ungroup()

lcg.matrix= as.data.frame.matrix(table(orders.segm$buy_freq, orders.segm$segm.rec))
lcg.matrix$buy_freq = row.names(lcg.matrix) 
lcg.matrix

# 繪製RFM分析圖
lcg.adv <- lcg %>%
  mutate(rec.type = ifelse(segm.rec %in% c(">55 天", "31-55 天", "23-30 天"), "not recent", "recent"),
         freq.type = ifelse(buy_freq %in% c(">5", "5", "4"), "frequent", "infrequent"),
         customer.type = interaction(rec.type, freq.type))

ggplot(lcg.adv, aes(x=client, y=quantity, fill=customer.type)) +
  theme_bw() +
  theme(panel.grid = element_blank()) +
  geom_rect(aes(fill = customer.type), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, alpha = 0.1) +
  facet_grid(buy_freq ~ segm.rec) +
  geom_bar(stat='identity', alpha=0.7) +
  geom_text(aes(y=max(quantity)/2, label=quantity), size=4) +
  ggtitle("R與F分析圖") +
  xlab("最近一次消費天數") + ylab("購買頻率")+ 
  theme(plot.title = element_text(color="red", size=30 ),
        axis.title.x = element_text(color="blue", size=20, face="bold"),
        axis.title.y = element_text(color="#993333", size=20, face="bold"))+
  guides(fill=guide_legend(title="客群顏色指示表"))+
  scale_fill_discrete(name="Experimental\nCondition",breaks = c('not recent.frequent','recent.frequent','not recent.infrequent','recent.infrequent'), labels = c('先前客','常貴客','一次性消費客人','新顧客'))

lcg.sub <- orders.segm %>%
  group_by(gender, cart, segm.rec, buy_freq) %>%
  summarise(quantity=n()) %>%
  mutate(client='顧客人數') %>%
  ungroup()



# 繪製RFM分析圖(性別分類)  fill=
lcg.sub$gender = factor(lcg.sub$gender, levels = c('女性', '男性'))
ggplot(lcg.sub, aes(x=client, y=quantity, fill=gender)) +
  theme_bw() +
  scale_fill_brewer(palette='Set1') +
  theme(panel.grid = element_blank())+
  geom_bar(stat='identity', position='fill' , alpha=0.6) +
  facet_grid(buy_freq ~ segm.rec) +
  ggtitle("R與F分析圖（性別）") +
  xlab("最近一次消費天數") + ylab("購買頻率")+ 
  theme(plot.title = element_text(color="red", size=30),
        axis.title.x = element_text(color="blue", size=20, face="bold"),
        axis.title.y = element_text(color="#993333", size=20, face="bold"))+
  guides(fill=guide_legend(title="顧客性別"))



# 繪製RFM分析圖(商品分類)
ggplot(lcg.sub, aes(x=gender, y=quantity, fill=cart)) +
  theme_bw() +
  scale_fill_brewer(palette='Set1') +
  theme(panel.grid = element_blank())+
  geom_bar(stat='identity', position='fill' , alpha=0.6) +
  facet_grid(buy_freq ~ segm.rec) +
  ggtitle("R與F分析圖(商品分類)") +
  xlab("最近一次消費天數") + ylab("購買頻率")+ 
  theme(plot.title = element_text(color="red", size=30),
        axis.title.x = element_text(color="blue", size=20, face="bold"),
        axis.title.y = element_text(color="#993333", size=20, face="bold"))+
  guides(fill=guide_legend(title="商品顏色指示表"))
