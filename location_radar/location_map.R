#匯入爬下來的資料
brand_df_2=readRDS("brand_df_2.rds")

######################## library ##############################
suppressPackageStartupMessages({
  library(httr)
  library(data.table)
  library(stringr)
  library(rvest)
  require(jiebaR)
  require(data.table)
  library(tidyverse)
  library(plotly)
})


######################## 繪圖 ##############################

#使用聚集函數將數據從寬數據轉換為長數據

brand_df_2_tmp <- brand_df_2 %>% gather("item", value, -1:-4)

brand_df_2_tmp_2 <- brand_df_2 %>% 
  group_by(brand_name, article_count, rec,positive, negative) %>% 
  summarise(sum=sum(positive, negative)) %>% mutate(pct=positive/sum * 100)

#權重
brand_df_2_tmp_2$trn=scale(brand_df_2_tmp_2$rec)*brand_df_2_tmp_2$pct

#設置繪圖的主題
howard_theme <- function(base_size = 12, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      axis.text = element_text(size = 12),
      axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5),
      axis.title = element_text(size = 14),
      panel.grid.major = element_line(color = "grey"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "aliceblue"),
      strip.background = element_rect(fill = "navy", color = "navy", size = 1),
      strip.text = element_text(face = "bold", size = 12, color = "white"),
      legend.position = "right",
      legend.justification = "bottom", 
      legend.background = element_blank(),
      panel.border = element_rect(color = "grey", fill = NA, size = 0.5),
      title = element_text(size = 20),
      plot.caption=element_text(size = 10)
    )
}
theme_set(howard_theme())
attach(brand_df_2)

# 正面評價較多的品牌
ggplot(brand_df_2, aes(reorder(brand_name, -positive), positive, fill = brand_name)) + 
  geom_bar(position = "dodge", stat="identity", width = 0.8) + 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  theme(legend.position = "none") +
  labs( x = "品牌名稱",
        y = "正面評價數量",
        title = "正面評價較多的品牌",
        caption = "數據來源: PTT汽車板")

# 負面評價較多的品牌
ggplot(brand_df_2, aes(x = reorder(brand_name, -negative), y =  negative, fill = brand_name)) + 
  geom_bar(position = "dodge", stat="identity", width = 0.8) + 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  theme(legend.position = "none") +
  labs( x = "品牌名稱",
        y = "負面評價數量",
        title = "負面評價較多的品牌",
        caption = "數據來源: PTT汽車板")

# 品牌在PTT中的文章數量比較圖
ggplot(brand_df_2, aes(reorder(brand_name, -article_count), article_count, fill = brand_name)) +
  geom_bar(position = "dodge", stat="identity", width = 0.8) + 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  theme(legend.position = "none") +
  labs( x = "品牌名稱",
        y = "品牌在PTT中的文章數量",
        title = "品牌在PTT中的文章數量比較圖",
        caption = "數據來源: PTT汽車板")

# 品牌的正評/負評數量比較圖
ggplot(brand_df_2_tmp, aes(reorder(brand_name, -article_count), value, fill = item)) +
  geom_bar(stat = "identity",position = "dodge", width = 0.8)+ 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  scale_fill_discrete(name="評價狀況",
                      breaks=c("negative", "positive"),
                      labels=c("負評價", "正評價"))+
  labs( x = "品牌名稱",
        y = "品牌的正評/負評數量",
        title = "品牌的正評/負評數量比較圖",
        caption = "數據來源: PTT汽車板")

# 品牌的正評/負評數量比較百分比
ggplot(brand_df_2_tmp, aes(reorder(brand_name, -article_count), value, fill = item)) +
  geom_bar(stat = "identity",position = "fill", width = 0.8)+ 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  scale_fill_discrete(name="評價狀況",
                      breaks=c("negative", "positive"),
                      labels=c("負評價", "正評價"))+
  labs( x = "品牌名稱",
        y = "品牌的正評/負評數量百分比",
        title = "品牌的正評/負評數量比較百分比",
        caption = "數據來源: PTT汽車板")

# 正評價分數比較圖
ggplot(brand_df_2_tmp, aes(reorder(brand_name, -rec),y =  rec,fill = brand_name)) +
  geom_bar(stat = "identity",position = "dodge", width = 0.8)+ 
  theme(axis.text.x =  element_text(angle = 35, color="black"))+
  theme(legend.position = "none") +
  labs( x = "品牌名稱",
        y = "正評價分數",
        title = "正評價分數比較圖",
        caption = "數據來源: PTT汽車板")
######################## 未縮放的定位圖 ##############################

#聲量與好感度分佈圖
ggplot(data=brand_df_2_tmp_2, aes(x = article_count  , y = pct, colour = brand_name)) +
  geom_point(aes(size = rec, colour =brand_name )) +
  geom_text(aes(label=brand_name), size = 3, hjust=.5, vjust=-.99) +
  geom_vline(xintercept=mean(brand_df_2_tmp_2$article_count), lty=2) +
  geom_hline(yintercept=mean(brand_df_2_tmp_2$pct, na.rm = T), lty=2) + 
  theme(legend.title=element_blank())+
  labs(x = "網路聲量",y = "網路好感度",
       title = "聲量與好感度分佈圖",
       caption = "數據來源: PTT汽車板")


