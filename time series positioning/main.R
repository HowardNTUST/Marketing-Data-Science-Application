

# 載入library
library(dplyr)
library(reshape2)
library(ggplot2)
library(stringr)
#匯入資料檔案
brand_df_year <- readRDS("brand_df_year.rds")

ggplot(data=brand_df_year[order(brand_df_year$year),], aes(x = article_count  , y = pct, colour = as.factor(brand_name) ,group = brand_name )) +
  geom_point(aes(size = rec, colour = factor(brand_name) )) +
  geom_text(aes(label= sprintf('%s_%s',year, gsub('[a-z]|[A-Z]| |&','', brand_name))), size = 3, hjust=.5, vjust=-.99) +
  geom_vline(xintercept=mean(brand_df_year$article_count), lty=2) +
  geom_hline(yintercept=mean(brand_df_year$pct), lty=2) +
  labs(x = "網路聲量",y = "網路好感度",
       title = "2015-2017的縱貫性「網路聲量」與「網路好感度」定位比較圖")+
  geom_path(arrow = arrow(length=unit(0.50,"cm"),  type = "closed"))


ggplot(data=brand_df_year[order(brand_df_year$year),], aes(x = article_count  , y = rec, colour = as.factor(brand_name) ,group = brand_name )) +
  geom_point(aes(size = rec, colour = factor(brand_name) )) +
  geom_text(aes(label= sprintf('%s_%s',year, gsub('[a-z]|[A-Z]| |&','', brand_name))), size = 3, hjust=.5, vjust=-.99) +
  geom_vline(xintercept=mean(brand_df_year$article_count), lty=2) +
  geom_hline(yintercept=mean(brand_df_year$pct), lty=2) +
  labs(x = "網路聲量",y = "熱門程度",
       title = "2015-2017的縱貫性「網路聲量」與「熱門程度」定位比較圖")+
  geom_path(arrow = arrow(length=unit(0.50,"cm"),  type = "closed"))



ggplot(data=brand_df_year, aes(x = factor(year)  , y = article_count, colour = as.factor(brand_name) ,group = brand_name )) +
  geom_point(aes(size = rec, colour = factor(brand_name) )) +
  geom_text(aes(label= sprintf('%s_%s',year, gsub('[a-z]|[A-Z]| |&','', brand_name))), size = 3, hjust=.5, vjust=-.99) +
  labs(x = "年份",y = "網路聲量",
       title = "2015-2017的縱貫性網路聲量")+
  geom_line()

