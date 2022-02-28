#匯入爬下來的資料
brand_df_2 =readRDS("brand_df_2.rds")


#'# 3.將data轉換成圖片可用的形式
######################## 3.將data轉換成圖片可用的形式 ##############################

# 將寬資料brand_df_2轉換成 依照 情緒資料的長資料形式
brand_df_2_tmp <- brand_df_2 %>% gather("item", value, -1:-4)

# 將brand_df_2統計出每一個產品的正負字詞加總數，然後產出正面比例pct
brand_df_2_tmp_2 <- brand_df_2 %>% 
  group_by(brand_name, article_count, rec,positive, negative) %>% 
  dplyr::summarise(sum=sum(positive, negative)) %>% mutate(pct= (positive/sum) * 100)

# 將人氣rec * pct加權，人氣越高，好感度越高
# 好感度事好感度
brand_df_2_tmp_2$goodfeeling= scale(brand_df_2_tmp_2$rec*brand_df_2_tmp_2$pct)

library(scales)

scales::rescale(brand_df_2_tmp_2$rec*brand_df_2_tmp_2$pct,to = c(0, 1))

#'# 4.市場定位
######################## 4.市場定位 ##############################

# 設定ggplot的繪圖主題
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


######################## 4.2 好感度與聲量品牌 - 定位圖 ##############################

#plot of unscaled 聲量/好感度 volume of brands
ggplot(data=brand_df_2_tmp_2, aes(x = article_count  , y = pct, colour = brand_name)) +
  geom_point(aes(size = rec, colour =brand_name )) +
  geom_text(aes(label=brand_name), size = 3, hjust=.5, vjust=-.99) +
  geom_vline(xintercept=mean(brand_df_2_tmp_2$article_count), lty=2) +
  geom_hline(yintercept=mean(brand_df_2_tmp_2$pct, na.rm = T), lty=2) + 
  labs(x = "網路聲量",y = "網路好感度",
       title = "未標準化-產品聲量與好感度定位圖",
       caption = "made by TMR")


mud=brand_df_2_tmp_2[brand_df_2_tmp_2$brand_name=='A',]
brand_df_mud_scale=brand_df_2_tmp_2
brand_df_mud_scale$pct=as.numeric(scale(brand_df_2_tmp_2$pct, center = as.numeric(mud['pct'])))
brand_df_mud_scale$article_count=as.numeric(scale(brand_df_2_tmp_2$article_count,center = as.numeric(mud['article_count'])))
brand_df_mud_scale$rec=as.numeric(scale(brand_df_2_tmp_2$rec, center = as.numeric(mud['rec'])))
brand_df_mud_scale$goodfeeling=as.numeric(scale(brand_df_2_tmp_2$goodfeeling, center = as.numeric(mud['goodfeeling'])))


#plot of scaled 聲量/好感度 volume of brands centered 
ggplot(data=brand_df_mud_scale, aes(x = article_count  , y = pct, colour = brand_name)) +
  geom_point(aes(size = rec, colour =brand_name )) +
  geom_text(aes(label=brand_name), size = 3, hjust=.5, vjust=-.99) +
  #scale_colour_gradient(low="gold", high="red") +
  geom_vline(xintercept=brand_df_mud_scale[brand_df_mud_scale$brand_name=='A',]$article_count, lty=2) +
  geom_hline(yintercept=brand_df_mud_scale[brand_df_mud_scale$brand_name=='A',]$pct, lty=2) +  
  labs(x = "網路聲量",y = "網路好感度",
       title = "標準化-產品聲量與好感度定位圖",
       subtitle = "以產品為中心",
       caption = "made by TMR")


# 產品聲量與好感度定位圖
ggplot(data=brand_df_2_tmp_2, aes(x = article_count  , y = goodfeeling, colour = brand_name)) +
  geom_point(aes(size = rec, colour =brand_name )) +
  geom_text(aes(label=brand_name), size = 3, hjust=.5, vjust=-.99) +
  #scale_colour_gradient(low="gold", high="red") +
  geom_vline(xintercept=brand_df_2_tmp_2[brand_df_2_tmp_2$brand_name=='A',]$article_count, lty=2) +
  geom_hline(yintercept=brand_df_2_tmp_2[brand_df_2_tmp_2$brand_name=='A',]$goodfeeling, lty=2) +  
  labs(x = "網路聲量",y = "網路好感度",
       title = "產品聲量與好感度定位圖",
       subtitle = "以產品為中心",
       caption = "made by TMR")

# 好感度與plusvoice比較圖
ggplot(data=brand_df_2_tmp_2, aes(x = rec  , y = goodfeeling, colour = brand_name)) +
  geom_point(aes(size = rec, colour =brand_name )) +
  geom_text(aes(label=brand_name), size = 3, hjust=.5, vjust=-.99) +
  #scale_colour_gradient(low="gold", high="red") +
  geom_vline(xintercept=brand_df_2_tmp_2[brand_df_2_tmp_2$brand_name=='A',]$rec, lty=2) +
  geom_hline(yintercept=brand_df_2_tmp_2[brand_df_2_tmp_2$brand_name=='A',]$goodfeeling, lty=2) +  
  labs(x = "網路好感度",y = "正面聲亮",
       title = "好感度與正面聲亮比較圖",
       subtitle = "以產品為中心",
       caption = "made by TMR")