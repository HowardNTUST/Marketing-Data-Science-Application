setwd( "C:/Users/howar/Desktop/marketing analysis")
library(tidyverse)
library(Metrics)
library(ggstatsplot)
library(ggplot2)
source("ggcoefstats2.R")

# read data
marketing_raw <- read.csv("digital_marketing_raw.csv")
# marketing_raw$Youtube = marketing_raw$Youtube *.5
# write.csv(marketing_raw, 'digital_marketing_raw.csv')
# marketing_raw <- read.csv("digital_marketing.csv")

marketing = marketing_raw
names(marketing)
marketing_raw$week<- NULL
marketing$week <- NULL
marketing$week_contiue <- NULL
marketing$X <- NULL
marketing$Date <- NULL

# ---------all - ---------

# 建模
model <- lm(weekly_revenue ~ ., data = marketing)

model_summary <- summary(model)
model_summary
summary(marketing)


# 建立esti表單
model_summary_df <- as.data.frame(coef(model_summary))
model_summary_df <- cbind(model_summary_df, confint(model))
model_summary_df$variables <- row.names(model_summary_df)
names(model_summary_df) <- c("Estimate", "Std_Error", "t_value", "pvalue", "conf_lb", "conf_ub", "variables")

# mape
mape(marketing$weekly_revenue, model$fitted.values)

# 期望投入YouTube的每 1 塊錢成本能貢獻 1.3 元的營收
# 單獨的參數篩選

ad_individual_effect_roi <- function(ad_name = "Youtube", model_summary_df, roi_rev) {
  
  model_summary_df_yt <- model_summary_df[model_summary_df$variables == ad_name, ]

  # 假設檢定
  # roi_rev = 1.3
  newt <- (model_summary_df_yt$Estimate - roi_rev) / model_summary_df_yt$Std_Error
  # pvalue <- 2 * pt( abs(newt), model_summary$df[2], lower.tail = F)
  pvalue <- 2*pt( newt, model_summary$df[2], lower.tail = F)

  model_summary_df_yt$h0 <- roi_rev
  model_summary_df_yt$t_value_h0 <- newt
  model_summary_df_yt$pvalue_h0 <- pvalue
  model_summary_df_yt$df_error <- model_summary$df[2]

  # 繪圖
  df_full <-
    tibble::tribble(
      ~term, ~statistic, ~estimate, ~std.error, ~p.value,
      ~df.error, ~conf.low, ~conf.high,
      "study1",
      model_summary_df_yt$t_value_h0,
      model_summary_df_yt$Estimate,
      model_summary_df_yt$Std_Error,
      model_summary_df_yt$pvalue_h0,
      # model_summary_df_yt$df_error,
      'obs',
      model_summary_df_yt$conf_lb,
      model_summary_df_yt$conf_ub,
    )

  df_full$term <- ad_name

  
  add_x_break <- function(plot, xval) {
    
    p2 <- ggplot_build(plot)
    breaks <- p2$layout$panel_params[[1]]$x$breaks
    breaks <- breaks[!is.na(breaks)]
    
    plot +
      # geom_vline(xintercept = xval) +
      scale_x_continuous(breaks = sort(c(xval, breaks)))
  }
  library(magrittr)
  p <- ggcoefstats2(
    x = df_full,
    xintercept = roi_rev,
    exclude.intercept = TRUE,
    vline = T,
    # meta.analytic.effect = TRUE,
    statistic = 't',
    package = "LaCroixColoR",
    palette = "paired",
    # title = paste( 'Yielded ROI for Revenue by marketing expense'),
    xlab = 'Return on investment for revenue',
    ylab = 'marketing expense'
    # xlab = '投入1元的成本，可以創造多少營收',
    # ylab = '廣告行銷通路'
  )  #+ geom_text(aes(x=roi_rev+0.1, label=roi_rev, y=2), colour="red", angle=180)
  
  p = add_x_break(p, roi_rev)

  return(list(model_summary_df_yt, p))
}


model_summary_df_ig <- ad_individual_effect_roi(ad_name = "Facebook", model_summary_df = model_summary_df, roi_rev = 4)
model_summary_df_yt <- ad_individual_effect_roi(ad_name = "Youtube", model_summary_df = model_summary_df, roi_rev = 5.5)
model_summary_df_m <- ad_individual_effect_roi(ad_name = "Magazine", model_summary_df = model_summary_df, roi_rev = 12)


model_summary_df_ig <- ad_individual_effect_roi(ad_name = "Facebook", model_summary_df = model_summary_df, roi_rev = 0)
model_summary_df_yt <- ad_individual_effect_roi(ad_name = "Youtube", model_summary_df = model_summary_df, roi_rev = 0)
model_summary_df_m <- ad_individual_effect_roi(ad_name = "Magazine", model_summary_df = model_summary_df, roi_rev = 0)

model_summary_df_yt[[1]]
model_summary_df_ig[[1]]
model_summary_df_m[[1]]

model_summary_df_yt[[2]]+
model_summary_df_ig[[2]]+
model_summary_df_m[[2]]


# 總參數篩選

ad_all_effect_roi <- function( model_summary_df, roi_rev) {
  
  model_summary_df_yt <- model_summary_df#[model_summary_df$variables == ad_name, ]
  
  # 假設檢定
  # roi_rev = 1.3
  newt <- (model_summary_df_yt$Estimate - roi_rev) / model_summary_df_yt$Std_Error
  pvalue <- 2 * pt( abs(newt), model_summary$df[2], lower.tail = F)
  
  model_summary_df_yt$h0 <- roi_rev
  model_summary_df_yt$t_value_h0 <- newt
  model_summary_df_yt$pvalue_h0 <- pvalue
  model_summary_df_yt$df_error <- model_summary$df[2]
  
  
  # 繪圖
  df_full <-
    tibble::tribble(
      ~term, ~statistic, ~estimate, ~std.error, ~p.value,
      ~df.error, ~conf.low, ~conf.high,
      "study1",
      model_summary_df_yt$t_value_h0,
      model_summary_df_yt$Estimate,
      model_summary_df_yt$Std_Error,
      model_summary_df_yt$pvalue_h0,
      model_summary_df_yt$df_error,
      model_summary_df_yt$conf_lb,
      model_summary_df_yt$conf_ub,
    )
  
  model_summary_df_yt = model_summary_df_yt[c('variables', 't_value_h0',
                                              'Estimate', 'Std_Error',
                                              'pvalue_h0', 'df_error',
                                              'conf_lb', 'conf_ub'
                                              )]
  
  names(model_summary_df_yt) = names(df_full)
  model_summary_df_yt = model_summary_df_yt[model_summary_df_yt$term!='(Intercept)',]
 
  add_x_break <- function(plot, xval) {
    
    p2 <- ggplot_build(plot)
    breaks <- p2$layout$panel_params[[1]]$x$breaks
    breaks <- breaks[!is.na(breaks)]
    
    plot +
      # geom_vline(xintercept = xval) +
      scale_x_continuous(breaks = sort(c(xval, breaks)))
  }
  
  p <- ggcoefstats2(
    x = model_summary_df_yt,
    xintercept = roi_rev,
    exclude.intercept = TRUE,
    vline = T,
    # meta.analytic.effect = TRUE,
    statistic = "t",
    package = "LaCroixColoR",
    palette = "paired",
    title = paste( '95% 信賴區間'),
    xlab = '投入1元成本，可以造成多少營收',
    ylab = '廣告通路'
  )  #+ geom_text(aes(x=roi_rev+0.1, label=roi_rev, y=2), colour="red", angle=180)
  
  p = add_x_break(p, roi_rev)
    # annotate(geom = "text",
    #             label = c('aaa'),
    #             x = c(1.3),
    #             y = c(3),vjust = 1
    #             )
  #+geom_vline(xintercept = roi_rev, size = 1, linetype = "dashed")
  
  return(list(model_summary_df_yt, p))
}



adall = ad_all_effect_roi( model_summary_df=model_summary_df,
                   roi_rev =1.3)



# ---- 該設定多少錢For有用的通路 ---------
# Load rpart and rpart.plot
library(rpart)
library(rpart.plot)

# Create a decision tree model
tree <- rpart(weekly_revenue ~ ., data = marketing, cp = .009)
# Visualize the decision tree with rpart.plot
rpart.plot(tree, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
mape(marketing$weekly_revenue, predict(tree, marketing))




#--------  Checking the regression model assumptions on residuals--------
# ----normality----
shapiro.test(model$residuals)
# Therefore,NORMALITY IS MET on residuals.Therefore,NORMALITY IS MET on residuals.

# ----Checking for independency on residuals----
library(car)
library(olsrr)
durbinWatsonTest(model)
# Therefore,INDEPENDENCY IS MET on residuals.

# ---- 預測 ---------
pred_df =as.data.frame( predict(model, newdata = marketing ,interval = 'prediction'))
names(pred_df) = c('pred_revenue', 'pred_revenue_lb', 'pred_revenue_ub')
a = cbind(marketing_raw, pred_df)

write.csv(a, 'marketing_pred.csv')


#-------- Checking the accuracy of a model by regression evaluation--------
library(Metrics)
MAE = mae(marketing$weekly_revenue, model$fitted.values)
MSE = mse(marketing$weekly_revenue, model$fitted.values)
RMSE = rmse(marketing$weekly_revenue, model$fitted.values)
MAPE= mape(marketing$weekly_revenue, model$fitted.values)
Adj_R = model_summary$adj.r.squared
Objective_function_table = data.frame(metrics = c('MAE','MSE','RMSE','MAPE','Adj_R'),
                                      Objective_function_table = c(MAE,MSE,RMSE,MAPE,Adj_R))
Objective_function_table$Objective_function_table = round(Objective_function_table$Objective_function_table, 2)

write.csv(Objective_function_table, 'model_Objective_function_table.csv')


ggplot(data = marketing_raw, aes(x = week_contiue, y = weekly_revenue)) + 
  geom_point(color='blue')+
  geom_line(color='blue') +
  geom_line(color='red',linetype = "dashed",data = marketing_raw, aes(x=week_contiue, y=pred_revenue))+
  geom_point(color='red',data = marketing_raw, aes(x=week_contiue, y=pred_revenue))+
  theme_bw()
