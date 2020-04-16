source('model.R')
library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library("lubridate")
library(readxl)
library(filesstrings)
library(sjPlot)



# setwd('C:\\Users\\Howie\\Desktop\\fandora新')
data <- read.csv("dodgers.csv")

# Bobblehead on attendance
#Bobblehead data
BH_count <- data %>% group_by(opponent, bobblehead) %>% summarize (n = n())
BH_count

BH_count %>% ggplot(aes(bobblehead,n)) + 
  geom_col(fill = "cyan4") + 
  facet_wrap(~opponent) + 
  geom_text(aes(label = n), position = "dodge", color = "black")
ggsave("Number of bobblehead per team.png", width = 5, height = 5)


#Scatterplot of attendance by home teams with or without bobblehead promotion
data %>% ggplot(aes(bobblehead, attend)) + geom_boxplot() + 
  facet_wrap(~opponent)
ggsave('bobblehead_attendance_by_team.png', width = 6, height =6)


# time effect
data %>% ggplot(aes(day_of_week, bobblehead)) + geom_count()
ggsave('Bobblehead by day of week.png', width = 6, height = 3)


data %>% ggplot(aes(day_of_week, attend)) + geom_boxplot()
ggsave('Attendance by day of week.png', width = 6, height = 4)



# Other promotions on attendance

# Attendance by shirt promotion
data %>% ggplot(aes(shirt, attend)) + geom_boxplot()
ggsave('Attendance by shirt.png', width = 6, height = 4)

data %>% group_by(shirt) %>% summarize(mean(attend))

#Attendance by cap promotion
data %>% ggplot(aes(cap, attend)) + geom_boxplot()
ggsave('Attendance by cap.png', width = 6, height = 4)
data %>% group_by(cap) %>% summarize(mean(attend))


#Attendance by fireworks
data %>% ggplot(aes(fireworks, attend)) + geom_boxplot()
ggsave('Attendance by fireworks.png', width = 6, height = 4)

data %>% group_by(fireworks) %>% summarize(mean(attend))


# Weather on attendance
data %>% ggplot(aes(skies, attend)) + geom_boxplot()
ggsave('Attendance by skies.png', width = 6, height = 4)

data %>% ggplot(aes(temp, attend)) + geom_point()
ggsave('Temp vs. Attendance.png', width = 7, height = 4)


# Month on attendance
data %>% ggplot(aes(month, attend)) + geom_boxplot()
ggsave('Attendance by month.png', width = 6, height = 4)

# day_night on attendance
data %>% ggplot(aes(day_night, attend)) + geom_boxplot()
ggsave('Attendance by day_night.png', width = 6, height = 4)



#--------------linear model-----------------


# linear model
game_model <- lm(attend ~  month + opponent+
                  day_of_week + temp + skies  + 
                  cap + shirt + fireworks + bobblehead, data)

# output
linear_m_df = all_para_df(game_model)
linear_m_df
write.csv(linear_m_df, 'first_model.csv')

# model selection

library(olsrr)

# stepwise regression
step_game_model = ols_step_both_aic(game_model)
step_game_model
plot(step_game_model)
formula = as.formula(paste('attend', "~", paste(step_game_model$predictors, collapse = " + ")))

# re-model again
game_model <- lm(formula, data)
linear_m_df = all_para_df(game_model)
linear_m_df
write.csv(linear_m_df, 'sec_model.csv')

linear_m_df$factor = row.names(linear_m_df)


# 95％ confidence interval for all selected factors
p <- ggplot(linear_m_df, aes(Estimate, factor, colour = as.character(Estimate)))
p + geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper), size = 0.8)+
  geom_point(size = 1.8)+ theme(legend.position = "none")+
  geom_vline(xintercept = 0, linetype="dotted", 
             color = "blue", size=0.5) +
  ggtitle("95％ confidence interval for all selected factors\n based on AIC stepwise regression; R^2 = .52") +
  xlab("attendance") + ylab("selected factors")+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_text(aes(label=Estimate),hjust=0.5, vjust=-0.5)+
  howard_theme()

ggsave("1. 95％ confidence interval for all selected factors.png", width = 8, height =5)
# percentage interval compared to average attendance
# 平均可以增加多少

linear_m_df$Estimate_percent  = linear_m_df$Estimate /  mean(data$attend)
linear_m_df$CI_lower_percent  = linear_m_df$CI_lower /  mean(data$attend)
linear_m_df$CI_upper_percent  = linear_m_df$CI_upper /  mean(data$attend)
linear_m_df$Estimate_percent_label = round(linear_m_df$Estimate_percent, 2)*100
linear_m_df$Estimate_percent_label =sprintf('%s％', linear_m_df$Estimate_percent_label)

p <- ggplot(linear_m_df, aes(Estimate_percent, factor, colour = as.character(Estimate)))
p + geom_errorbarh(aes(xmin = CI_lower_percent, xmax = CI_upper_percent), size = 0.8)+
  geom_point(size = 1.8)+ theme(legend.position = "none")+
  geom_vline(xintercept = 0, linetype="dotted", 
             color = "blue", size=0.5) +
  ggtitle("95％ confidence interval in percentage compared to average attendance for all selected factors\n based on AIC stepwise regression; R^2 = .52") +
  xlab("attendance") + ylab("selected factors")+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_text(aes(label=Estimate_percent_label),hjust=0.5, vjust=-0.5) +
  howard_theme()

ggsave("2. 95％ confidence interval in percentage compared to average attendance for all selected factors.png", width = 8, height =5)

# 95％ confidence interval for sig selected factors
linear_m_df_sig = linear_m_df[linear_m_df$sig_var=='sig',]
linear_m_df_sig = linear_m_df_sig[order(linear_m_df_sig$Estimate, decreasing = T), ]
linear_m_df_sig$CI = sprintf('%s; CI = [%s, %s]', round(linear_m_df_sig$Estimate), 
                             round(linear_m_df_sig$CI_lower), 
                             round(linear_m_df_sig$CI_upper))

p <- ggplot(linear_m_df_sig, aes(Estimate, factor, colour = as.character(Estimate)))
p + geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper), size = 0.8)+
  geom_point(size = 1.8)+ theme(legend.position = "none")+
  geom_vline(xintercept = 0, linetype="dotted", 
             color = "blue", size=0.5) +
  ggtitle("95％ confidence interval for 【significantly】 selected factors\n based on AIC stepwise regression; R^2 = .52") +
  xlab("attendance") + ylab("selected factors")+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_text(aes(label=CI),hjust=0.5, vjust=-0.5)
ggsave("3. 95％ confidence interval for【significantly】selected factors.png", width = 8, height =5)+
  howard_theme()


# percentage interval compared to average attendance
# 平均可以增加多少
linear_m_df_sig$Estimate_percent  = linear_m_df_sig$Estimate /  mean(data$attend)
linear_m_df_sig$CI_lower_percent  = linear_m_df_sig$CI_lower /  mean(data$attend)
linear_m_df_sig$CI_upper_percent  = linear_m_df_sig$CI_upper /  mean(data$attend)

linear_m_df_sig$Estimate_percent_label = round(linear_m_df_sig$Estimate_percent, 2)*100
linear_m_df_sig$Estimate_percent_label =sprintf('%s％', linear_m_df_sig$Estimate_percent_label)

linear_m_df_sig$CI_lower_percent_label = round(linear_m_df_sig$CI_lower_percent, 2)*100
linear_m_df_sig$CI_lower_percent_label =sprintf('%s％', linear_m_df_sig$CI_lower_percent_label)

linear_m_df_sig$CI_upper_percent_label = round(linear_m_df_sig$CI_upper_percent, 2)*100
linear_m_df_sig$CI_upper_percent_label =sprintf('%s％', linear_m_df_sig$CI_upper_percent_label)


linear_m_df_sig$CI_percentag = sprintf('%s; CI = [%s, %s]', linear_m_df_sig$Estimate_percent_label, 
                             linear_m_df_sig$CI_lower_percent_label, 
                             linear_m_df_sig$CI_upper_percent_label)

p <- ggplot(linear_m_df_sig, aes(Estimate_percent, factor, colour = as.character(Estimate)))
p + geom_errorbarh(aes(xmin = CI_lower_percent, xmax = CI_upper_percent), size = 0.8)+
  geom_point(size = 1.8)+ theme(legend.position = "none")+
  geom_vline(xintercept = 0, linetype="dotted", 
             color = "blue", size=0.5) +
  ggtitle("95％ confidence interval for 【significantly】selected factors\n based on AIC stepwise regression; R^2 = .52") +
  xlab("attendance") + ylab("selected factors")+
  theme(plot.title = element_text(hjust = 0.5))+
  geom_text(aes(label=CI_percentag),hjust=0.5, vjust=-0.5)+
  howard_theme()

ggsave("4. 95％ confidence interval for 【significantly】 selected factors.png", width = 8, height =5)

#--------------linear mixed effect model using opponent as random effect-----------------

# modeling
for (cared_word in c("fireworks", "bobblehead")) {
  
  # cared_word = 'bobblehead'
  lmer_formual_word =  as.formula(sprintf('attend ~  month + 
                         day_of_week + temp + skies  + 
                         cap + shirt + fireworks + bobblehead+ (1 + %s | opponent)',cared_word) )
  
  diff_promotion_effect_model(lmer_formual_word=lmer_formual_word, cared_word = cared_word,linear_m_df=linear_m_df)
  
}
