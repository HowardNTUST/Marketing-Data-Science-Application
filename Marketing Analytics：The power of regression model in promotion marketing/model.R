library(ggplot2)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library("lubridate")
library(readxl)
library(filesstrings)
library(sjPlot)
library(ggplot2)
library(cAIC4)
library(MuMIn)

howard_theme <- function(base_size = 12, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "aliceblue"),
      strip.background = element_rect(fill = "navy", color = "navy", size = 1),
      legend.position="none"
    )
}

all_para_df = function(game_model){
  sum_model = summary(game_model)
  print(sum_model)
  linear_m_df =data.frame(sum_model$coefficients)
  linear_m_df = cbind(linear_m_df,confint(game_model))
  
  
  # rename and org
  names(linear_m_df)= c("Estimate", "SE", "t-statistic", "pvalue",
                        "CI_lower",
                        "CI_upper")
  
  linear_m_df =round(linear_m_df, 3)
  
  linear_m_df$sig_var = ifelse(linear_m_df$pvalue<0.05, 'sig','')
  return(linear_m_df)  
}
# CI
all_para_lmm_df = function(game_model){
  sum_model = summary(game_model)
  
  print(sum_model)
  cac = cAIC(game_model)
  rr = r.squaredGLMM(game_model)
  rr = data.frame(rr)
  rr$CAIC = cac$caic
  names(rr) = c('Marginal R-squared for GLMM', 'Conditional R-squared for GLMM', 'Conditional Akaike information criterion')
  
  linear_m_df =data.frame(sum_model$coefficients)
  linear_m_df = cbind(linear_m_df, na.omit(confint.merMod(game_model, method="Wald")  ))
  linear_m_df$df = NULL
  linear_m_df =round(linear_m_df, 3)
  library(data.table)
  setDT(linear_m_df, keep.rownames = TRUE)[]
  
  # rename and org
  names(linear_m_df)= c('factor',"Estimate", "SE", "t-statistic", "pvalue",
                        "CI_lower",
                        "CI_upper")
  
  
  
  linear_m_df$sig_var = ifelse(linear_m_df$pvalue<0.05, 'sig','')
  
  
  
  return(list(linear_m_df, rr)  )
}




cared_effect = function(game_model, cared_word = 'bobbleheadYES', linear_m_df,glmm_eval){
  
  bobblehead_ranef <- as.data.frame( ranef(game_model), 
                                     stringsAsFactors = default.stringsAsFactors()) %>% filter(term ==cared_word)
  
  
  bobblehead_ranef <- bobblehead_ranef[-c(1,2,5)]
  
  colnames(bobblehead_ranef) <- c("Team", "Random_effect")
  
  # BH_count <- data %>% group_by(opponent, bobblehead) %>% summarize (n = n())
  # BH_count
  # 
  # Team_significant <- BH_count %>% filter(bobblehead == "YES" & n > 1)
  # bobblehead_ranef <- merge(bobblehead_ranef, Team_significant, by.x = "Team", by.y = "opponent", sort = TRUE)
  # bobblehead_ranef <- bobblehead_ranef[-c(3,4)]
  
  bobblehead_ranef$Final_avg_effect <- bobblehead_ranef$Random_effect + fixef(game_model)[cared_word]
  
  # sketch  bar plot
  bobblehead_ranef %>% ggplot(aes(x = reorder(Team,Final_avg_effect), y = Final_avg_effect)) + 
    geom_col(fill = "cyan4") + coord_flip() + 
    geom_text(aes(label = round(Final_avg_effect, digits = 0)), position = "identity", digits =0) + 
    labs(x = "Teams", y = "Final effect")+
    ggtitle(sprintf("Final average effect of %s by teams with barplot", cared_word))+
    howard_theme()
  
  ggsave(sprintf("Final average effect of %s by teams with bar plot.png", cared_word), width = 8, height =5)
  
  # sketch  random effect
  plot_model(game_model, type = "re", sort.est = cared_word, show.intercept = TRUE,
             free.scale = FALSE, show.values= TRUE,value.size =3, digits =0, value.offset = 0.5, dot.size = 2)+
    ggtitle(sprintf("random effect of %s by teams", cared_word))
  ggsave(sprintf("random effect of %s by teams.png", cared_word), width = 8, height =5)  
  
  # 95% ci
  
  
  # cbind
  
  linear_m_dfbb = linear_m_df[linear_m_df$factor ==cared_word,]
  
  # Final_avg_effect
  bobblehead_ranef$Final_avg_effect <- bobblehead_ranef$Random_effect + fixef(game_model)[cared_word]
  
  # Final_avg_effect lower and upper
  bobblehead_ranef$Final_avg_effect_lower = bobblehead_ranef$Random_effect +linear_m_dfbb$CI_lower
  bobblehead_ranef$Final_avg_effect_upper = bobblehead_ranef$Random_effect +linear_m_dfbb$CI_upper
  
  # upper and lowre label
  bobblehead_ranef$CI_label = sprintf('%s; CI = [%s, %s]', round(bobblehead_ranef$Final_avg_effect), 
                                      round(bobblehead_ranef$Final_avg_effect_lower), 
                                      round(bobblehead_ranef$Final_avg_effect_upper))
  # reorder team
  bobblehead_ranef = bobblehead_ranef[order(bobblehead_ranef$Final_avg_effect, decreasing = T), ]
  bobblehead_ranef$Team <- factor(bobblehead_ranef$Team, levels = rev(levels(bobblehead_ranef$Team)))
  
  # sketch
  p <- ggplot(bobblehead_ranef, aes(Final_avg_effect, Team , colour = as.character(Team)))
  
  p + geom_errorbarh(aes(xmin =  Final_avg_effect_lower , xmax = Final_avg_effect_upper), size = 0.8)+
    geom_point(size = 1.8)+ theme(legend.position = "none")+
    geom_vline(xintercept = 0, linetype="dotted", 
               color = "blue", size=0.5) +
    ggtitle(  sprintf("95¢H confidence interval for Fianl average effect of %s by teams \n Marginal R^2 = %s;  Conditional R^2 = %s; CAIC = %s", cared_word,
                      round(glmm_eval$`Marginal R-squared for GLMM`,2), 
                      round(glmm_eval$`Conditional R-squared for GLMM`,2), 
                      round(glmm_eval$`Conditional Akaike information criterion`,2))) +
    xlab("attendance") + ylab("selected factors")+
    theme(plot.title = element_text(hjust = 0.5))+
    geom_text(aes(label=CI_label),hjust=0.5, vjust=-0.5)+
    howard_theme()
  ggsave(sprintf("Final average effect of %s by teams with CI plot.png", cared_word), width = 8, height =10)+
    howard_theme()
  
  write.csv(bobblehead_ranef, sprintf("random effect of %s by teams.csv", cared_word))
  
  return(bobblehead_ranef)
  
}


diff_promotion_effect_model = function(lmer_formual_word, cared_word,linear_m_df){
  
  
  game_model <- lmer( lmer_formual_word, data = data)
  
  # result
  glmm_result = all_para_lmm_df(game_model)
  glmm_result_coeff = glmm_result[[1]]
  glmm_eval = glmm_result[[2]]
  
  write.csv(glmm_result_coeff, sprintf('%s_glmm_coeff.csv', cared_word))
  write.csv(glmm_eval,sprintf('%s_glmm_eval.csv', cared_word))
  
  
  #---------- sketch random effect with cared fixed effect----------
  cared_word2 = grep(cared_word, glmm_result_coeff$factor,value = T)
  bobblehead_ranef = cared_effect(game_model=game_model, cared_word = cared_word2, linear_m_df=linear_m_df,
                                  glmm_eval=glmm_eval)
  
  
  dir.create(cared_word)
  movefiel = grep(cared_word,list.files(), value = T)
  movefiel = movefiel[ grep('\\.', movefiel) ]
  file.move(movefiel, cared_word)
}
