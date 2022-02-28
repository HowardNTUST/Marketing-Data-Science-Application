#匯入爬下來的資料
brand_label <- readRDS("brand_label.rds")

######################## 載入library ##############################
suppressPackageStartupMessages({
  library(reshape2)
  library(shiny)
  library(radarchart)
  library(httr)
  library(data.table)
  library(stringr)
  library(rvest)
  require(jiebaR)
  require(data.table)
  library(tidyverse)
  library(caTools)
  library(text2vec)
  library(stringr)
  library(pbapply)
  library(plyr)
})


#視覺化圖表-雷達圖

ser=shinyServer(function(input, output) {
  output$radar <- renderChartJSRadar({
    
    chartJSRadar(brand_label[, c("keyword", input$selectedPeople)], 
                 maxScale = 0.5, showToolTipLabel=TRUE)
  })
})



ui=shinyUI(pageWithSidebar(
  headerPanel('品牌雷達圖'),
  sidebarPanel(
    checkboxGroupInput('selectedPeople', 'select checkbox', 
                       names(brand_label)[-1], selected="B")
  ),
  mainPanel(
    chartJSRadarOutput("radar", width = "450", height = "300"), width = 8
  )
))

shinyApp(ui = ui,server = ser )