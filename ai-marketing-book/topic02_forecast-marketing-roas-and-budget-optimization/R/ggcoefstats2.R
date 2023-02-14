# Generated from function body. Editing this file has no effect.
ggcoefstats2= function (x, output = "plot", statistic = NULL, conf.int = TRUE, 
          conf.level = 0.95, k = 2L, exclude.intercept = FALSE, effsize = "eta", 
          meta.analytic.effect = FALSE, meta.type = "parametric", bf.message = TRUE, 
          sort = "none", xlab = NULL, ylab = NULL, title = NULL, subtitle = NULL, 
          caption = NULL, only.significant = FALSE, point.args = list(size = 3, 
                                                                      color = "blue"), errorbar.args = list(height = 0), vline = TRUE, 
          vline.args = list(size = 1, linetype = "dashed"), stats.labels = TRUE, 
          xintercept = 0,
          stats.label.color = NULL, stats.label.args = list(size = 3, 
                                                            direction = "y", min.segment.length = 0), package = "RColorBrewer", 
          palette = "Dark2", ggtheme = ggstatsplot::theme_ggstatsplot(), 
          ...) 
{
  if (!insight::is_model(x)) {
    tidy_df <- as_tibble(x)
    if (is.null(statistic)) 
      stats.labels <- FALSE
  }
  if (insight::is_model(x)) {
    statistic <- insight::find_statistic(x)
    eta_squared <- omega_squared <- NULL
    if (effsize == "eta") 
      eta_squared <- "partial"
    if (effsize == "omega") 
      omega_squared <- "partial"
    tidy_df <- parameters::model_parameters(model = x, eta_squared = eta_squared, 
                                            omega_squared = omega_squared, ci = conf.level, verbose = FALSE, 
                                            table_wide = TRUE, ...) %>% parameters::standardize_names(style = "broom") %>% 
      rename_all(~gsub("omega2.|eta2.", "", .x))
    if (all(c("df", "df.error") %in% names(tidy_df))) 
      tidy_df %<>% mutate(effectsize = paste0("partial ", 
                                              effsize, "-squared"))
  }
  if (is.null(tidy_df) || !"estimate" %in% names(tidy_df)) {
    rlang::abort("The tidy data frame *must* contain 'estimate' column.")
  }
  if (!"term" %in% names(tidy_df)) {
    tidy_df %<>% mutate(term = paste("term", row_number(), 
                                     sep = "_"))
  }
  if (any(duplicated(select(tidy_df, term)))) {
    tidy_df %<>% tidyr::unite(col = "term", matches("term|variable|parameter|method|curve|response|component|contrast|group"), 
                              remove = TRUE, sep = "_")
  }
  if (any(duplicated(tidy_df$term))) 
    rlang::abort("Elements in `term` column must be unique.")
  if (!(all(c("p.value", "statistic") %in% names(tidy_df)))) 
    stats.labels <- FALSE
  if (!"conf.low" %in% names(tidy_df)) {
    tidy_df %<>% mutate(conf.low = NA, conf.high = NA)
    conf.int <- FALSE
  }
  if (exclude.intercept) 
    tidy_df %<>% filter(!grepl("(Intercept)", term, TRUE))
  if (stats.labels) {
    tidy_df %<>% statsExpressions::tidy_model_expressions(statistic, 
                                                          k, effsize)
    if (only.significant && ("p.value" %in% names(tidy_df))) {
      tidy_df %<>% mutate(expression = ifelse(p.value >= 
                                                0.05, list(NULL), expression))
    }
  }
  tidy_df %<>% parameters::sort_parameters(sort = sort, column = "estimate")
  tidy_df %<>% dplyr::mutate(term = factor(term, tidy_df$term))
  glance_df <- performance::model_performance(x, verbose = FALSE) %>% 
    as_tibble()
  if (meta.analytic.effect) {
    meta.type <- stats_type_switch(meta.type)
    subtitle_df <- meta_analysis(tidy_df, type = meta.type, 
                                 k = k)
    subtitle <- subtitle_df$expression[[1]]
    if (meta.type == "parametric" && bf.message) {
      caption_df <- meta_analysis(tidy_df, type = "bayes", 
                                  k = k)
      caption <- caption_df$expression[[1]]
    }
  }
  if (output == "plot") {
    plot <- ggplot(tidy_df, mapping = aes(estimate, term)) + 
      exec(geom_point, !!!point.args)
    if (conf.int) {
      plot <- plot + exec(geom_errorbarh, data = tidy_df, 
                          mapping = aes(xmin = conf.low, xmax = conf.high), 
                          !!!errorbar.args)
    }
    if (vline) 
      plot <- plot + exec(geom_vline, xintercept =xintercept , !!!vline.args)
    if (stats.labels) {
      plot <- plot + exec(ggrepel::geom_label_repel, data = tidy_df, 
                          mapping = aes(x = estimate, y = term, label = expression), 
                          parse = TRUE, color = stats.label.color %||% 
                            "black", na.rm = TRUE, !!!stats.label.args)
    }
    plot <- plot + labs(x = xlab %||% "estimate", y = ylab %||% 
                          "term", caption = caption, subtitle = subtitle, title = title) + 
      ggtheme + theme(plot.caption = element_text(size = 10))
  }
  switch(output, subtitle = subtitle, caption = caption, tidy = tidy_df, 
         glance = glance_df, plot)
}
