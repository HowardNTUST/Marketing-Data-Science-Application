# %%
# SECTION - 主題2：AI幫您擬定最適行銷預算分配戰略

# %%
# SECTION - 程式碼1
import statsmodels.api as sm
from topic02_lib import *
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

# 讀入檔案
X, y, data = load_data(data_path='digital_marketing_raw.csv',
                       drop_columns=['weekly_revenue', 'Date'],
                       y_column='weekly_revenue')

# 輸出data.info() --> 將data.dtypes 與 data.count() 合併
# 產出X的資料型態與非空值的資料筆數
data_type_X = pd.concat(
    [pd.DataFrame(X.dtypes), X.count()], axis=1).reset_index()

# 產出y的資料型態與非空值的資料筆數
y1 = pd.DataFrame(y)
data_type_y = pd.concat(
    [pd.DataFrame(y1.dtypes), y1.count()], axis=1).reset_index()

# 將X與y的資料型態與非空值的資料筆數合併
data_type = pd.concat(
    [data_type_X, data_type_y], axis=0)

# 產出欄位名稱與資料型態Excel檔案
data_type.columns = ['欄位名稱', '資料型態', '非空值的資料筆數']
data_type.to_excel('01_資料形態.xlsx', index=False)

# !SECTION 程式碼1

# %%
# SECTION - 程式碼2
X_const = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X_const)
results = model.fit()
print(results.summary())
roas_summary_df = (results.summary2().tables[1])  # .reset_index()
roas_summary_df = round(roas_summary_df, 3)
roas_summary_df = roas_summary_df.reset_index()
roas_summary_df.columns = ["marketing_channel", "coefficient",
                           "std_error", "t_value", "pvalue", "conf_lb", "conf_ub"]
roas_summary_df.to_excel('02_多元迴歸模型_估計的ROAS係數.xlsx', index=False)

# !SECTION 程式碼2

# %%
# SECTION - 程式碼3（Todo）
fb = results.t_test('Facebook=6')
results.t_test('Facebook=0')
fb.summary_frame()
results.summary().tables[1]
LRresult = (results.summary2().tables[1])

# !SECTION 程式碼3


# %%
# SECTION - 程式碼4
marketing_expense = 150000
roas = 1.8
revenue = marketing_expense * roas
cost = revenue * 0.48

roas_assumed_table = pd.DataFrame({
    '項目': ['營收', '產品成本', '行銷費用', '利潤'],
    '總計金額': [revenue, cost, marketing_expense, revenue - cost - marketing_expense],
    '備註': ['營收 = 行銷費用*1.8倍的ROAS', '產品成本 = 營收*48%', '行銷費用為 = 150000', '利潤 = 營收-產品成本-行銷費用']
})

roas_assumed_table.to_excel('03_ROAS假設情境表.xlsx', index=False)

# !SECTION 程式碼4

# %%
# SECTION - 程式碼5（Todo）
fb = results.t_test('Facebook=6')
results.t_test('Facebook=0')
fb.summary_frame()
results.summary().tables[1]
LRresult = (results.summary2().tables[1])

# !SECTION 程式碼5

# %%
# SECTION - 程式碼6

# 訓練迴歸模型
reg_model = LinearRegression()
reg_model.fit(X, y)

# 建立預測模型
results_all, fig_all = prediction_interval(data=data,
                                           X=X, y=y,
                                           X_new=X,
                                           tuned_model=reg_model,
                                           weeks=None,
                                           output_name = '04_1_真實的週營收 vs 預測的週營收'
                                           )

# !SECTION - 程式碼6

# %%
# SECTION - 程式碼7

# 模型表現評估
perf_table = performance_metrics(X, y, results_all)

# !SECTION - 程式碼7


# %%
# SECTION - 程式碼8

# 將carryover與saturation融合到迴歸模型，並訓練之
tuned_model, best_param = train_marketing_carryover_saturation(X, y, n_trials=1000)

# 秀出最好的參數組合
print(f"best_param\n: {best_param}")


# !SECTION - 程式碼8


# %%
# SECTION - 程式碼9

# ----- load tuned marketing mix model ----- #
tuned_model, best_param = load_tuned_model(tune_model_path="tuned_model.dat")

print(f"best_param\n: {best_param}")
print(f"best_param\n: {tuned_model}")

# !SECTION - 程式碼9

# %%
# SECTION - 程式碼10

# Facebook的遞延效應與飽和效應
fb_prop_budget, Carryover_plot, saturation_fig, budget_margin_plot = effect_plot(
    X=X, variable_name=(variable_name := 'Facebook'),
    tuned_model=tuned_model, best_param=best_param, only_prop_budget=False)

# !SECTION - 程式碼10

# %%
# SECTION - 程式碼11
# Youtube的遞延效應與飽和效應
yt_prop_budget, Carryover_plot, saturation_fig, budget_margin_plot = effect_plot(
    X=X, variable_name=(variable_name := 'Youtube'),
    tuned_model=tuned_model, best_param=best_param, only_prop_budget=False)

# !SECTION - 程式碼11

# %%
# SECTION - 程式碼12
# Magazine的遞延效應與飽和效應
mg_prop_budget, Carryover_plot, saturation_fig, budget_margin_plot = effect_plot(
    X=X, variable_name=(variable_name := 'Magazine'),
    tuned_model=tuned_model, best_param=best_param, only_prop_budget=False)

# !SECTION - 程式碼12

# %%
# SECTION - 程式碼13

# 新行銷模型的預測表現
results_all, fig_all = prediction_interval(data=data,
                                           X=X, y=y,
                                           X_new=X,
                                           tuned_model=tuned_model,
                                           weeks=None,
                                           output_name = '05_1_營收分析_行銷迴歸模型預測效果與真實狀況',
                                           )

# performance_metrics in practice
perf_table = performance_metrics(X, y, results_all,
                                 output_name = '05_2_營收分析_行銷迴歸模型預測效果與真實狀況表現評估表')

# !SECTION - 程式碼13
