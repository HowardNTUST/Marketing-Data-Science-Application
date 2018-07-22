
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


df_train = pd.read_csv(
    'df_train_change.csv', usecols=[ 1,2, 3, 4, 5],
    dtype={'促銷與否': bool},
    converters={'銷售量_log': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=['日期']
)

df_train.head()
a = df_train.iloc[0:100,::]


df_2017 = df_train.loc[df_train.日期>=pd.datetime(2017,1,1)]
del df_train

df_test = pd.read_csv(
    "df_test_change.csv", usecols=[  2, 3, 4,5,6],
    dtype={'促銷與否': bool},
    parse_dates=["日期"]  # , 日期_parser=parser
).set_index(
    ['分店編號', '商品編號', '日期']
)


items = pd.read_csv(
    "items_change.csv",
).set_index("商品編號")

stores = pd.read_csv(
    "stores_change.csv",
).set_index("分店編號")

promo_2017_train = df_2017.set_index(
    ["分店編號", "商品編號", "日期"])[["促銷與否"]].unstack(
        level=-1).fillna(False)


promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["促銷與否"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train


df_2017 = df_2017.set_index(
    ["分店編號", "商品編號", "日期"])[['銷售量_log']].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))
stores = stores.reindex(df_2017.index.get_level_values(0))


df_2017_item = df_2017.groupby('商品編號')[df_2017.columns].sum()
promo_2017_item = promo_2017.groupby('商品編號')[promo_2017.columns].sum()

df_2017_store_class = df_2017.reset_index()
df_2017_store_class['小分類'] = items['小分類'].values
df_2017_store_class_index = df_2017_store_class[['小分類', '分店編號']]
df_2017_store_class = df_2017_store_class.groupby(['小分類', '分店編號'])[df_2017.columns].sum()

df_2017_promo_store_class = promo_2017.reset_index()
df_2017_promo_store_class['小分類'] = items['小分類'].values
df_2017_promo_store_class_index = df_2017_promo_store_class[['小分類', '分店編號']]
df_2017_promo_store_class = df_2017_promo_store_class.groupby(['小分類', '分店編號'])[promo_2017.columns].sum()


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def add_new_feature(df, 銷售_df, t2017, is_train=True, name_prefix=None):
    
    X = {
        "銷售_14_2017": get_timespan(銷售_df, t2017, 14, 14).sum(axis=1).values,
        "銷售_60_2017": get_timespan(銷售_df, t2017, 60, 60).sum(axis=1).values,
        "銷售_140_2017": get_timespan(銷售_df, t2017, 140, 140).sum(axis=1).values,
        "銷售_3_2017_aft": get_timespan(銷售_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,
        "銷售_7_2017_aft": get_timespan(銷售_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
        "銷售_14_2017_aft": get_timespan(銷售_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
    }

    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        tmp = get_timespan(銷售_df, t2017, i, i)
        X['has_銷售_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_銷售_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_銷售_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    tmp = get_timespan(銷售_df, t2017 + timedelta(days=16), 15, 15)
    X['has_銷售_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values
    X['last_has_銷售_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values
    X['first_has_銷售_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

    for i in range(1, 16):
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    for i in range(-16, 16):
        X["銷售_{}".format(i)] = 銷售_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X



def Naive_add_new_feature(df, promo_df, t2017, is_train=True, name_prefix=None):
    
    X = {
        "promo_14_2017": get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_df, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_df, t2017, 140, 140).sum(axis=1).values,
        "promo_3_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,
        "promo_7_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
        "promo_14_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
    }

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


print("專家經驗 - 量化變數")
t2017 = date(2017, 5, 31)
num_days = 6
X_l, y_l = [], []
i = 0

for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = add_new_feature(df_2017, promo_2017, t2017 + delta)

    X_tmp2 = add_new_feature(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')
    X_tmp2.index = df_2017_item.index
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

    X_tmp3 = add_new_feature(df_2017_store_class, df_2017_promo_store_class, t2017 + delta, is_train=False, name_prefix='store_class')
    X_tmp3.index = df_2017_store_class.index
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp2
    gc.collect()
    
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

## val
X_val, y_val = add_new_feature(df_2017, promo_2017, date(2017, 7, 26))

X_val2 = add_new_feature(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = df_2017_item.index
X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_val3 = add_new_feature(df_2017_store_class, df_2017_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = df_2017_store_class.index
X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)

X_val.to_csv('X_val.csv')

y_val = np.save('y_val.npy', y_val)
X_val = pd.read_csv('X_val.csv')
X_val= X_val[X_val.columns.drop(list(X_val.filter(regex='Unnamed')))]

print("建模")
params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}

MAX_ROUNDS = 50
val_pred = []
test_pred = []
cate_vars = []

#label=y_train[:, 0]

for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items['生鮮與否']] * num_days) * 0.25 + 1
    )
    
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items['生鮮與否'] * 0.25 + 1,
        categorical_feature=cate_vars)
    
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=5
    )
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

mse =  mean_squared_error(
    y_val, np.array(val_pred).transpose())

weight = items['生鮮與否'] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)
print('nwrmsle = {}'.format(err))

y_val_new = np.array(val_pred).transpose()
df_preds = pd.DataFrame(
    y_val_new, index=df_2017.index,
    columns=pd.date_range("2017-07-26", periods=16)
).stack().to_frame('銷售量_log_pred')
df_preds.index.set_names(["分店編號", "商品編號", "日期"], inplace=True)
df_preds['銷售量_log_pred_b1'] = df_preds['銷售量_log_pred'] / np.expm1(mse)
df_preds['銷售量_log_pred_b1'] = np.expm1(df_preds['銷售量_log_pred_b1'] )
df_preds['銷售量_log_pred_b2'] = df_preds['銷售量_log_pred'] * np.expm1(mse)
df_preds['銷售量_log_pred_b2'] = np.expm1(df_preds['銷售量_log_pred_b2'] )
df_preds['銷售量_log_pred'] = np.expm1(df_preds['銷售量_log_pred'])


df_true = pd.DataFrame(
    y_val, index=df_2017.index,
    columns=pd.date_range("2017-07-26", periods=16)
).stack().to_frame('銷售量_log_true')
df_true.index.set_names(["分店編號", "商品編號", "日期"], inplace=True)
df_true['銷售量_log_true'] = np.expm1(df_true['銷售量_log_true'])
a= pd.concat([df_preds,df_true],axis = 1)

a['銷售量_log_pred_b1'] = a['銷售量_log_pred'] / np.expm1(mse)
a['銷售量_log_pred_b2'] = a['銷售量_log_pred'] * np.expm1(mse)
a['採購量'] = df_preds['銷售量_log_pred_b2'] * 2
a['採購量'] =np.round(a['採購量'] )

# sell
len(a[a['銷售量_log_true']<a['採購量']]) / len(a)
a['採購量-銷售量'] = a['採購量'] - a['銷售量_log_true']

# rev
a['真正銷售量'] = np.where(a['採購量-銷售量']<0, a['銷售量_log_true'] + a['採購量-銷售量'],a['銷售量_log_true']  )
rev = a['真正銷售量'] * 27.898
rev_sum = rev.sum()
inventory_cost = a[a['採購量-銷售量']>=0]['採購量-銷售量'] * 3.57
inventory_cost.sum()
inventory_amount = a[a['採購量-銷售量']>=0]['採購量-銷售量'].sum()
purchase_cost = inventory_amount * 4 
all_cost  = inventory_cost.sum()+purchase_cost#+adjinv_cost.abs().sum()
gross_margin =  rev_sum-all_cost


aa = a.iloc[500:1000,::]
機器學習_df = aa[['真正銷售量','銷售量_log_true','採購量', '採購量-銷售量']]
