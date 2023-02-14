# autopep8: off
import pandas as pd
import numpy as np
from scipy.signal import convolve2d

import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from mealpy.evolutionary_based import GA
from mealpy.bio_based import SMA

from joblib import Parallel, delayed
from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution
from optuna.distributions import IntUniformDistribution

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs
from plotly.offline import init_notebook_mode
from plotly.offline import plot
from plotly.offline import iplot
from plotly.subplots import make_subplots
# autopep8: on


def load_data(data_path='digital_marketing_raw.csv',
              drop_columns=['weekly_revenue', 'Date'],
              y_column='weekly_revenue'):

    data = pd.read_csv(
        data_path

    )

    X = data.drop(columns=drop_columns)
    y = data[y_column]
    X.columns
    return X, y, data


class ExponentialSaturation(BaseEstimator, TransformerMixin):
    def __init__(self, exponent=1.):
        self.exponent = exponent

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return 1 - np.exp(-self.exponent * X)


class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, window=1):
        self.strength = strength
        self.window = window

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (
            self.strength ** np.arange(self.window + 1)
        ).reshape(-1, 1)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.window > 0:
            convolution = convolution[: -self.window]
        return convolution


def rmse(y_actual, y_predicted):
    rms = mean_squared_error(y_actual, y_predicted, squared=False)
    return rms


def train_marketing_carryover_saturation(X, y, n_trials=1000):
    adstock = ColumnTransformer(
        [
            ('Facebook_pipe', Pipeline([
                ('carryover', ExponentialCarryover()),
                ('saturation', ExponentialSaturation())
            ]), ['Facebook']),
            ('Youtube_pipe', Pipeline([
                ('carryover', ExponentialCarryover()),
                ('saturation', ExponentialSaturation())
            ]), ['Youtube']),
            ('Magazine_pipe', Pipeline([
                ('carryover', ExponentialCarryover()),
                ('saturation', ExponentialSaturation())
            ]), ['Magazine']),
        ]
    )

    model = Pipeline([
        ('adstock', adstock),
        ('regression', LinearRegression())
    ])

    mape_scorer = make_scorer(rmse)

    tuned_model = OptunaSearchCV(
        estimator=model,
        param_distributions={
            'adstock__Facebook_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__Facebook_pipe__carryover__window': IntUniformDistribution(0, 10),
            'adstock__Facebook_pipe__saturation__exponent': UniformDistribution(0, 0.01),
            'adstock__Youtube_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__Youtube_pipe__carryover__window': IntUniformDistribution(0, 10),
            'adstock__Youtube_pipe__saturation__exponent': UniformDistribution(0, 0.01),
            'adstock__Magazine_pipe__carryover__strength': UniformDistribution(0, 1),
            'adstock__Magazine_pipe__carryover__window': IntUniformDistribution(0, 10),
            'adstock__Magazine_pipe__saturation__exponent': UniformDistribution(0, 0.01),
        },
        n_trials=n_trials,
        cv=TimeSeriesSplit(),
        random_state=0

    )

    # make scorer from custome function
    # cv =  cross_val_score(tuned_model, X, y, cv=TimeSeriesSplit(),
    #                       scoring=mape_scorer
    #                       )

    tuned_model.fit(X, y)
    best_param = tuned_model.best_params_
    best_param = pd.DataFrame.from_dict(
        best_param, orient='index').reset_index()
    best_param.columns = ['para', 'effect']
    pickle.dump(tuned_model, open("tuned_model.dat", 'wb'))
    return tuned_model, best_param

# carry over
# variable_name = 'Youtube'


def load_tuned_model(tune_model_path="tuned_model.dat"):

    class ExponentialSaturation(BaseEstimator, TransformerMixin):
        def __init__(self, exponent=1.):
            self.exponent = exponent

        def fit(self, X, y=None):
            X = check_array(X)
            self._check_n_features(X, reset=True)  # from BaseEstimator
            return self

        def transform(self, X):
            check_is_fitted(self)
            X = check_array(X)
            self._check_n_features(X, reset=False)  # from BaseEstimator
            return 1 - np.exp(-self.exponent * X)

    class ExponentialCarryover(BaseEstimator, TransformerMixin):
        def __init__(self, strength=0.5, window=1):
            self.strength = strength
            self.window = window

        def fit(self, X, y=None):
            X = check_array(X)
            self._check_n_features(X, reset=True)
            self.sliding_window_ = (
                self.strength ** np.arange(self.window + 1)
            ).reshape(-1, 1)
            return self

        def transform(self, X: np.ndarray):
            check_is_fitted(self)
            X = check_array(X)
            self._check_n_features(X, reset=False)
            convolution = convolve2d(X, self.sliding_window_)
            if self.window > 0:
                convolution = convolution[: -self.window]
            return convolution

    tuned_model = pickle.load(open(tune_model_path, 'rb'))
    best_param = tuned_model.best_params_
    best_param = pd.DataFrame.from_dict(
        best_param, orient='index').reset_index()
    best_param.columns = ['para', 'effect']
    return tuned_model, best_param


def effect_plot(X, variable_name, tuned_model, best_param, only_prop_budget):

    # ----------- calculate Carryover_effect-----------
    best_param_tmp = best_param[best_param['para'].str.contains(variable_name)]
    window = best_param_tmp[best_param_tmp['para'].str.contains(
        'window')]['effect'].values[0]
    strength = best_param_tmp[best_param_tmp['para'].str.contains(
        'strength')]['effect'].values[0]
    exponent = best_param_tmp[best_param_tmp['para'].str.contains(
        'exponent')]['effect'].values[0]

    Carryover_effect = ExponentialCarryover(window=window,
                                            strength=strength,
                                            ).fit(X[variable_name].values.reshape(1, -1))
    Carryover_effect = pd.DataFrame(
        Carryover_effect.sliding_window_, columns=['Carryover_effect'])
    Carryover_effect['week'] = range(1, len(Carryover_effect) + 1)
    Carryover_effect

    # sketch Carryover_effect
    Carryover_effect['Carryover_effect'] = round(
        Carryover_effect['Carryover_effect'], 4)

    fig = px.bar(Carryover_effect, x='week', y='Carryover_effect',
                 text="Carryover_effect", title='【' + variable_name + '】 - ' + "Carry-over effect plot")
    fig.update_traces(textfont_size=12, textangle=0,
                      textposition="outside", cliponaxis=False)

    # plot(fig, filename='00_COE_【' + variable_name +
    #      '】 - ' + 'Carry-over effect plot.html')
    plot(fig, filename='00_【' + variable_name +
         '】 - ' + '廣告遞延效用圖.html')

    Carryover_plot = fig

    # ----------- calculate Saturation_effect-----------
    Saturation_effect = ExponentialSaturation(exponent=exponent).fit_transform(
        X[variable_name].values.reshape(1, -1))

    df_tmp = X[[variable_name]]
    df_tmp[variable_name + '_Saturation'] = Saturation_effect[0]
    df_tmp = df_tmp[df_tmp[variable_name] != 0]

    exp_num_list = []
    max_ = round(df_tmp[variable_name].max())
    for num in range(max_, max_ + round(df_tmp[variable_name].std() * 2500), 10):
        exp_num = 1 - np.exp(-exponent * num)
        exp_num_list.append([num, exp_num])
        print(num, exp_num)
        if round(exp_num, 2) == 1:
            break
    exp_num_df = pd.DataFrame(exp_num_list)
    exp_num_df.columns = df_tmp.columns
    exp_num_df[variable_name +
               '_Saturation'] = round(exp_num_df[variable_name + '_Saturation'], 4)
    min_sat_cost = exp_num_df[exp_num_df[variable_name + '_Saturation'] ==
                              exp_num_df[variable_name + '_Saturation'].max()][variable_name].min()
    exp_num_df = exp_num_df[exp_num_df[variable_name] <= min_sat_cost]
    df_tmp = pd.concat([df_tmp, exp_num_df])
    df_tmp = df_tmp.sort_values(variable_name)

    # compute ROAS
    print('Calculate ROAS using array vectorization')

    default_zeros = [0]*len(df_tmp[variable_name])
    match variable_name:
        case 'Facebook':
            X_new = pd.DataFrame({
                'Youtube': default_zeros,
                'Facebook': df_tmp[variable_name].values,
                'Magazine': default_zeros
            })
        case 'Youtube':
            X_new = pd.DataFrame({
                'Youtube': df_tmp[variable_name].values,
                'Facebook': default_zeros,
                'Magazine': default_zeros
            })
        case _: # 'Magazine'
            X_new = pd.DataFrame({
                'Youtube': default_zeros,
                'Facebook': default_zeros,
                'Magazine': df_tmp[variable_name].values
            })
    pred_value = tuned_model.predict(X_new)
    roas_list = pred_value / df_tmp[variable_name].values
    # print('Calculate ROAS')
    # from tqdm import tqdm
    # roas_list = []
    # for expense in tqdm(df_tmp[variable_name]):

    #     if variable_name == 'Facebook':
    #         X_new = pd.DataFrame({
    #             'Youtube': [0,],
    #             'Facebook': [expense,],
    #             'Magazine': [0,]
    #         })
    #     elif variable_name == 'Youtube':
    #         X_new = pd.DataFrame({
    #             'Youtube': [expense,],
    #             'Facebook': [0,],
    #             'Magazine': [0,]
    #         })
    #     else:
    #         X_new = pd.DataFrame({
    #             'Youtube': [0,],
    #             'Facebook': [0,],
    #             'Magazine': [expense,]
    #         })

    #     pred_value = tuned_model.predict(X_new)
    #     roas = pred_value / expense
    #     roas_list.append(roas[0])

    df_tmp['roas'] = roas_list
    df_tmp['revenue'] = df_tmp['roas'] * df_tmp[variable_name]
    df_tmp['revenue_diff'] = df_tmp['revenue'].diff()

    # df_tmp['dydx'] = np.gradient(
    #     df_tmp[variable_name+'_Saturation'], df_tmp['roas'])

    df_tmp['margin'] = df_tmp['revenue'] - df_tmp[variable_name]

    df_tmp['revenue_diff'].describe()
    # df_tmp['dydx'] = np.gradient(
    #     df_tmp['revenue'], df_tmp[variable_name])

    # df_tmp['dydx'] = np.gradient(
    #     df_tmp[variable_name],df_tmp['revenue'])

    prop_point = df_tmp[df_tmp['margin'] ==
                        df_tmp['margin'].max()][variable_name + '_Saturation'].iloc[0]

    df_tmp_prop = df_tmp[df_tmp[variable_name + '_Saturation'] == prop_point]
    df_tmp_prop = df_tmp_prop[df_tmp_prop[variable_name]
                              == df_tmp_prop[variable_name].min()]

    # prop calculation
    df_tmp = df_tmp.sort_values(variable_name)
    X_prop = df_tmp_prop[variable_name].iloc[0]
    Y_prop = df_tmp_prop[variable_name + '_Saturation'].iloc[0]
    roas_prop = round(df_tmp_prop['roas'].iloc[0], 2)
    rev_prop = round(df_tmp_prop['revenue'].iloc[0])
    margin_prop = round(df_tmp_prop['margin'].iloc[0])
    if only_prop_budget:
        return X_prop

    else:
        # max so far calculation
        max_cost = X[variable_name].max()
        df_tmp_max_sofar = df_tmp[df_tmp[variable_name] == max_cost]
        X_max = df_tmp_max_sofar[variable_name].iloc[0]
        Y_max = round(df_tmp_max_sofar[variable_name + '_Saturation'].iloc[0], 4)
        roas_max = round(df_tmp_max_sofar['roas'].iloc[0], 2)
        rev_max = round(df_tmp_max_sofar['revenue'].iloc[0])
        margin_max = round(df_tmp_max_sofar['margin'].iloc[0])

        # sketch Saturation_effect
        # fig = px.line(df_tmp, x=variable_name, y=variable_name+'_Saturation')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tmp[variable_name],
                                 y=df_tmp[variable_name + '_Saturation'],
                                 #  z= df_tmp['revenue'],
                                 mode='lines',
                                 name='Saturation effect',
                                 customdata=df_tmp[[
                                     'revenue', 'roas', 'margin']],
                                 hovertemplate="<br>".join([
                                     "Suggested budget = $ %{x:.2f}",
                                     'Saturation point = %{y:.4f}',
                                     "Expected ROAS = %{customdata[1]:.2f}",
                                     "Expected revenue = $ %{customdata[0]:.2f}",
                                     "Expected margin = $ %{customdata[2]:.2f}",

                                 ])
                                 ))

        # 加入字詞prop
        fig.add_annotation(
            x=X_prop,
            y=Y_prop,
            text="Suggested budget = $" + str(X_prop) + '; '
            + 'Saturation point = ' + str(round(Y_prop * 100, 4)) + '%'
            + '<br> Expected ROAS = ' + str(roas_prop)
            + '; Expected Revenue = $' + str(rev_prop)
            + '<br> Expected max margin = ' + str(margin_prop)
        )  # change here

        # 加入利潤最好的圖示點prop
        fig.add_trace(go.Scatter(
            x=[X_prop],
            y=[Y_prop],
            mode='markers',
            # name=i
        )
        )

        # 加入字詞now
        fig.add_annotation(
            x=X_max,
            y=Y_max,
            text="max budget so far = $" + str(X_max) + '; '
            + 'Saturation point = ' + str(Y_max * 100) + '%'
            + '<br> Expected ROAS = $' + str(roas_max)
            + '; Expected Revenue = $' + str(rev_max)
            + '<br> Expected margin = ' + str(margin_max)
        )  # change here

        # 加入利潤最好的圖示點now
        fig.add_trace(go.Scatter(
            x=[X_max],
            y=[Y_max],
            mode='markers',
            # name=i
        )
        )

        fig.update_layout(
            title='【' + variable_name + '】 - ' + "Saturation effect plot",
            xaxis_title="budget",
            yaxis_title="Saturation effect",
            legend_title="points",
        )

        fig.update_annotations(clicktoshow='onout')

        plot(fig, filename='01_【' + variable_name +
             '】 - ' + '廣告飽和效用圖.html')
        saturation_fig = fig
        # plot(fig, filename='01_【' + variable_name +
        #      '】 - ' + 'Saturation effect plot.html')
        #

        # ----------- calculate margin for Carryover_effect-----------

        # prop calculation
        X_prop = df_tmp_prop[variable_name].iloc[0]
        Y_prop = round(df_tmp_prop['margin'].iloc[0])
        roas_prop = round(df_tmp_prop['roas'].iloc[0], 2)
        rev_prop = round(df_tmp_prop['revenue'].iloc[0])
        margin_prop = round(df_tmp_prop['margin'].iloc[0])
        sat_prop = round(df_tmp_prop[variable_name + '_Saturation'].iloc[0], 4)

        # max so far calculation
        # max_cost = df_tmp[variable_name].max()
        max_cost = X[variable_name].max()
        df_tmp_max_sofar = df_tmp[df_tmp[variable_name] == max_cost]
        X_max = df_tmp_max_sofar[variable_name].iloc[0]
        Y_max = round(df_tmp_max_sofar['margin'].iloc[0])
        roas_max = round(df_tmp_max_sofar['roas'].iloc[0], 2)
        rev_max = round(df_tmp_max_sofar['revenue'].iloc[0])
        margin_max = round(df_tmp_max_sofar['margin'].iloc[0])
        sat_max = round(
            df_tmp_max_sofar[variable_name + '_Saturation'].iloc[0], 4)

        df_tmp = df_tmp.sort_values(variable_name)
        # fig = px.line(df_tmp, x='Facebook', y='margin')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tmp[variable_name],
                                 y=df_tmp['margin'],
                                 #  z= df_tmp['revenue'],
                                 mode='lines',
                                 name='Saturation effect',
                                 customdata=df_tmp[[
                                     'revenue', 'roas', 'margin', variable_name + '_Saturation']],
                                 hovertemplate="<br>".join([
                                     "Suggested budget = $ %{x:.2f}",
                                     'Saturation point = %{customdata[3]:.4f}',
                                     "Expected ROAS = %{customdata[1]:.2f}",
                                     "Expected revenue = $ %{customdata[0]:.2f}",
                                     "Expected margin = $ %{customdata[2]:.2f}",

                                 ])
                                 ))

        # 加入字詞prop
        fig.add_annotation(
            x=X_prop,
            y=Y_prop,
            text="Suggested budget = $" + str(X_prop) + '; '
            + 'Expected max margin = $' + str(margin_prop)
            + '<br> Expected ROAS = ' + str(roas_prop)
            + '; Expected Revenue = $' + str(rev_prop)
            + '<br> Saturation point = ' + str(sat_prop * 100) + '%'
        )  # change here

        # 加入利潤最好的圖示點prop
        fig.add_trace(go.Scatter(
            x=[X_prop],
            y=[Y_prop],
            mode='markers',
            # name=i
        )
        )

        # 加入字詞now
        fig.add_annotation(
            x=X_max,
            y=Y_max,
            text="max budget so far = $" + str(X_max) + '; '
            + 'Expected margin = $' + str(Y_max)
            + '<br> Expected ROAS = $' + str(roas_max)
            + '; Expected Revenue = $' + str(rev_max)
            + '<br> Saturation point = ' + str(sat_max * 100) + '%'
        )  # change here

        # 加入利潤最好的圖示點now
        fig.add_trace(go.Scatter(
            x=[X_max],
            y=[Y_max],
            mode='markers',
            # name=i
        )
        )

        fig.update_layout(
            title='【' + variable_name + '】 - ' + "budget vs margin plot",
            xaxis_title="budget",
            yaxis_title="margin",
            legend_title="points",
        )

        fig.update_annotations(clicktoshow='onout')

        plot(fig, filename='02_【' + variable_name +
             '】 - ' + '預算與毛利分析圖.html')
        budget_margin_plot = fig
        return X_prop, Carryover_plot, saturation_fig, budget_margin_plot


def channel_contribution(X, y, data, tuned_model):

    adstock_data = pd.DataFrame(
        tuned_model.best_estimator_.named_steps['adstock'].transform(X),
        columns=X.columns,
        index=X.index
    )

    weights = pd.Series(
        tuned_model.best_estimator_.named_steps['regression'].coef_,
        index=X.columns
    )

    base = tuned_model.best_estimator_.named_steps['regression'].intercept_

    unadj_contributions = adstock_data.mul(weights).assign(Base=base)
    adj_contributions = (unadj_contributions
                         .div(unadj_contributions.sum(axis=1), axis=0)
                         .mul(y, axis=0)
                         )

    adj_contributions.to_excel('03_各個廣告通路的貢獻金額.xlsx')
    adj_contributions = pd.concat(
        [data[['Date', 'weekly_revenue']], adj_contributions], axis=1)

    adj_contributions['Date'] = pd.to_datetime(adj_contributions['Date'])

    adj_contributions_long = pd.melt(adj_contributions, id_vars='Date', value_vars=[
        'Base', 'Magazine', 'Youtube', 'Facebook'])

    fig = px.area(adj_contributions_long, x="Date", y="value", color="variable",
                  pattern_shape="variable")  # , pattern_shape_sequence=[".", "x", "+"]
    fig.update_layout(hovermode="x unified")

    plot(fig, filename='03_各個廣告通路的貢獻金額.html')


# channel_contribution(X, tuned_model)

# ------------ budget_allocation str---------------
# TODO:
# 1. add constraints function outside of fitness function
# 2. add g1 function to distribute violation function


def budget_allocation(X, tuned_model,
                      budget=65949,
                      fb_min=100,
                      fb_prop_budget=0.5,

                      yt_min=1000,
                      yt_prop_budget=0.5,

                      mg_min=None,
                      mg_prop_budget=0.5,
                      weeks=8,):
    if fb_min is None:
        fb_min = X[X['Facebook'] != 0]['Facebook'].min()
    if yt_min is None:
        yt_min = X[X['Youtube'] != 0]['Youtube'].min()
    if mg_min is None:
        mg_min = X[X['Magazine'] != 0]['Magazine'].min()

    import numpy as np
    np.random.seed(111)

    def fitness(sol):

        def g1(x):
            return sum(list(map(violate_fb_fun, sol[0:8]))) +\
                sum(list(map(violate_yt_fun, sol[8:16]))) +\
                sum(list(map(violate_mg_fun, sol[16::])))

        def violate1(value):
            return 0 if value <= budget else value

        def violate_fb_fun(value):
            return 0 if value <= fb_min else value

        def violate_yt_fun(value):
            return 0 if value <= yt_min else value

        def violate_mg_fun(value):
            return 0 if value <= mg_min else value

        X_new = pd.DataFrame({
            'Facebook': list(map(violate_fb_fun, sol[0:8])),
            'Youtube': list(map(violate_yt_fun, sol[8:16])),
            'Magazine': list(map(violate_mg_fun, sol[16::]))
        })

        pred_results = tuned_model.predict(X_new).sum()

        pred_results -= violate1(g1(sol))**2

        return pred_results

    problem = {
        "fit_func": fitness,
        "lb": [0] * 24,
        "ub": [fb_prop_budget] * 8 + [yt_prop_budget] * 8 + [mg_prop_budget] * 8,
        "minmax": "max",
    }

    problem_model = SMA.BaseSMA(epoch=100, pop_size=100, pr=0.01)
    # problem_model = GA.BaseGA(epoch=100, pop_size=100)
    best_position, best_fitness = problem_model.solve(problem)
    print(problem_model.solution)
    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
    # Best fitness: 1431028.475838841
    # 1560994

    # problem_model = pickle.load(open("metahue_model.dat", 'rb'))

    # def violate_yt_fun(value):
    #     return 0 if value <= yt_min else value

    def violate_fb_fun(value):
        return 0 if value <= fb_min else value

    def violate_yt_fun(value):
        return 0 if value <= yt_min else value

    def violate_mg_fun(value):
        return 0 if value <= mg_min else value

    solution_ = problem_model.solution[0]

    X_new = pd.DataFrame({
        'Facebook': solution_[:8],
        'Youtube': solution_[8:16],
        'Magazine': solution_[16::]
    })

    X_new['Facebook'] = list(map(violate_fb_fun, X_new['Facebook']))
    X_new['Youtube'] = list(map(violate_yt_fun, X_new['Youtube']))
    X_new['Magazine'] = list(map(violate_mg_fun, X_new['Magazine']))
    return X_new, problem_model


# X_new, problem_model = budget_allocation(X=X, tuned_model=tuned_model,
#                                          budget=65949, fb_min=100,
#                                          yt_min=1000, mg_min=None, weeks=8,)
# pickle.dump(problem_model, open("metahue_model.pkl", 'wb'))

# from mealpy.utils import io
# io.save_model(problem_model, "metahue_model.pkl")
# X_new.sum().sum()

# tuned_model.predict(X_new)
# tuned_model.predict(X_new).sum()

# pred inter


def plot_intervals(predictions,
                   lower_bound_col="lower",
                   upper_bound_col="upper",
                   pred_col="pred",
                   actual_col="actual",
                   pred=False, title=None,
                   title_ch=None,):
    # Subset if required
    # predictions = (
    #     predictions.loc[start:stop].copy()
    #     if start is not None or stop is not None
    #     else predictions.copy()
    # )
    data = []

    # Lower trace will fill to the upper trace
    trace_low = go.Scatter(
        x=predictions.index,
        y=predictions[lower_bound_col],
        fill="tonexty",
        line=dict(color="darkblue"),
        fillcolor="rgba(173, 216, 230, 0.4)",
        showlegend=True,
        name=lower_bound_col,
    )
    # Upper trace has no fill
    trace_high = go.Scatter(
        x=predictions.index,
        y=predictions[upper_bound_col],
        fill=None,
        line=dict(color="orange"),
        showlegend=True,
        name=upper_bound_col,
    )

    # Must append high trace first so low trace fills to the high trace
    data.append(trace_high)
    data.append(trace_low)

    if pred:
        trace_pred = go.Scatter(
            x=predictions.index,
            y=predictions[pred_col],
            fill=None,
            line=dict(color="green"),
            showlegend=True,
            name=pred_col,
        )
        data.append(trace_pred)

    # Trace of actual values
    trace_actual = go.Scatter(
        x=predictions.index,
        y=predictions[actual_col],
        fill=None,
        line=dict(color="black"),
        showlegend=True,
        name=actual_col,
    )
    data.append(trace_actual)

    # Layout with some customization
    layout = go.Layout(
        hovermode="x unified",
        # height=900,
        # width=1400,
        title=dict(text="Prediction Intervals" if title is None else title),
        yaxis=dict(title=dict(text="Revenue")),
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day",
                             stepmode="backward"),
                        dict(count=7, label="1w", step="day",
                             stepmode="backward"),
                        dict(count=1, label="1m", step="month",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    # Make sure font is readable
    fig["layout"]["font"] = dict(size=20)
    fig.layout.template = "plotly_white"
    plot(fig, filename=title_ch + '.html')
    # plot(fig, filename='04_Finance_'+'營收分析_人工預算分配與機器學習優化預算分配之比較圖.html')
    return fig


# origin_y = y.iloc·[0:8]
# budget_table(X, title = 'Budget Allocation', title_ch = '人工預算分配與機器學習優化預算分配之比較圖')
def budget_table(X, data, title=None, title_ch=None, weeks=8):
    X = X.iloc[0:weeks]
    data['Date'] = pd.to_datetime(data['Date'])
    data2 = data.iloc[0:weeks]['Date']
    X = pd.concat([data2, X], axis=1)
    X_save = X.copy()
    X = X.sum().reset_index().rename(columns={'index': 'ads', 0: 'budget'})
    fig = go.Figure(data=[go.Pie(labels=X['ads'], values=X['budget'], hole=.3, textinfo='label+percent',
                                 textposition="inside", title=title)])
    # plot(fig, filename='人工預算分配與機器學習優化預算分配之「成本」比較圖.html')
    plot(fig, filename=title_ch + '.html')
    X_save.to_excel(title_ch + '.xlsx')
    return X_save, fig
# fig.add_trace(go.Pie(labels=age["Age_Group"], values=age["percent"],customdata=age["ACC_ID"], textinfo='label+percent',insidetextorientation='horizontal', textfont=dict(color='#000000'), marker_colors=px.colors.qualitative.Plotly),1, 1)
# fig.add_trace(go.Pie(labels=gender["Gender"], values=gender["percent"], customdata=gender["ACC_ID"],textinfo='label+percent',insidetextorientation='horizontal',textfont=dict(color='#000000'),marker_colors=gender["Gender"].map(gender_color)),1, 2)
# fig.add_trace(go.Pie(labels=sample["Sample_Type"],
# values=sample["percent"],
# customdata=sample["ACC_ID"],textinfo='label+percent',texttemplate='%{label}<br>%{percent:.1%f}',insidetextorientation='horizontal',textfont=dict(color='#000000'),marker_colors=px.colors.qualitative.Prism),1,
# 3)


def prediction_interval(data, X, y, X_new, tuned_model, weeks,
                        output_name = '04_1_營收分析_模型與真實狀況',
                        
                        ):

    X_train_fit = tuned_model.predict(X)
    MSE = sum((X_train_fit - y)**2) / (X.shape[0] - X.shape[1] - 1)
    X_train = X.copy()
    X_train.loc[:, 'const_one'] = 1
    XTX_inv = np.linalg.inv(
        np.dot(np.transpose(X_train.values), X_train.values))

    X_test = X_new.copy()
    pred = tuned_model.predict(X_test)
    X_test.loc[:, 'const_one'] = 1
    SE = [np.dot(np.transpose(X_test.values[i]), np.dot(
        XTX_inv, X_test.values[i])) for i in range(len(X_test))]
    results = pd.DataFrame(pred, columns=['pred_weekly_revenue'])

    import scipy.stats

    # find T critical value
    t_value = scipy.stats.t.ppf(q=1 - .05 / 2, df=y.shape[0] - X.shape[1])
    results.loc[:, "lower"] = results['pred_weekly_revenue'].subtract(
        (t_value) * (np.sqrt(MSE + np.multiply(SE, MSE))), axis=0)
    results.loc[:, "upper"] = results['pred_weekly_revenue'].add(
        (t_value) * (np.sqrt(MSE + np.multiply(SE, MSE))), axis=0)
    wn = pd.DataFrame(range(1, len(results) + 1), columns=['week'])
    results = pd.concat([wn, X_new, results], axis=1)
    results = round(results, 2)

    if weeks is None:
        selected_xy = data.reset_index().drop(columns='index')
        selected_xy = selected_xy[['Date', 'weekly_revenue']]
        results = pd.concat([results, selected_xy], axis=1)
        results.to_excel(output_name+'.xlsx')
        results.set_index('Date', inplace=True)
        fig_all = plot_intervals(predictions=results,
                                 lower_bound_col="lower",
                                 upper_bound_col="upper",
                                 pred_col="pred_weekly_revenue",
                                 actual_col="weekly_revenue",
                                 pred=True,
                                #  title='True weekly revenue versus predicted weekly revenue using our new regression model',
                                 title='真實的週營收 vs 預測的週營收',
                                 title_ch=output_name)
        return results, fig_all

    else:
        selected_xy = data[0:weeks]
        selected_xy = selected_xy.reset_index().drop(columns='index')
        manual_budget_xy = selected_xy.copy()
        selected_xy = selected_xy[['Date', 'weekly_revenue']]
        selected_xy.columns = ['Date', 'manual_weekly_revenue']
        # selected_xy['Date'] = pd.to_datetime(selected_xy['Date'])
        results = pd.concat([results, selected_xy], axis=1)
        results.set_index('Date', inplace=True)
        fig_opt = plot_intervals(predictions=results,
                                 lower_bound_col="lower",
                                 upper_bound_col="upper",
                                 pred_col="pred_weekly_revenue",
                                 actual_col="manual_weekly_revenue",
                                 pred=True,
                                 title='Revenue analysis of optimized VS manual budget allocation',
                                 title_ch='06_3_營收分析_人工預算分配與機器學習優化預算分配之比較圖')
        results.to_excel('06_3_營收分析_人工預算分配與機器學習優化預算分配之比較表.xlsx')

        # ----- financial analysis for manual budget allocation -----

        xx = manual_budget_xy[X.columns]
        cost = round(xx .sum().sum(), 2)
        rev = manual_budget_xy['weekly_revenue'].sum()
        roas = round(rev / cost, 2)

        financial_est = pd.DataFrame({
            'interval_items': ['revenue'],
            'total_revenue': [rev],
            'total_ads_spending': [cost],
            'ROAS': [roas]
        })

        financial_est
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=financial_est['interval_items'],
                y=financial_est['total_revenue'],
                name="total revenue",
            ), secondary_y=False,
        )

        fig.add_trace(

            go.Bar(
                x=financial_est['interval_items'],
                y=financial_est['total_ads_spending'],
                name="total ads spending",
            ), secondary_y=False,)

        fig.add_trace(
            go.Scatter(
                x=financial_est['interval_items'],
                y=financial_est['ROAS'],
                text=financial_est['ROAS'],
                textposition='top center',
                mode="lines+text+markers",
                name="ROAS",
                customdata=financial_est[['ROAS']],
                hovertemplate="<br>".join([
                    "ROAS = %{customdata[0]:.2f}",
                ]
                )
            ), secondary_y=True,)

        fig.update_layout(hovermode="x unified")

        fig.update_layout(
            title="Financial analysis of manual budget allocation",
            xaxis_title="95% prediction interval of weekly revenue",
            yaxis_title="revenue",
            legend_title="points",
        )
        plot(fig, filename='06_1_營收分析_人工優化預算分配之財務分析.html')

        financial_est.set_index('interval_items', inplace=True)
        financial_est = financial_est.T
        financial_est.to_excel('06_1_營收分析_人工優化預算分配之財務分析.xlsx')

        # ----- financial analysis for optimized budget allocation -----
        # pred
        cost = round(X_new.sum().sum(), 2)
        rev = results['pred_weekly_revenue'].sum()
        roas = round(rev / cost, 2)

        # pred lower
        # cost = X_new.sum().sum()
        lb_rev = results['lower'].sum()
        lb_roas = round(lb_rev / cost, 2)

        # pred upper
        # cost = X_new.sum().sum()
        ub_rev = results['upper'].sum()
        ub_roas = round(ub_rev / cost, 2)

        financial_est = pd.DataFrame({
            'interval_items': ['predicted_revenue', 'lower_revenue', 'upper_revenue'],
            'total_revenue': [rev, lb_rev, ub_rev],
            'total_ads_spending': [cost, cost, cost],
            'ROAS': [roas, lb_roas, ub_roas]
        })

        financial_est.sort_values(by='ROAS', ascending=True, inplace=True)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=financial_est['interval_items'],
                y=financial_est['total_revenue'],
                name="total revenue",
            ), secondary_y=False

        )

        fig.add_trace(

            go.Bar(
                x=financial_est['interval_items'],
                y=financial_est['total_ads_spending'],
                name="total ads spending",
            ), secondary_y=False

        )

        fig.add_trace(
            go.Scatter(
                x=financial_est['interval_items'],
                y=financial_est['ROAS'],
                text=financial_est['ROAS'],
                textposition='top center',
                mode="lines+text+markers",
                name="ROAS",
                customdata=financial_est[['ROAS']],
                hovertemplate="<br>".join([
                    "Expected ROAS = %{customdata[0]:.2f}",
                ]
                )
            ), secondary_y=True,)
        fig.update_layout(hovermode="x unified")

        fig.update_layout(
            title="Financial analysis of optimized budget allocation",
            xaxis_title="95% prediction interval of weekly revenue",
            yaxis_title="revenue",
            legend_title="points",
        )
        plot(fig, filename='06_2_營收分析_機器學習優化預算分配之財務分析.html')
        fig_06_2 = fig

        financial_est.set_index('interval_items', inplace=True)
        financial_est = financial_est.T
        financial_est.to_excel('06_2_營收分析_機器學習優化預算分配之財務分析.xlsx')
        # financial_est.to_excel('2_營收差異分析_人工預算分配與機器學習優化預算分配之比較表.xlsx')
        return results, financial_est, fig_opt, fig_06_2


# pred result of 8
# results_opt_budget, financial_est_opt_budget = prediction_interval(
#     data=data,
#     X=X, y=y, X_new=X_new, tuned_model=tuned_model,
#     weeks=8
# )

#  pred all result
# results_all, financial_est_all = prediction_interval(data=data,
#                                              X=X, y=y,
#                                              X_new=X,
#                                              tuned_model=tuned_model,
#                                              weeks=None)


# metrics
def performance_metrics(X, y, results,
                        output_name='04_2_營收分析_模型表現.xlsx'
                        ):
    mae = mean_absolute_error(y, results['pred_weekly_revenue'])
    mse = mean_squared_error(y, results['pred_weekly_revenue'], squared=True)
    rmse = mean_squared_error(y, results['pred_weekly_revenue'], squared=False)
    mape = mean_absolute_percentage_error(
        results['pred_weekly_revenue'], y)  # 0.08
    Adj_r2 = 1 - (1 - r2_score(y, results['pred_weekly_revenue'])
                  ) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    perf_table = pd.DataFrame({
        'metrics': ['MAE', 'MSE', 'RMSE', 'MAPE', 'Adj_R'],
        'Objective_function_table': [mae, mse, rmse, mape, Adj_r2],
    })

    perf_table['Objective_function_table'] = round(
        perf_table['Objective_function_table'], 2)

    perf_table.to_excel(output_name+'.xlsx')
    return perf_table

# performance_metrics(y, results_opt_budget)

# Revenue difference between manual & optimized budget allocation


def revenue_difference_analysis(data, X, results, weeks):
    org_xx = data[0:weeks]

    # pred
    cost = round(X.iloc[0:weeks].sum().sum(), 2)
    rev = org_xx['weekly_revenue'].sum()
    roas = round(rev / cost, 2)

    financial_estorg = pd.DataFrame({
        'interval_items': ['predicted_revenue'],
        'total_revenue': [rev],
        'total_ads_spending': [cost],
        'ROAS': [roas]
    })
    financial_estorg.to_excel('original budget and revenue_financial.xlsx')

    # 看上下限與平均狀況對manual的rev
    results.reset_index(inplace=True)
    lower_difference = (results['lower'] - org_xx['weekly_revenue']).sum()
    upper_difference = (results['upper'] - org_xx['weekly_revenue']).sum()
    weekly_revenue_difference = (
        results['pred_weekly_revenue'] - org_xx['weekly_revenue']).sum()

    difff = pd.DataFrame({
        'lower_difference': [lower_difference],
        'weekly_revenue_difference': [weekly_revenue_difference],
        'upper_difference': [upper_difference],
    })
    difff2 = difff.T.reset_index()
    difff2.columns = ['interval_items', 'difference']

    fig = px.bar(difff2, x='interval_items', y='difference',
                 text='difference', title='Revenue difference between manual & optimized budget allocation')
    fig.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
    plot(fig, filename='07_營收差異分析_人工預算分配與機器學習優化預算分配之比較圖.html')

    difff.to_excel(
        '07_營收差異分析_人工預算分配與機器學習優化預算分配之比較表.xlsx')


# revenue_difference_analysis(data=data,
#                             X=X,
#                             y=y,
#                             results=results_opt_budget,
#                             weeks = 8)

# origin_y = y.iloc[0:8]

# import researchpy as rp
# rp.ttest(, y_com)


# se = stdev*np.sqrt(1/len(y) + )
# pd.DataFrame(
#     tuned_model.best_estimator_.named_steps['adstock'].transform(X_new))


# X_new = pd.DataFrame({
#     'Youtube': [0,],
#     'Facebook': [28892,],
#     'Magazine': [0,]
# })

# tuned_model.best_estimator_.named_steps['adstock'].transform(X_new)

# x = np.array([2, 3, 4, 5, 6])
# 1-np.exp(-2*x)

# ------------ TODO : 這邊是考量季節因素等 ---------------
# from mamimo.time_utils import add_time_features, add_date_indicators

# X = (X
#      .pipe(add_time_features, month=True)
#      .pipe(add_date_indicators, special_date=["2020-01-05"])
#      .assign(trend=range(200))
# )


# from mamimo.time_utils import PowerTrend
# from mamimo.carryover import ExponentialCarryover
# from mamimo.saturation import ExponentialSaturation
# # from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# cats =  [list(range(1, 13))] # different months, known beforehand
# cats =  [list(range(1, 13))] # different months, known beforehand

# preprocess = ColumnTransformer(
#     [
#      ('Youtube_pipe', Pipeline([
#             ('carryover', ExponentialCarryover()),
#             ('saturation', ExponentialSaturation())
#      ]), ['Youtube']),
#      ('Facebook_pipe', Pipeline([
#             ('carryover', ExponentialCarryover()),
#             ('saturation', ExponentialSaturation())
#      ]), ['Facebook']),
#      ('Magazine_pipe', Pipeline([
#             ('carryover', ExponentialCarryover()),
#             ('saturation', ExponentialSaturation())
#      ]), ['Magazine']),
#     ('month', OneHotEncoder(sparse=False, categories=cats), ['month']),
#     ('trend', PowerTrend(), ['trend']),
#     ('special_date', ExponentialCarryover(), ['special_date'])
#     ]
# )

# new_model = Pipeline([
#     ('preprocess', preprocess),
#     ('regression', LinearRegression(
#         positive=True) # no intercept because of the months
#     )
# ])

# tuned_model2 = OptunaSearchCV(
#     estimator=new_model,
#     param_distributions={
#         'adstock__Youtube_pipe__carryover__window': IntUniformDistribution(1, 10),
#         'adstock__Youtube_pipe__carryover__strength': UniformDistribution(0, 1),
#         'adstock__Youtube_pipe__saturation__exponent': UniformDistribution(0, 1),
#         'adstock__Facebook_pipe__carryover__window': IntUniformDistribution(1, 10),
#         'adstock__Facebook_pipe__carryover__strength': UniformDistribution(0, 1),
#         'adstock__Facebook_pipe__saturation__exponent': UniformDistribution(0, 1),
#         'adstock__Magazine_pipe__carryover__window': IntUniformDistribution(1, 10),
#         'adstock__Magazine_pipe__carryover__strength': UniformDistribution(0, 1),
#         'adstock__Magazine_pipe__saturation__exponent': UniformDistribution(0,1),
#     },
#     cv=TimeSeriesSplit(),
#     random_state=0,
#     n_trials=100
# )

# print(cross_val_score(tuned_model2, X, y, cv=TimeSeriesSplit(),
#                     #   scoring=mape_scorer
#                       ))
