#!/usr/bin/env python
# coding: utf-8

# !pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org plotly


# ## Import Libraries

from __future__ import absolute_import, division, print_function

import pandas as pd
import os
from time import strptime
import numpy as np
import missingno

import seaborn as sns
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew
from calendar import day_abbr, month_abbr, mdays
import holidays
from datetime import datetime,timedelta

import matplotlib as plt
from matplotlib import pyplot
from prophet import Prophet
import statsmodels
import datetime as dt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import r2_score

import prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import MonthBegin
from pandas.tseries.offsets import MonthEnd

# For better visualization, I will use this plotting parameters

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Set figure width to 12 and height to 9
fig_size[0] = 30
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

def creating_time_series(input_path, ts_path):
    # ## Load Data

    data =  pd.read_excel(input_path)
    print("data is loaded!")
    # Stripping out spaces from ends of names, and replacing internal spaces or different characters with "_"
    data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]
    data.columns = [col.strip().replace('-', '_').lower() for col in data.columns]
    data.columns = [col.strip().replace('&', '_').lower() for col in data.columns]
    data.columns = [col.strip().replace('/', '_').lower() for col in data.columns]

    data= data.replace('[\$,/]', '', regex=True)

    # ## Data Exploration

    # Any of the features has null data

    # -- There are 5 numeric features (all int): Year, Day, Week Number, Ridership and Total Train Trips    
    # -- Year, Month, and Day can be combined to create Date feature
    # -- Month, Rail Corridor, Weekend&Holidays/weekday, Station, and Time Period are categorical 

    categorical_features = data[['month', 'rail_corridor', 'weekend_holidays_weekday', 'station', 'time_period']]

    #Summary of the numerical features
    numeric_features = data[['year','day','week_number','ridership','total_train_trips']]

    # ## Feature Engineering

    # Since this is a time series analysis, I create a Date column

    data['Month'] = [strptime(str(i), '%B').tm_mon for i in data['month']]
    data["date"] = pd.to_datetime(data[['year', 'Month', 'day']])

    # data["ridership_by_trips"] = data["ridership"]/data["total_train_trips"]
    # data.head()

    # Since this is time series data, I will create time series for each different staion in each rail corridors.

    data["weekend_holidays_weekday"].unique()

    data = data.drop(columns = 'weekend_holidays_weekday')
    
    df1 = pd.pivot_table(data, values=['ridership', 'total_train_trips'], index=['date'],
                       columns=['rail_corridor'], aggfunc=np.sum) #, fill_value=0
    df1.columns = [' '.join(col).strip() for col in df1.columns.values]
    df1.index.freq = 'd'
    missingno.bar(df1)
    
    df2 = pd.pivot_table(data, values=['ridership', 'total_train_trips'], index=['date'],
                       columns=['rail_corridor','station'], aggfunc=np.sum) #, fill_value=0
    df2.columns = [' '.join(col).strip() for col in df2.columns.values]
    df2.index.freq = 'd'
    missingno.bar(df2)
    
    df3 = pd.pivot_table(data, values=['ridership', 'total_train_trips'], index=['date'],
                       columns=['rail_corridor','station', 'time_period'], aggfunc=np.sum) #, fill_value=0
    df3.columns = [' '.join(col).strip() for col in df3.columns.values]
    df3.index.freq = 'd'
    missingno.bar(df3) 

    
    data.to_csv(f"{ts_path}/Metrolinx_prepocessed_data.csv")
    df1.to_csv(f"{ts_path}/Rail_Corridor_data.csv")
    df2.to_csv(f"{ts_path}/Station_data.csv")
    df3.to_csv(f"{ts_path}/Time_period_data.csv")
 

    rc_names =[]
    rc_st_names = []
    rc_st_tp_names = []


    # Rail Corridor
    for rc in data.rail_corridor.unique(): 
        df_rc = data[(data["rail_corridor"] == rc)]
        df_rc.rail_corridor.unique()

        df_rc = pd.pivot_table(df_rc, values=['ridership', 'total_train_trips'], index=['date'],
                                               columns=['rail_corridor'], aggfunc=np.sum, fill_value=0)

        df_rc.columns = [' '.join(col).strip() for col in df_rc.columns.values]
        if len(df_rc) > 1000: #(4*365)*0.75:
            rc_names.append(f"Metrolinx_{rc}_Data")
            df_rc.to_csv(f"{ts_path}/Metrolinx_{rc}_Data.csv")

        # Rail Corridor and Station
        for st in data.station.unique():
            df_rc_st = data[(data["rail_corridor"] == rc) & (data["station"] == st)]
            df_rc_st = pd.pivot_table(df_rc_st, values=['ridership', 'total_train_trips'], index=['date'],
                           columns=['rail_corridor', 'station'], aggfunc=np.sum, fill_value=0)
            df_rc_st.columns = [' '.join(col).strip() for col in df_rc_st.columns.values]
            if len(df_rc_st) > 1000: #(4*365)*0.75:
                rc_st_names.append(f"Metrolinx_{rc}_{st}_Data")
                df_rc_st.to_csv(f"{ts_path}/Metrolinx_{rc}_{st}_Data.csv")

            # Rail Corridor, Station, and time_period
            for tp in data.time_period.unique():
                df_rc_st_tp = data[(data["rail_corridor"] == rc) & (data["station"] == st) & (data["time_period"] == tp)]
                df_rc_st_tp = pd.pivot_table(df_rc_st_tp, values=['ridership', 'total_train_trips'], index=['date'],
                               columns=['rail_corridor', 'station', 'time_period'], aggfunc=np.sum, fill_value=0)
                df_rc_st_tp.columns = [' '.join(col).strip() for col in df_rc_st_tp.columns.values]
                if len(df_rc_st_tp) > 1000: #(4*365)*0.75:
                    rc_st_tp_names.append(f"Metrolinx_{rc}_{st}_{tp}_Data")
                    df_rc_st_tp.to_csv(f"{ts_path}/Metrolinx_{rc}_{st}_{tp}_Data.csv")
        
        
    return rc_names, rc_st_names, rc_st_tp_names


def model_evaluation(true_val, pred_values, model, ts_name):
    mse = mean_squared_error(true_val, pred_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_val, pred_values)
    print(f'\n {ts_name} {model} MSE Error: {mse:11.10}\n {ts_name} {model}  RMSE Error: {rmse:11.10} \n {ts_name} {model}  R square: {r2:11.10}' )
    return mse, rmse, r2

def load_single_timeseries(ts_data_path):
    data_all = pd.read_csv(ts_data_path)
    print('time series loaded!')
    column_names = ['date', 'ridership', 'trips']
    data_all.columns = column_names
#     print(data_all.head())
#
#     print(data_all.info())
#     print(len(data_all)) 
    #Imputing missing in the timestamps
   
    data_all['date'] = pd.to_datetime(data_all['date']) + dt.timedelta(days=1)
    data_all = data_all.set_index('date')
    data_all = data_all.resample('1D').mean()
#     print(data_all.index)

    data_all.isnull().sum()
    data_all = data_all.fillna('0')
#     print(data_all.isnull().sum())

#     print(data_all.info())
    data_all = data_all.astype(int)
#     print(data_all.info())
    print("Data All: \n ", data_all.head())
    
    data = data_all.filter(["date","ridership"])
    print("Data: \n ", data.head())
    
    data_regressors = data_all.filter(["date", "trips"])
    print("Data Regressors: \n ", data_regressors.head())
    
    return data, data_all, data_regressors


def timeseries_eda(data):
    data.reset_index().plot(x='date', y='ridership', kind='line', grid=1)
    plt.pyplot.show()

#     adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(data.ridership.values)
#     print('ADF test statistic:', adf)
#     print('ADF p-values:', pval)
#     print('ADF number of lags used:', usedlag)
#     print('ADF number of observations:', nobs)
#     print('ADF critical values:', crit_vals)
#     print('ADF best information criterion:', icbest)

    data.index = pd.to_datetime(data.index)
    data_decompose_add = seasonal_decompose(data, model='additive')
    data_decompose_add.plot().show()

    data_decompose_add_resid = data_decompose_add.resid.sum()
#     print("additive residual" ,data_decompose_add_resid)


def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print(f"Data has a unit root and is non-stationary")


def acf_pacf(ts):    
    # ACF and PACF 

    lag_acf = acf(ts, nlags=20)
    lag_pacf = pacf(ts, nlags=40, method='ols')
    
    # ACF
    plt.figure(figsize=(22,10))

    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    # PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()



def make_plot_block(verif, start_date, end_date, ax=None): 
    df = verif.loc[start_date:end_date,:]
    df.loc[:,'yhat'].plot(lw=2, ax=ax, color='r', ls='-', label='forecasts')
    ax.fill_between(df.index, df.loc[:,'yhat_lower'], df.loc[:,'yhat_upper'], color='coral', alpha=0.3)
    df.loc[:,'y'].plot(lw=2, ax=ax, color='steelblue', ls='-', label='observations')
    ax.grid(ls=':')
    ax.legend(fontsize=15)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
    ax.set_ylabel('piece number', fontsize=15)
    ax.set_xlabel('', fontsize=15)
    ax.set_title(f'{start_date} to {end_date}', fontsize=18)



def regressor_index(m, name):
    """Given the name of a regressor, return its (column) index in the `beta` matrix.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    name: Name of the regressor, as passed into the `add_regressor` function.
    Returns
    -------
    The column index of the regressor in the `beta` matrix.
    """
    return np.extract(
        m.train_component_cols[name] == 1, m.train_component_cols.index
    )[0]


def regressor_coefficients(m):
    """Summarise the coefficients of the extra regressors used in the model.
    For additive regressors, the coefficient represents the incremental impact
    on `y` of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact is equal to `trend(t)` multiplied by the coefficient.
    Coefficients are measured on the original scale of the training data.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    Returns
    -------
    pd.DataFrame containing:
    - `regressor`: Name of the regressor
    - `regressor_mode`: Whether the regressor has an additive or multiplicative
        effect on `y`.
    - `center`: The mean of the regressor if it was standardized. Otherwise 0.
    - `coef_lower`: Lower bound for the coefficient, estimated from the MCMC samples.
        Only different to `coef` if `mcmc_samples > 0`.
    - `coef`: Expected value of the coefficient.
    - `coef_upper`: Upper bound for the coefficient, estimated from MCMC samples.
        Only to different to `coef` if `mcmc_samples > 0`.
    """
    assert len(m.extra_regressors) > 0, 'No extra regressors found.'
    coefs = []
    for regressor, params in m.extra_regressors.items():
        beta = m.params['beta'][:, regressor_index(m, regressor)]
        if params['mode'] == 'additive':
            coef = beta * m.y_scale / params['std']
        else:
            coef = beta / params['std']
        percentiles = [
            (1 - m.interval_width) / 2,
            1 - (1 - m.interval_width) / 2,
        ]
        coef_bounds = np.quantile(coef, q=percentiles)
        record = {
            'regressor': regressor,
            'regressor_mode': params['mode'],
            'center': params['mu'],
            'coef_lower': coef_bounds[0],
            'coef': np.mean(coef),
            'coef_upper': coef_bounds[1],
        }
        coefs.append(record)

    return pd.DataFrame(coefs)


def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)





