import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import seaborn as sns
from datetime import datetime

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


data = pd.read_csv('sampled_ts_train.csv', keep_default_na=False)
## Exploratory Data Analysis


data['sales'].plot()
data['store_region'].value_counts().plot(kind='bar')
data['sku_category'].value_counts().sort_values()
data['sku_department'].value_counts()

HighSales = data[data['sku_key'] == 48676]
HighSales.head()


con=HighSales['tran_date']
HighSales['tran_date']=pd.to_datetime(HighSales['tran_date'])
HighSales.set_index('tran_date', inplace=True)

# ### Exploring the rate of sales of sku_key 48676

#convert to time series:
df = HighSales['sales']
df.head() #Exploring sales in 2016
df.describe()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(HighSales):
    
    #Determing rolling statistics
    rolmean = HighSales.rolling( window=12).mean()
    rolstd = HighSales.rolling( window=12).std()
   # moving_avg = ts_log.rolling(12).mean()

    #Plot rolling statistics:
    orig = plt.plot(HighSales, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(HighSales, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# ## Making data Stationary

# Making the mean and variance to remain constant overtime, we are investigating if the mean and standard deviation follows the trends of sales.
# Obviously can note that the time series is not stationary since the mean and standard deviation move in the direction of the sales 

test_stationarity(df)

# #### Using Logarithmic to reduce the trend
# We will be using smoothing method to reduce the trend: Rolling/Moving average

# ### Using moving average to reduce the trend


moving_avg = df.rolling(12).mean()
plt.plot(df)
plt.plot(moving_avg, color = 'red')
plt.figure(figsize = (25,25))


# Subtracting the rolling mean from the moving average

df_moving_avg_diff = df - moving_avg
df_moving_avg_diff.head(13)

df_moving_avg_diff.replace([np.inf, -np.inf], np.nan, inplace = True)


df_moving_avg_diff.dropna(inplace = True)
df_moving_avg_diff.head()


# ## Testing the stationarity after we used the moving average
# we can observe that the mean and variance don't follow any specific trend, but vary slightly with the number of sales


test_stationarity(df_moving_avg_diff)


# The test statitic shows that we are 95% confident that the series is stationary. Since our test statistic is less than 5%

# ### Using the exponential weight mean average to decrease our trend further

exp_weighted_avg = df.ewm(com=0.5).mean()
plt.plot(df)
plt.plot(exp_weighted_avg, color = 'red')


df_ewma_diff = df - exp_weighted_avg
df_ewma_diff.replace([np.inf, -np.inf], np.nan, inplace = True)
df_ewma_diff.dropna(inplace = True)
df_ewma_diff.head()

test_stationarity(df_ewma_diff)


# ## Finally our model is stationary
# There are less variations in mean and standard deviation in magnitude
# -The test statistic is less than 1%, so we are 99% confident that our time series is staionary

# ## Now we are looking at seasonality with trend
# 
# We will be looking at two methods: Differencing and Decomposition to remove trends with seasonality

# ## Differencing


df_diff = df - df.shift(1)
df_diff.replace([np.inf, -np.inf], np.nan, inplace = True)
df_diff.dropna(inplace = True)
df_diff.head()


test_stationarity(df_diff)


# We can see that mean and standard deviation have small variations with time, hence stationary

# ## Seasonal Difference


df_seasonal_difference = df - df.shift(12)
df_seasonal_difference.replace([np.inf, -np.inf], np.nan, inplace = True)
df_seasonal_difference.dropna(inplace = True)
df_seasonal_difference.head()

test_stationarity(df_seasonal_difference)


# ## Decomposing


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df, freq = 42)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# ## After removing the trend and seasonality, we have the following graph

df_decompose = residual
df_decompose.dropna(inplace=True)
test_stationarity(df_decompose)


# ## Forecasting

# Plotting the Atocorrelation and Partial correlation to find the optimal parameters


from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df_diff, nlags=20)
lag_pacf = pacf(df_diff, nlags=20, method='ols')

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_seasonal_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_seasonal_difference.iloc[13:], lags=40, ax=ax2)


# # Grid search(Hyperparameter optimization)


from statsmodels.tsa.arima_model import ARIMA



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# ARIMA(1, 0, 1)x(1, 1, 1, 12) yields the lowest- AIC:1376.914929113491

mod = sm.tsa.statespace.SARIMAX(df, 
                                order=(0,1,1), 
                                seasonal_order=(0,1,1,12),   
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())



results.resid.plot()


print(results.resid.describe())


results.resid.plot(kind='kde')

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start =300, end = 390, dynamic=False)
pred_ci = pred.conf_int()
pred_ci.head()


df_forecast = pred.predicted_mean
df_truth = df['2018-01-01':]

# Compute the mean square error
mse = ((df_forecast - df_truth) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((df_forecast-df_truth)**2)/len(df_forecast))))


df_pred_concat = pd.concat([df_truth, df_forecast])


pred_dynamic = results.get_prediction(start=pd.to_datetime('2017-08-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


ax = df['2016':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], 
                color='r', 
                alpha=.3)

ax.fill_betweenx(ax.get_ylim(), 
                 pd.to_datetime('2017-08-01'), 
                 df.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Sales')

plt.legend()
plt.show()


# Extract the predicted and true values of our time series
df_forecast = pred_dynamic.predicted_mean
df_original = df['2017-08-01':]

# Compute the mean square error
mse = ((df_forecast - df_original) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((df_forecast-df_original)**2)/len(df_forecast))))


# Get forecast of 10 years or 90 months steps ahead in future
forecast = results.get_forecast(steps= 90)
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
forecast_ci.head()


ax = df.plot(label='observed', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='g', alpha=.4)
ax.set_xlabel('Time (months)')
ax.set_ylabel('Sales')

plt.legend()
plt.show()

