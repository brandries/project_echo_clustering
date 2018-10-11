
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import keras
from keras import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
import time
import tensorflow as tf
from matplotlib import rc
from TimeSeriesPrediction import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.models import load_model


font = {'family' : 'sans-serif',
        'size'   : 24}

rc('font', **font)


def periods_w_demand_new(df, sales):
    nuls = []
    not_nuls = []
    nul_count = 0
    notnul_count = 0
    for i in df[sales]:
        nuls.append(nul_count)
        not_nuls.append(notnul_count)
        if i == 0:
            nul_count += 1
            notnul_count = 0
        else:
            nul_count =0
            notnul_count +=1
    return pd.DataFrame([nuls, not_nuls], index=['nuls', 'notnuls']).T


def parameterize_output(x, threshold=0.5):
    '''This small function just takes in a number predicted by the model,
    and makes it discrete.
    The function will round up or down with an equal probability, unless specified. 
    The aim would be to use the number of 0's in the dataset as a guideline for
    rounding up or down. The proportion of 0's can be used to on a scale from 0 to 0.5
    where 0 is all 0's and 0.5 is no zeros.
    
    '''
    import math
    chance = np.random.uniform()
    if chance <= threshold:
        return np.round(x)
    else:
        return np.fix(x)

def is_sales(sales):
    return pd.DataFrame([0 if x != 0 else 1 for x in sales], columns = ['is_sales'])

df = pd.read_csv('../data/snapshot_full_df.csv')
df_train = df.copy()
df_train.sort_values(['store_key', 'sku_key', 'tran_date'], inplace=True)

print('Dataset shape: {}\n {}'.format(df_train.shape, df_train.head()))

df_train = df_train.iloc[:500000]
df_train.drop('Unnamed: 0', axis=1, inplace=True)


def preprocess_df(df):
    df.loc[:,'weekday'] = df.loc[:,'tran_date'].dt.weekday_name
    df.loc[:,'day'] = df.loc[:,'tran_date'].dt.day
    df.loc[:,'month'] = df.loc[:,'tran_date'].dt.month
    df.loc[:,'week'] = df.loc[:,'tran_date'].dt.week
    cat = ['store_region', 'store_grading', 'sku_department',
           'sku_subdepartment', 'sku_category', 'sku_subcategory',
           'time_of_week', 'monthend', 'month', 'week', 'weekday']
    df.loc[:,'time_of_week'] = ['Weekend' if x in ['Saturday', 'Sunday', 'Friday'] else 'Weekday' for x in df.loc[:,'weekday']]
    df.loc[:,'monthend'] = ['Monthend' if x in [25, 26, 27, 28, 29, 30,
                                            31, 1, 2, 3, 4, 5] else 'Not-Monthend' for x in df.loc[:,'day']]
    df.drop(['day'], axis=1, inplace=True)
    for i in cat:
        df = df.join(pd.get_dummies(df[i], prefix=i))

    df.drop(cat, axis=1, inplace=True)
    df = df.reset_index(drop=True)
    for i in range(2,12):
        df.loc[:,'agg{}'.format(i)] = df.loc[:,'sales'].rolling(i).mean()
    for i in range(10,11):
        df.loc[:,'max{}'.format(i)] = df.loc[:,'sales'].rolling(i).max()
    for i in range(10,11):
        df.loc[:,'min{}'.format(i)] = df.loc[:,'sales'].rolling(i).min()
    df.dropna(inplace=True)
    
    extra = periods_w_demand_new(df, 'sales')
    sales = is_sales(df['sales'])
    df = df.reset_index(drop=True).join(extra).join(sales)
    
    return df


df_train['tran_date'] = pd.to_datetime(df_train['tran_date'])

print('Preprocessing...')

start = time.time()
full_sales = preprocess_df(df_train)
end = time.time()

print('It took a full {} minutes to preprocess'.format((end-start)/60))

cat = full_sales.columns[4:]
print('The categoricals include {}'.format(cat))

scaler = MinMaxScaler()
scaler.fit(full_sales['sales'].values.reshape(-1,1))

cat2 = ['selling_price', 'avg_discount', 'max10', 'min10',
        'agg2', 'agg3', 'agg4', 'agg5', 'agg6', 'agg7',
        'agg8', 'agg9', 'agg10', 'agg11', 'nuls', 'notnuls', 'is_sales']

scale_add = MinMaxScaler()
scale_add.fit(full_sales[cat2])
full_sales[cat2] = scale_add.transform(full_sales[cat2])

subset_df = full_sales#[full_sales['sku_key']==48676]

ts = TimeSeriesModelling(subset_df)
#ts.set_date('tran_date')
df = ts.scale_data(subset_df, scaler=scaler, values_index='sales')
df = pd.DataFrame(df, columns=['sales'])

X, y = ts.preprocessing(df, lagshift=1)
X = ts.add_additional_feat(subset_df, cat, scale_add)
X = X.reshape(X.shape[0], X.shape[2])
X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)

ts.plot_timeseries('sales')

## Model all features:
print('Training the model:')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
K.clear_session()
adam = Adam(lr=0.0001)
model = Sequential()
model.add(Dense(128, input_shape=(473,), activation='tanh'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=adam)

print(model.summary())

model.fit(X_train, y_train, batch_size=128, epochs=200, callbacks=[early_stop], validation_split=0.1)


#THis is a work in progress, you have to evaluate the entire model first
# me = ModelEvaluation(model)
# me.evaluate_model(X_train, X_test, y_train, y_test)
# me.plot_evaluation(y_train, y_test)


# ### Evaluate only one:
groupings = full_sales[['sku_key', 'store_key', 'sales']].groupby(['sku_key', 'store_key']).sum().sort_values('sales', ascending=False)

#I think this will be the place where you do the full evaluation, looping through this list, predicting each and then getting the sum of the MSE's
subset_df = full_sales[(full_sales['sku_key']==47593)&(full_sales['store_key'] == 4)]


ts = TimeSeriesModelling(subset_df)
df = ts.scale_data(subset_df, scaler=scaler, values_index='sales')
df = pd.DataFrame(df, columns=['sales'])
X, y = ts.preprocessing(df, lagshift=1)
X = ts.add_additional_feat(subset_df, cat, scale_add)
X = X.reshape(X.shape[0], X.shape[2])
X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)
ts.plot_timeseries('sales')

me = ModelEvaluation(model)
me.evaluate_model(X_train, X_test, y_train, y_test)
me.plot_evaluation(y_train, y_test)

#TODO : implement this part of the code somethere write the function
if save_model == True:
    model.save('all_features_ann.h5')  # creates a HDF5 file 'my_model.h5'

elif load_model == True:
    model = load_model('all_features_ann.h5')



# # Blind predict

ts = TimeSeriesModelling(subset_df)
ts.set_date('tran_date')
train = subset_df.iloc[:-90]
test = subset_df.iloc[-90:]

#Write a statement which checks if the last date of train is one day behind the first of test
train.tail(1)
test.head(1)


# BLIND 90 Days
import datetime
predicted = []

zeros = ((train['sales'] == 0).sum() / len(train))
print('We have {} zeros'.format(zeros))
for i in range(90):
    df = ts.scale_data(train, scaler=scaler, values_index='sales')
    df = pd.DataFrame(df, columns=['sales'])
    X, y = ts.preprocessing(df, lagshift=1)
    X = ts.add_additional_feat(train, cat, scale_add)
    X = X.reshape(X.shape[0], X.shape[2])

    predict = model.predict(X[-1,:].reshape(1,X.shape[-1]))    
       
    changes = train.iloc[-1,:].copy()
    changes = pd.DataFrame(changes).T
    changes[cat2] = scale_add.inverse_transform(changes[cat2].values.reshape(1,-1))
    #This is where the add features go
    #Dates
    next_date = pd.to_datetime(train.index[-1] + datetime.timedelta(days=1))
    last_day = next_date.day
    last_week = next_date.week
    last_month = next_date.month
    last_year = next_date.year
    last_dayname = next_date.weekday_name
    
    if last_day in [25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5]:
        changes.loc[:,'monthend_Not-Monthend'] = 0
        changes.loc[:,'monthend_Monthend'] = 1
    else:
        changes.loc[:,'monthend_Not-Monthend'] = 1
        changes.loc[:,'monthend_Monthend'] = 0
    
    if last_dayname in ['Saturday', 'Sunday', 'Friday']:
        changes.loc[:,'time_of_week_Weekday'] = 0
        changes.loc[:,'time_of_week_Weekend'] = 1
    else:
        changes.loc[:,'time_of_week_Weekday'] = 1
        changes.loc[:,'time_of_week_Weekend'] = 0

    for i in range(1, 53):
        changes['week_{}'.format(i)] = 0
        if i == last_week:
            changes['week_{}'.format(i)] = 1
            
    for i in range(1, 13):
        changes['month_{}'.format(i)] = 0
        if i == last_month:
            changes['month_{}'.format(i)] = 1
            
    for i in ['weekday_Friday', 'weekday_Monday', 'weekday_Saturday',
              'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday',
              'weekday_Wednesday']:
        changes[i] = 0
        if i == 'weekday_{}'.format(last_dayname):
            changes.loc[:,i] = 1
    
    random_nr = np.random.uniform()
    
    if random_nr < zeros/10:
        changes.loc[:,'sales'] = 0
        predicted.append(0)
    elif random_nr < zeros/2:
        changes.loc[:,'sales'] = float(scaler.inverse_transform(predict).reshape(-1))/2
        predicted.append(float(scaler.inverse_transform(predict).reshape(-1))/2)
    else:
        changes.loc[:,'sales'] = float(scaler.inverse_transform(predict).reshape(-1))
        predicted.append(float(scaler.inverse_transform(predict).reshape(-1)))
    
    #changes = pd.DataFrame(changes).T
    changes.index = [next_date]
    train = pd.concat([train, changes])
    
    #Aggregates
    for i in range(2,12):
        train.loc[next_date,'agg{}'.format(i)] = train.iloc[-i:,2].rolling(i).mean()[-1]
    train.dropna(inplace=True)
    
    for i in range(10,11):
        train.loc[next_date,'max{}'.format(i)] = train.iloc[-i:,2].rolling(i).max()[-1]
    train.dropna(inplace=True)
    
    for i in range(10,11):
        train.loc[next_date,'min{}'.format(i)] = train.iloc[-i:,2].rolling(i).min()[-1]
    train.dropna(inplace=True)
    
    
    #Nuls
    notnuls = []
    
    nul_counter = 0
    nulidx = -2
    ntick = True
    while ntick == True:
        if np.fix(train.iloc[nulidx, 2]) == 0:
            nul_counter += 1
            nulidx -= 1
        else:
            ntick = False
            
    nnul_counter = 0
    nnulidx = -2
    tick = True
    while tick == True:
        if np.fix(train.iloc[nnulidx, 2]) != 0:
            nnul_counter += 1
            nnulidx -= 1
        else:
            tick = False
    
    is_sales = 0
    if round(train.loc[next_date, 'sales']) == 0:
        is_sales = 0
    else:
        is_sales = 1
    
    train.loc[next_date, 'nuls'] = nul_counter
    train.loc[next_date, 'notnuls'] = nnul_counter
    train.loc[next_date, 'is_sales'] = is_sales
    
    train.loc[next_date, cat2] = scale_add.transform(train.loc[next_date, cat2].values.reshape(1, -1)).reshape(-1)
    

train.reset_index(inplace=True)
train['index'] = pd.to_datetime(train['index'])
train.set_index('index', inplace=True)

test.reset_index(inplace=True)
test['tran_date'] = pd.to_datetime(test['tran_date'])
test.set_index('tran_date', inplace=True)


print('Some of the last predictions: {}'.format(train.tail()))

train['sales'] = train['sales'].astype('float')


f, ax = plt.subplots(figsize=(12,8))

train['sales'].plot(ax=ax)
test['sales'].plot(ax=ax, color='r')
ax.set_title('55991 Sales')
#ax.set_xlim(right=pd.datetime(2018,1,1))
#ax.set_ylim(bottom=-5, top=50)
#f.savefig('90day_pred_vanilla_correct_zeros.png')

test_mse = metrics.mean_squared_error(test['sales'].values, predicted)
print('We have a test MSE of {}'.format(test_mse))



cum_test = sum(test['sales'].values)
cum_pred = sum(predicted)
print('There were {} cumulative test and {} cumulative predicted sales'.format(cum_test, cum_pred))

##This is all old code, verify that it still works and check where it fits in

plot_df = pd.DataFrame(train).join(test, lsuffix='train', rsuffix='test')

plot_df = plot_df[['salestrain', 'salestest']]
plot_df.columns = ['pred', 'true']

threshold = (1-plot_df['true'].value_counts()[0]/len(plot_df))
print('Parameterization threshold is {}'.format(threshold))

plot_df['pred'] = plot_df['pred'].apply(parameterize_output, args=[threshold])
pred = plot_df['pred'].apply(parameterize_output, args=[threshold]).values


plot_df.reset_index(inplace=True)
plot_df['index'] = pd.to_datetime(plot_df['index'])
plot_df.set_index('index', inplace=True)

f, ax = plt.subplots(figsize=(18,8),)
plot_df.plot(ax=ax, color=['darkviolet', 'orange'], linewidth=2.5)
ax.set_title('True and predicted sales', fontdict=font)
ax.set_xlabel('Date', fontdict=font)
ax.set_ylabel('Sales', fontdict=font)

ax.axvline('2017-10-27', color='black')
plt.tick_params(labelsize=14)

plt.show()
f.savefig('55991_unsmoothed_new_14days.png')

plot_df.index[-90].date()

test_df = plot_df[plot_df.index.date >= plot_df.index[-90].date()]


rmse = np.sqrt(metrics.mean_squared_error(test_df.iloc[:,1], test_df.iloc[:,0]))
mse = metrics.mean_squared_error(test_df.iloc[:,1], test_df.iloc[:,0])
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)


test_df.iloc[:,0].sum()


rmse = np.sqrt(metrics.mean_squared_error(test_df.iloc[1:,1], test_df.iloc[:-1,1]))
mse = metrics.mean_squared_error(test_df.iloc[1:,1], test_df.iloc[:-1,1])
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)


test_df.iloc[:,1].sum()

