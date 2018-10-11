# # Importing of modules
from math import sqrt
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

# Load training data
train = pd.read_csv('sampled_ts_train.csv')
test = pd.read_csv('sampled_ts_test.csv')


# # Exploratory data analysis
# ### Note: 
# From just looking at the object data types, we can see that the tran_data aka the transaction date is not in right format.  We will write and employ a custom function to parse the date into the right format.

# ### Note:
# 
# At a glance, for store_grading, there appears to be NaN values which we will have to explore further. Ideally, the transaction date or date in general should be our index column.


# Getting all unique store regions
store_regions = set(train['store_region'])
store_regions


# ### Clarification of store region
# Our project leader were able to get further information from our client.  nan values in the store regions actually means Namibia and will be coded as such.

# Getting all unique store gradings
store_grading = set(train['store_grading'])
store_grading


# ### Clarification of store region
# Our project leader were able to get further information from our client.  nan values in the store gradings actually means that the store has not yet been given a rating and we will create a unique category for that.

# ## Sanity check: Value counts 
# We need to do value counts before and after data preprocessing to ensure that we do not discard data without due consideration for data integrity.

columns = train.columns

# ## Note:
# From just looking at the dataframe or raw data, it is not very clear what hidden data issues might still be present over and above what we have already explored.  
# 
# We will now visually explore our data. After all, a picture is worth a thousand words.


# ## Plotting of our our continuous variables, namely sales, selling_price and avg_discount


cols = ['store_key', 'sku_key', 'selling_price', 'avg_discount', 'store_region', 
        'store_grading', 'sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory']


for col in cols:
    train.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()


# ## Note and explanation:
# We can see that there negative values present for sales and discount which merits further investigation to interpret. Subsequently, our supervisor were able to confirm our suspicions that negative values, alludes to the fact that customers might return goods when it comes to sales.  A negative discount rate probably, means that the discount was added to adjust for taxation or to overcharge customers, etc. please see the following link https://archive.sap.com/discussions/thread/3810897

# ## Exploring sales by sku (stock keeping unit or item on sale)


for i in train['sku_key'].unique():
    print('Item with sku_key {}'.format(i))
    train[train['sku_key']==i]['sales'].plot()
    plt.show()


for i in test['sku_key'].unique():
    print('Item with sku_key {}'.format(i))
    test[test['sku_key']==i]['sales'].plot()
    plt.show()


# ## Plotting sum of all sales by transaction date

all_sales = train.reset_index().groupby('tran_date').sum()['sales']
all_sales2 = test.reset_index().groupby('tran_date').sum()['sales']


all_sales.plot(figsize=(15,5))
plt.show()


all_sales2.plot(figsize=(15,5))
plt.show()


# ---

# # Data preprocessing


# Date parsing function
def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')

# 1.Load training data
# 2.Parsing the date into the appropriate format
# 3.Dealing with nan values in the store grading

train = pd.read_csv('sampled_ts_train.csv', parse_dates = ['tran_date'],
                   index_col='tran_date', date_parser=parse,keep_default_na=False)

test = pd.read_csv('sampled_ts_test.csv', parse_dates = ['tran_date'],
                   index_col='tran_date', date_parser=parse,keep_default_na=False)



# Getting all unique store regions
store_regions = set(train['store_region'])
store_regions


# ## Sanity check: Post preprocessing
# We are checking that no data values were discarded indiscriminately

for col in cols:
    train.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()

for col in cols:
    test.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()


# --- 

# # Preparing our data for applying Linear Exponential Smoothing

# ### Subsetting data - drop all columns except tran_date, sku_key and sales

train2 = train.copy()
train2 = train2.drop(['store_key','selling_price', 'avg_discount',
       'store_region', 'store_grading', 'sku_department', 'sku_subdepartment',
       'sku_category', 'sku_subcategory'], axis=1)

test2 =test.copy()
test2 = test2.drop(['store_key','selling_price', 'avg_discount',
       'store_region', 'store_grading', 'sku_department', 'sku_subdepartment',
       'sku_category', 'sku_subcategory'], axis=1)


# #### Selecting for item with sku_key = 48676

filter1 = train2['sku_key'] == 48676
filter2 = test2['sku_key'] == 48676
train2 = train2[filter1]
test2 = test2[filter2]


train2['sales'].plot()
test2['sales'].plot()


# ### We can see that there is a lot of null values in our data, which is characteristic of intermittent demand

train2['sales'].value_counts().head(10)
test2['sales'].value_counts().head(10)


# ### Aggregation of our data by weeks

# ### Scaling our sales data from daily to a weekly aggregation level

#train = train2.resample('W').sum()
#train.head(1)


#test = test2.resample('W').sum()
#test.head(1)


# ### Our aggregation has removed all zero demand occurrences. We will disaggregate our data later after making our forecasts.

# Let’s visualize the data (train and test together) to know how it varies over a time period.

#Plotting data
train2['sales'].plot(figsize=(15,8), title= 'Sales of item with sku_key 48676', fontsize=14)
test2['sales'].plot(figsize=(15,8), title= 'Sales of item with sku_key 48676', fontsize=14,)
plt.show()


# # Linear Exponential Smoothing Implementation

# Initialize parameters
y_hat = 1
tau = 1
tau_hat = 1
alpha = 0.1
beta = 0.1

series = train2['sales']

series.plot()
plt.show()


# ### Calculating our inter-demand interval

my_list = list() # for the entire training set
my_list2 = list() # for sku_key = 48676

def inter_demand_interval(series, my_list):
    for i in series:
        if i != 0:
            count = 0
            my_list.append(count)
        else:
            count += 1
            my_list.append(count)


inter_demand_interval(train['sales'],my_list)
print(my_list)

# on item with sku_key = 48676
inter_demand_interval(train2['sales'], my_list2)
print(my_list2)

train['inter_demand_interval'] = my_list
train2['inter_demand_interval'] = my_list2

train2['inter_demand_interval'].plot(figsize=(15,8),title='Inter-demand interval for sku_key 48676')
plt.show()


def smoothed_demand(demand,alpha=None):
    smoothed_demand = [demand[0]] #first value is the same as the demand
    for n in range(1,len(demand)):
        smoothed_demand.append(alpha*demand[n] + (1-alpha) * smoothed_demand[n-1])
    return smoothed_demand

train2['smoothed_demand'] = smoothed_demand(train2['sales'], alpha=0.8)
train2.columns



train2[['sales','smoothed_demand']].plot(figsize = (16,12))


def smoothed_inter_demand_interval(inter_demand_inverval,beta=None):
    smoothed_inter_demand_interval = [inter_demand_inverval[0]] #first value is the same as the demand
    for n in range(1,len(inter_demand_inverval)):
        smoothed_inter_demand_interval.append(beta*inter_demand_inverval[n] + (1-beta) * smoothed_inter_demand_interval[n-1])
    return smoothed_inter_demand_interval



train2['smoothed_inter_demand_interval'] = smoothed_inter_demand_interval(train2['inter_demand_interval'], beta=0.1)


train2[['inter_demand_interval','smoothed_inter_demand_interval']].plot(figsize = (16,12))


# ## Calculating our forecast


def my_forecast(sales,smoothed_demand, inter_demand_interval, smoothed_inter_demand_interval, beta):
    forecast = list()
    for j in smoothed_inter_demand_interval:
        for i in sales:
            for k in smoothed_demand:
                for l in inter_demand_interval:
                    if j == 0:
                         forecast.append(k)
                    elif i != 0:
                        forecast.append(k/j)
                    else:
                         forecast.append((k/j) * (1 - beta*l/ 2*j))


sales = train2['sales']
inter_demand_interval = train2['inter_demand_interval']
smoothed_demand = train2['smoothed_demand']
smoothed_inter_demand_interval = train2['smoothed_inter_demand_interval']

train2['forecast'] = my_forecast(sales,smoothed_demand,inter_demand_interval, smoothed_inter_demand_interval,0.1)

train2[['sales','forecast']].plot()


# ### Alternative

train2[['sales','forecast','smoothed_demand','smoothed_inter_demand_interval']]


train2[['sales','forecast']].plot()


def result(sales,smoothed_demand, inter_demand_interval, smoothed_inter_demand_interval, beta):
    my_forecast = list()
    for n in (0,len(sales)):
        if smoothed_inter_demand_interval[n] == 0:
            my_forecast.append(sales[n])
            print(my_forecast)
        elif sales[n] != 0:
            my_forecast.append(smoothed_demand[n]/smoothed_inter_demand_interval[n])
            print(my_forecast)
        else:
            my_forecast.append((smoothed_demand[n]/smoothed_inter_demand_interval[n])*(1-beta/2*smoothed_inter_demand_interval[n]))                   
            print(my_forecast)
    return my_forecast


sales = train2['sales']
inter_demand_interval = train2['inter_demand_interval']
smoothed_demand = train2['smoothed_demand']
smoothed_inter_demand_interval = train2['smoothed_inter_demand_interval']


train2['forecast'] = result(sales,smoothed_demand,inter_demand_interval,smoothed_inter_demand_interval,1)

