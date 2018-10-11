
# coding: utf-8

# # Importing of modules

# In[1]:


# Module imports
from math import sqrt
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


# # Loading of data

# In[2]:


# Load training data
train = pd.read_csv('sampled_ts_train.csv')


# In[3]:


test = pd.read_csv('sampled_ts_test.csv')


# # Exploratory data analysis

# In[4]:


train.head(1) 


# In[5]:


train.tail(1)


# In[6]:


test.head(1)


# In[7]:


test.tail(1)


# In[8]:


train.dtypes


# ### Note: 
# From just looking at the object data types, we can see that the tran_data aka the transaction date is not in right format.  We will write and employ a custom function to parse the date into the right format.

# ### Note:
# 
# At a glance, for store_grading, there appears to be NaN values which we will have to explore further. Ideally, the transaction date or date in general should be our index column.

# ## Checking for null values in our data

# In[9]:


train.isnull().sum()


# ### Note:
# 
# We see that there is a substantial number of null values being counted, which merits further investigation.

# In[10]:


# Getting all unique store regions
store_regions = set(train['store_region'])
store_regions


# ### Clarification of store region
# Our project leader were able to get further information from our client.  nan values in the store regions actually means Namibia and will be coded as such.

# In[11]:


# Getting all unique store gradings
store_grading = set(train['store_grading'])
store_grading


# ### Clarification of store region
# Our project leader were able to get further information from our client.  nan values in the store gradings actually means that the store has not yet been given a rating and we will create a unique category for that.

# ## Sanity check: Value counts 
# We need to do value counts before and after data preprocessing to ensure that we do not discard data without due consideration for data integrity.

# In[12]:


columns = train.columns
columns


# In[13]:


train.info()


# ## Note:
# From just looking at the dataframe or raw data, it is not very clear what hidden data issues might still be present over and above what we have already explored.  
# 
# We will now visually explore our data. After all, a picture is worth a thousand words.

# In[14]:


train.head(1)


# ## Plotting of our our continuous variables, namely sales, selling_price and avg_discount

# In[15]:


train.columns


# In[16]:


cols = ['store_key', 'sku_key', 'selling_price', 'avg_discount', 'store_region', 
        'store_grading', 'sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory']


# In[17]:


for col in cols:
    train.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()


# ## Note and explanation:
# We can see that there negative values present for sales and discount which merits further investigation to interpret. Subsequently, our supervisor were able to confirm our suspicions that negative values, alludes to the fact that customers might return goods when it comes to sales.  A negative discount rate probably, means that the discount was added to adjust for taxation or to overcharge customers, etc. please see the following link https://archive.sap.com/discussions/thread/3810897

# ## Exploring sales by sku (stock keeping unit or item on sale)

# In[18]:


for i in train['sku_key'].unique():
    print('Item with sku_key {}'.format(i))
    train[train['sku_key']==i]['sales'].plot()
    plt.show()


# In[19]:


for i in test['sku_key'].unique():
    print('Item with sku_key {}'.format(i))
    test[test['sku_key']==i]['sales'].plot()
    plt.show()


# ## Plotting sum of all sales by transaction date

# In[20]:


all_sales = train.reset_index().groupby('tran_date').sum()['sales']


# In[21]:


all_sales2 = test.reset_index().groupby('tran_date').sum()['sales']


# In[22]:


all_sales.plot(figsize=(15,5))
plt.show()


# In[23]:


all_sales2.plot(figsize=(15,5))
plt.show()


# ---

# # Data preprocessing

# In[24]:


# Date parsing function
def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')


# In[25]:


# 1.Load training data
# 2.Parsing the date into the appropriate format
# 3.Dealing with nan values in the store grading

train = pd.read_csv('sampled_ts_train.csv', parse_dates = ['tran_date'],
                   index_col='tran_date', date_parser=parse,keep_default_na=False)


# In[26]:


test = pd.read_csv('sampled_ts_test.csv', parse_dates = ['tran_date'],
                   index_col='tran_date', date_parser=parse,keep_default_na=False)


# In[27]:


train.head()


# In[28]:


# count all NaN values
train.isnull().sum()


# ### Note:
# We see that there are no more NAN values any more, whereas the store grading can be taken as NULL, the store region might not yet be encoded properly.

# In[29]:


# Getting all unique store regions
store_regions = set(train['store_region'])
store_regions


# ## Sanity check: Post preprocessing
# We are checking that no data values were discarded indiscriminately

# In[30]:


columns


# In[31]:


train.info()


# In[32]:


for col in cols:
    train.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()


# In[33]:


for col in cols:
    test.groupby(col).count().loc[:,'sales'].plot(kind='bar', figsize=(16,5))
    plt.show()


# --- 

# # Preparing our data for applying Linear Exponential Smoothing

# ### Subsetting data - drop all columns except tran_date, sku_key and sales

# In[34]:


train2 = train.copy()


# In[35]:


train2 = train2.drop(['store_key','selling_price', 'avg_discount',
       'store_region', 'store_grading', 'sku_department', 'sku_subdepartment',
       'sku_category', 'sku_subcategory'], axis=1)


# In[36]:


train2.columns


# In[37]:


test2 =test.copy()


# In[38]:


test2 = test2.drop(['store_key','selling_price', 'avg_discount',
       'store_region', 'store_grading', 'sku_department', 'sku_subdepartment',
       'sku_category', 'sku_subcategory'], axis=1)


# #### Selecting for item with sku_key = 48676

# In[39]:


filter1 = train2['sku_key'] == 48676


# In[40]:


filter2 = test2['sku_key'] == 48676


# In[41]:


train2 = train2[filter1]


# In[42]:


test2 = test2[filter2]


# In[43]:


train2.head(1)


# In[44]:


test2.head(1)


# In[45]:


train2['sales'].plot()


# In[46]:


test2['sales'].plot()


# ### We can see that there is a lot of null values in our data, which is characteristic of intermittent demand

# In[47]:


train2['sales'].value_counts().head(10)


# In[48]:


test2['sales'].value_counts().head(10)


# ### Aggregation of our data by weeks

# ### Scaling our sales data from daily to a weekly aggregation level

# In[49]:


#train = train2.resample('W').sum()
#train.head(1)


# In[50]:


#test = test2.resample('W').sum()
#test.head(1)


# In[51]:


train.columns


# In[52]:


train2.columns


# ### Our aggregation has removed all zero demand occurrences. We will disaggregate our data later after making our forecasts.

# Let’s visualize the data (train and test together) to know how it varies over a time period.
# 
# 

# In[53]:


#Plotting data
train2['sales'].plot(figsize=(15,8), title= 'Sales of item with sku_key 48676', fontsize=14)
test2['sales'].plot(figsize=(15,8), title= 'Sales of item with sku_key 48676', fontsize=14,)
plt.show()


# # Linear Exponential Smoothing Implementation

# In[54]:


# Initialize parameters
y_hat = 1
tau = 1
tau_hat = 1
alpha = 0.1
beta = 0.1


# In[55]:


series = train2['sales']


# In[56]:


series.plot()
plt.show()


# ### Calculating our inter-demand interval

# In[57]:


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


# In[58]:


inter_demand_interval(train['sales'],my_list)
print(my_list)


# In[59]:


# on item with sku_key = 48676
inter_demand_interval(train2['sales'], my_list2)
print(my_list2)


# In[60]:


train['inter_demand_interval'] = my_list


# In[61]:


train2['inter_demand_interval'] = my_list2


# In[62]:


train2['inter_demand_interval'].plot(figsize=(15,8),title='Inter-demand interval for sku_key 48676')
plt.show()


# In[63]:


def smoothed_demand(demand,alpha=None):
    smoothed_demand = [demand[0]] #first value is the same as the demand
    for n in range(1,len(demand)):
        smoothed_demand.append(alpha*demand[n] + (1-alpha) * smoothed_demand[n-1])
    return smoothed_demand


# In[64]:


train2['smoothed_demand'] = smoothed_demand(train2['sales'], alpha=0.8)


# In[65]:


train2.columns


# In[66]:


train2[['sales','smoothed_demand']].plot(figsize = (16,12))


# In[67]:


def smoothed_inter_demand_interval(inter_demand_inverval,beta=None):
    smoothed_inter_demand_interval = [inter_demand_inverval[0]] #first value is the same as the demand
    for n in range(1,len(inter_demand_inverval)):
        smoothed_inter_demand_interval.append(beta*inter_demand_inverval[n] + (1-beta) * smoothed_inter_demand_interval[n-1])
    return smoothed_inter_demand_interval


# In[68]:


train2['smoothed_inter_demand_interval'] = smoothed_inter_demand_interval(train2['inter_demand_interval'], beta=0.1)


# In[69]:


train2[['inter_demand_interval','smoothed_inter_demand_interval']].plot(figsize = (16,12))


# ## Calculating our forecast

# In[70]:


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


# In[71]:


sales = train2['sales']
inter_demand_interval = train2['inter_demand_interval']
smoothed_demand = train2['smoothed_demand']
smoothed_inter_demand_interval = train2['smoothed_inter_demand_interval']


# In[72]:


train2['forecast'] = my_forecast(sales,smoothed_demand,inter_demand_interval, smoothed_inter_demand_interval,0.1)


# In[ ]:


train2[['sales','forecast']].plot()


# ### Alternative

# In[ ]:


train2[['sales','forecast','smoothed_demand','smoothed_inter_demand_interval']]


# In[ ]:


train2[['sales','forecast']].plot()


# In[96]:


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


# In[97]:


sales = train2['sales']
inter_demand_interval = train2['inter_demand_interval']
smoothed_demand = train2['smoothed_demand']
smoothed_inter_demand_interval = train2['smoothed_inter_demand_interval']


# In[98]:


train2['forecast'] = result(sales,smoothed_demand,inter_demand_interval,smoothed_inter_demand_interval,1)
