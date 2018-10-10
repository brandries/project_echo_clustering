
# coding: utf-8

# # Monte Carlo

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# In[53]:


from matplotlib import rc
font = {'family' : 'sans-serif',
        'size'   : 24}

rc('font', **font)


# In[2]:


df_train = pd.read_csv('../../data/sampled_ts_train.csv', keep_default_na =False)


# In[3]:


df_train['tran_date'] = pd.to_datetime(df_train['tran_date'])
df_train.set_index('tran_date', inplace=True)


# In[4]:


df_train.head()


# In[5]:


listof = [38767, 55991, 50962, 48676]


# In[6]:


subset_df = df_train[df_train['sku_key']==38767]['sales']


# In[7]:


subset_df.head()


# In[8]:


def monte_carlo(df, item):
    subset_df = df[df['sku_key']==item]['sales']
    train = subset_df.iloc[:-90]
    test = subset_df.iloc[-90:]
    
    lists = []
    for i, j in zip(train.value_counts(), train.value_counts().index):
        lists.append([j, i/len(train)])
        
    lists = pd.DataFrame(lists, columns=['value', 'frequency'])
    
    true_freqs = lists.join(pd.DataFrame(lists['frequency'].cumsum()), rsuffix='_cul')
    
    pred = []
    for i in range(1, 91):
        randi = np.random.uniform()
        #print(randi)
        for j in range(len(true_freqs)):
            if randi <= true_freqs.loc[j, 'frequency_cul']:
                pred.append(true_freqs.loc[j, 'value'])
                break
                
    predictions = pd.DataFrame(pred, index=test.index)
    
    f, ax = plt.subplots(figsize=(12,8))
    
    subset_df.plot(ax=ax)
    predictions.plot(ax=ax)
    return train, test, predictions


# In[9]:


true_freqs


# In[10]:


listof


# In[136]:


sku = 50962


# In[137]:


train, test, predictions = monte_carlo(df_train, sku)


# In[138]:


test.sum()


# In[139]:


pred_rmse = np.sqrt(metrics.mean_squared_error(test, predictions))
pred_mse = metrics.mean_squared_error(test, predictions)
pred_sales = int(predictions.sum().values)
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)
print('There are predicted to sell {} items'.format(pred_sales))

true_rmse = np.sqrt(metrics.mean_squared_error(test.iloc[1:], test.iloc[:-1]))
true_mse = metrics.mean_squared_error(test.iloc[1:], test.iloc[:-1])
true_sales = int(test.sum())
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)
print('There are predicted to sell {} items'.format(true_sales))


# In[140]:


predictions.columns = ['Predictions']
test = pd.DataFrame(test)
test.columns = ['True']


# In[141]:


f, ax = plt.subplots(figsize=(15,9))
predictions.plot(ax=ax, color='darkslateblue', linewidth=2)
test.plot(color='darkorange', ax=ax, linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
f.text(1,0.9,'Predictions MSE: {}'.format(round(pred_mse, 3)))
f.text(1,0.85,'Predictions Sales: {}'.format(round(pred_sales)))

f.text(1,0.75,'True MSE: {}'.format(round(true_mse, 3)))
f.text(1,0.7,'True Sales: {}'.format(round(true_sales)))

f.savefig('./images/monte_carlo_{}.png'.format(sku), bbox_inches='tight')

