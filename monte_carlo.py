# Monte Carlo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib import rc
font = {'family' : 'sans-serif',
        'size'   : 24}

rc('font', **font)


df_train = pd.read_csv('../Group Project Stage/sampled_ts_train.csv', keep_default_na =False)

df_train['tran_date'] = pd.to_datetime(df_train['tran_date'])
df_train.set_index('tran_date', inplace=True)


listof = [38767, 55991, 50962, 48676]


subset_df = df_train[df_train['sku_key']==38767]['sales']


subset_df.head()


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


sku = 50962


train, test, predictions = monte_carlo(df_train, sku)



pred_rmse = np.sqrt(metrics.mean_squared_error(test, predictions))
pred_mse = metrics.mean_squared_error(test, predictions)
pred_sales = int(predictions.sum().values)
print('RMSE: %.3f' % pred_rmse)
print('MSE: %.3f' % pred_mse)
print('There are predicted to sell {} items'.format(pred_sales))

true_rmse = np.sqrt(metrics.mean_squared_error(test.iloc[1:], test.iloc[:-1]))
true_mse = metrics.mean_squared_error(test.iloc[1:], test.iloc[:-1])
true_sales = int(test.sum())
print('RMSE: %.3f' % true_rmse)
print('MSE: %.3f' % true_mse)
print('There are predicted to sell {} items'.format(true_sales))


predictions.columns = ['Predictions']
test = pd.DataFrame(test)
test.columns = ['True']


f, ax = plt.subplots(figsize=(15,9))
predictions.plot(ax=ax, color='darkslateblue', linewidth=2)
test.plot(color='darkorange', ax=ax, linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
f.text(1,0.9,'Predictions MSE: {}'.format(round(pred_mse, 3)))
f.text(1,0.85,'Predictions Sales: {}'.format(round(pred_sales)))

f.text(1,0.75,'True MSE: {}'.format(round(true_mse, 3)))
f.text(1,0.7,'True Sales: {}'.format(round(true_sales)))

f.savefig('./monte_carlo_{}.png'.format(sku), bbox_inches='tight')

