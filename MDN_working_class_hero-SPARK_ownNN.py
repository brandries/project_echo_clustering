
# coding: utf-8

# In[1]:


from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import col,lag
from pyspark.sql.types import IntegerType, DateType


# In[2]:


# create entry points to spark
try:
    sc.stop()
except:
    pass
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
sc=SparkContext()
spark = SparkSession(sparkContext=sc)


# In[13]:


from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    './BCX-Insights-89e8cf3bed78.json')

project_id = 'bcx-insights'
client = bigquery.Client(credentials= credentials,project=project_id)

query_job = client.query("""
  SELECT *
  FROM echo_retail_insights.sales_timeseries_train
  """)


# In[ ]:


results = query_job.result()
df = results.to_dataframe()


# In[119]:


df = spark.read.csv('sampled_ts_train.csv',  header=True)


# In[122]:


df = df.withColumn('tran_date', df['tran_date'].cast(DateType()))


# In[123]:


df.printSchema()


# In[124]:


df.show(1)


# In[125]:


def date_features(df,timestamp_column):
  
    import pyspark.sql.functions as F
  
    df = df.withColumn('date', F.to_date(F.col(timestamp_column), 'yyyy-mm-dd'))            .withColumn('day_of_week',F.date_format(F.col(timestamp_column),'u'))           .withColumn('month',F.month('date'))           .withColumn('day_of_month',F.dayofmonth('date'))           .withColumn('week_of_year',F.weekofyear('date'))
   
    df = df.withColumn('weekend',F.when(F.col('day_of_week') ==1,'weekend')                                  .when(F.col('day_of_week') ==6,'weekend')                                  .when(F.col('day_of_week') ==7,'weekend')                                  .otherwise('Weekday'))
 
    df = df.withColumn('day',F.when(F.col('day_of_week') == 7,'Saturday')                              .when(F.col('day_of_week') == 2,'Monday')                              .when(F.col('day_of_week') == 3,'Tuesday')                              .when(F.col('day_of_week') == 4,'Wednesday')                              .when(F.col('day_of_week') == 5,'Thursday')                              .when(F.col('day_of_week') == 6,'Friday')                              .otherwise('Sunday'))
    df = df.withColumn('month_end', F.when(F.col('day_of_month') == 25,'month_end')                                     .when(F.col('day_of_month') == 26,'month_end')                                     .when(F.col('day_of_month') == 27,'month_end')                                     .when(F.col('day_of_month') == 28,'month_end')                                     .when(F.col('day_of_month') == 29,'month_end')                                     .when(F.col('day_of_month') == 30,'month_end')                                     .when(F.col('day_of_month') == 31,'month_end')                                     .when(F.col('day_of_month') == 1,'month_end')                                     .when(F.col('day_of_month') == 2,'month_end')                                     .when(F.col('day_of_month') == 3,'month_end')                                     .when(F.col('day_of_month') == 4,'month_end')                                     .when(F.col('day_of_month') == 5,'month_end')                                     .otherwise('not_month_end'))

    df = df.withColumn('christmas', F.when(((F.col('month') == 12) & (F.col('day_of_month') == 20)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 21)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 22)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 23)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 24)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 25)),'christmas')                                     .otherwise('not_christmas'))

  

    return df


# In[126]:


df = date_features(df,'tran_date')


# In[127]:


df.select('date','month','day','day_of_week','weekend','day_of_month','week_of_year','month_end','christmas').show()


# In[128]:


df.show(10)


# In[129]:


df.columns


# Make sure all the dates are represented

# In[139]:


for i in engineered:
    print(i, df.select(i).toPandas()[i].unique())


# You are shifting the sales by one day. thus transferring some of the data of one product into the next product. 
# You will thus have to split the dataframes into individual 10 000 frames, and then perform the processing on each of the data frames. 

# In[130]:


w_1 = Window().partitionBy().orderBy(col('date'))
for i in range(2,8):
    df = df.withColumn('sales_lag_{}'.format(i),lag('sales',count=i).over(w_1).cast(IntegerType()))
    
df = df.withColumn('sales', df['sales'].cast(IntegerType()))
df = df.withColumn('avg_discount', df['avg_discount'].cast(DoubleType()))
df = df.withColumn('selling_price', df['selling_price'].cast(DoubleType()))

df = df.drop('tran_date')


# In[131]:


df.select("sales", "sales_lag_2", "sales_lag_3", "sales_lag_4", "sales_lag_5", "sales_lag_6", "sales_lag_7").show()


# In[132]:


df.printSchema()


# In[133]:


df.createOrReplaceTempView("sales_table")
spark.catalog.listTables()


# In[134]:


df = df.select('sales',
               'selling_price',
               'avg_discount',
               'store_region',
               'store_grading',
               'sku_department',
               'sku_subdepartment',
               'sku_category',
               'sku_subcategory',
               'day_of_week',
               'month',
               'day_of_month',
               'week_of_year',
               'weekend',
               'day',
               'month_end',
               'christmas',
               'sales_lag_7',
               'sales_lag_2',
               'sales_lag_3',
               'sales_lag_4',
               'sales_lag_5',
               'sales_lag_6')


# In[135]:


df.printSchema()


# In[136]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer


# In[144]:


df = df.na.drop()


# In[153]:


categoricals = ['store_region',
                'store_grading',
                'sku_department',
                'sku_subdepartment',
                'sku_category',
                'sku_subcategory',
                'day_of_week',
                'month',
                'day_of_month',
                'week_of_year',
                'weekend',
                'day',
                'month_end',
                'christmas',]
indexed_df = df
new_cats = []
for i in categoricals:
    indexer = StringIndexer(inputCol=i, outputCol='{}_numeric'.format(i)).fit(indexed_df)
    indexed_df = indexer.transform(indexed_df)
    indexed_df = indexed_df.drop(i)
    new_cats.append('{}_numeric'.format(i))


# In[154]:


for i in new_cats:
    encoder = OneHotEncoder(inputCol=i, outputCol='{}_vector'.format(i))
    indexed_df = encoder.transform(indexed_df)
    indexed_df = indexed_df.drop(i)


# In[155]:


indexed_df.show(1)


# In[162]:


feats = ['selling_price',
         'avg_discount',
         'sales_lag_7',
         'sales_lag_2',
         'sales_lag_3',
         'sales_lag_4',
         'sales_lag_5',
         'sales_lag_6',
         'store_region_numeric_vector',
         'store_grading_numeric_vector',
         'sku_department_numeric_vector',
         'sku_subdepartment_numeric_vector',
         'sku_category_numeric_vector',
         'sku_subcategory_numeric_vector',
         'day_of_week_numeric_vector',
         'month_numeric_vector',
         'day_of_month_numeric_vector',
         'week_of_year_numeric_vector',
         'weekend_numeric_vector',
         'day_numeric_vector',
         'month_end_numeric_vector',
         'christmas_numeric_vector']


# In[219]:


#ALTERNATIVE METHOD

input_data = indexed_df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))


# In[163]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feats, outputCol="features")

final_df = assembler.transform(indexed_df)


# In[165]:


final_df.show(1)


# In[ ]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer, encoder, vector_indexer, assembler])
model = pipeline.fit(df)
transformed = model.transform(df)


# In[168]:


final_df = final_df.select('sales', 'features')


# In[170]:


final_df.show()


# In[171]:


from pyspark.ml.feature import StandardScaler


# In[195]:


standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

scaler = standardScaler.fit(final_df)
scaled_df = scaler.transform(final_df)
scaled_df = scaled_df.drop('features')
scaled_df.take(2)


# In[196]:


train_data, test_data = scaled_df.randomSplit([.8,.2])


# In[197]:


train_data.show(1)


# In[175]:


from pyspark.ml.feature import OneHotEncoder
import keras
from keras import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# In[198]:


K.clear_session()
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

model = Sequential()
model.add(Dense(32, input_shape=(239,), activation='tanh'))
model.add(Dense(1))

opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)


# In[199]:


model.summary()


# In[200]:


from elephas.utils.rdd_utils import to_simple_rdd


# In[201]:


rdd = train_data.rdd


# In[203]:


from elephas.spark_model import SparkModel
from elephas.optimizers import Adam


# In[222]:


spark_model = SparkModel(model, frequency='epoch', mode='synchronous', num_workers=4, elephas_optimizer=Adam())
spark_model.fit(rdd, epochs=20, batch_size=500, verbose=1, validation_split=0.1)


# In[ ]:


spark_model.fit()


# In[23]:


def make_aggregates(self, df, sales_column, aggregate_range = 10):
        agg_index = []
        for i in range(2, 2 + aggregate_range):
            df.loc[:,'agg{}'.format(i)] = df.loc[:,sales_column].rolling(i).mean()
            agg_index.append('agg{}'.format(i))
        print('Aggregates done')
        return df


# In[ ]:


def make_min_max(self, df, sales_column, mins=True, maxes=True):
        min_index = []
        max_index = []
        if maxes == True:
            for i in range(10,11):
                df.loc[:,'max{}'.format(i)] = df.loc[:,sales_column].rolling(i).max()
                max_index.append('max{}'.format(i))
        if mins == True:
            for i in range(10,11):
                df.loc[:,'min{}'.format(i)] = df.loc[:,sales_column].rolling(i).min()
                min_index.append('min{}'.format(i))
        print('Done min max features')
        return df
    
    def periods_w_demand_new(self, df, sales_column):
        nuls = []
        not_nuls = []
        nul_count = 0
        notnul_count = 0
        for i in df[sales_column]:
            nuls.append(nul_count)
            not_nuls.append(notnul_count)
            if i == 0:
                nul_count += 1
                notnul_count = 0
            else:
                nul_count =0
                notnul_count +=1
        print('Not null done')
        return df.join(pd.DataFrame([nuls, not_nuls], index=['nuls', 'notnuls']).T)


# In[ ]:


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


# In[ ]:


class TimeSeriesModelling():
    from sklearn import metrics
    def __init__(self, df):
        self.df = df

    
    def train_test_split(self, X, y, train_size=0.75):
        self.X_train = X[:round((len(X)*train_size))]
        self.X_test = X[round((len(X)*train_size)):]
        self.y_train = y[:round((len(y)*train_size))]
        self.y_test = y[round((len(y)*train_size)):]
        return self.X_train, self.X_test, self.y_train, self.y_test

    
    def plot_timeseries(self, y, train=False, test=False):
        y.plot(kind='line', figsize=(15,5))  


# In[ ]:


class ModelEvaluation:
    def __init__(self, model):
        self.model = model
        
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        self.mse_test = self.model.evaluate(X_test, y_test, batch_size=1)
        self.mse_train = self.model.evaluate(X_train, y_train, batch_size=1)
        print('The model has a test MSE of {} and a train MSE of {}.'.format(self.mse_test, self.mse_train))
        
        self.y_pred = self.model.predict(X_test, batch_size=1)
        self.y_hat = self.model.predict(X_train, batch_size=1)
        
        self.r2_test = metrics.r2_score(y_test, self.y_pred)
        self.r2_train = metrics.r2_score(y_train, self.y_hat)
        
        print('The model has a test R2 of {} and a train R2 of {}.'.format(self.r2_test, self.r2_train))
        
    def plot_evaluation(self, y_train, y_test):
        plt.subplots(figsize=(15,8))
        plt.plot(y_train, c='darkorange')
        plt.plot(self.y_hat, c='teal')
        plt.title('Train dataset and predictions')
        plt.show()
        plt.subplots(figsize=(15,8))
        plt.plot(y_test, c='tomato')
        plt.plot(self.y_pred, c='indigo')
        plt.title('Test dataset and predictions')
        plt.show()
            
    


# In[ ]:


#df_full = pd.read_csv('../../data/sampled_ts_train.csv', keep_default_na =False)
df = pd.read_csv('./snapshot_full_df.csv')


# In[ ]:


df_train = df.copy()


# In[ ]:


df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_train.drop('Unnamed: 0', axis=1, inplace=True)


# In[ ]:


df_train.columns


# In[ ]:


#df_train = df_train.sample(frac=1).reset_index(drop=True)


# In[ ]:


df_train['tran_date'] = pd.to_datetime(df_train['tran_date'])


# In[ ]:


feats = FeatureEngineering()

df = feats.time_lag_features(df_train, 'sales')
df = feats.make_date_features(df, 'tran_date')
df = feats.make_aggregates(df, 'sales', 10)
df = feats.make_min_max(df, 'sales')
df = feats.periods_w_demand_new(df, 'sales')


# In[ ]:


df.columns


# In[ ]:


subset = df[df['sku_key'] == 48676]


# In[ ]:


cat = ['store_region', 'store_grading', 'sku_department', 'sku_subdepartment',
       'sku_category', 'sku_subcategory', 'month', 'week', 'time_of_week', 'monthend',
       'weekday']
cont = ['selling_price', 'avg_discount','shift_1', 'agg2', 'agg3', 'agg4', 'agg5',
        'agg6', 'agg7', 'agg8', 'agg9', 'agg10', 'min10', 'max10',
        'notnuls', 'nuls']


# In[ ]:


X, y = feats.finalize_preprocess(subset, cat, cont, 'sales')


# In[ ]:


features = X.columns
features


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = MinMaxScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1))


# In[ ]:


scale_add = MinMaxScaler()
X[cont] = scale_add.fit_transform(X[cont])


# In[ ]:


ts = TimeSeriesModelling(X)


# In[ ]:


X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)


# In[ ]:


X.shape


# In[ ]:


ts.plot_timeseries(pd.DataFrame(y))


# In[ ]:


y


# ## Model all features:

# In[ ]:


X_train.shape


# In[ ]:


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math

def get_mixture_coef(output, numComonents=24, outputDim=1):
    out_pi = output[:,:numComonents]
    out_sigma = output[:,numComonents:2*numComonents]
    out_mu = output[:,2*numComonents:]
    out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])
    return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)

def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss

class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, inputShape):
        return (inputShape[0], self.outputDim)


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)


# In[ ]:


from keras.optimizers import Adam


# In[ ]:


K.clear_session()
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
model = Sequential()
model.add(Dense(32, input_shape=(100,), activation='relu'))
model.add(MixtureDensity(1,24))

opt = Adam(lr=0.001)
model.compile(loss=mdn_loss(), optimizer=opt)


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, batch_size=4, epochs=200, callbacks=[early_stop], validation_split=0.1)


# In[ ]:


preds = model.predict(X_test)


# In[ ]:


pi, sigma, mu = get_mixture_coef(preds)


# In[ ]:


sess = tf.Session()
with sess.as_default():
    np_pi = pi.eval()
    np_sigma = sigma.eval()
    np_mu = mu.eval()
    
np_mu = np_mu.reshape(np_mu.shape[1], np_mu.shape[0])


# In[ ]:


gaussians = {}
for i in range(24):
     gaussians['gauss_{}'.format(i+1)] = pd.DataFrame([np_pi[:,i], np_sigma[:,i], np_mu[:,i]], index=['pi', 'sigma', 'mu']).T


# In[ ]:


gaussians['gauss_1']


# In[ ]:


import matplotlib.mlab as mlab
import math
f, ax = plt.subplots(figsize=(12,8))
col = ['red', 'blue', 'green', 'm', 'gold', 'black']
for i in gaussians.keys():
    mu = gaussians[i].loc[0, 'mu']
    sigma = gaussians[i].loc[0, 'sigma']
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
ax.set_ylim(bottom=0)
ax.set_xlim(left=-0.2, right=0.2)
plt.show()


# In[ ]:


#f, ax = plt.subplots(figsize=(12,8))
col = ['red', 'blue', 'green', 'm', 'gold', 'black']


for g in range(20):

    f, ax = plt.subplots(ncols=2, figsize=(16,5))
    pis = []
    keys = []
    for i in gaussians.keys():
        
        mu = gaussians[i].loc[g, 'mu']
        sigma = gaussians[i].loc[g, 'sigma']
        pi = gaussians[i].loc[g, 'pi']
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax[0].plot(x,mlab.normpdf(x, mu, sigma), linewidth=2, alpha=np.sqrt(np.sqrt(pi)))
        pis.append(pi)
        keys.append(i)
    ax[1].barh(keys, pis)
    #ax[0].legend(keys, prop={'size': 12})
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=-0.5, right=0.5)
    plt.tick_params(labelsize=12)
    #plt.show()
    f.savefig('./images/mdn/gaussian{}'.format(g), bbox_inches='tight')


# In[ ]:


len(gaussians['gauss_1'].loc[:,'mu'])


# In[ ]:


import scipy


# In[ ]:


#f, ax = plt.subplots(figsize=(12,8))
col = ['red', 'blue', 'green', 'm', 'gold', 'black']


for g in range(300):

    f, ax = plt.subplots(ncols=1, figsize=(16,5))
    pis = []
    keys = []
    for i in gaussians.keys():
        
        mu = gaussians[i].loc[g, 'mu']
        sigma = gaussians[i].loc[g, 'sigma']
        pi = gaussians[i].loc[g, 'pi']
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x,mlab.normpdf(x, mu, sigma), linewidth=2, alpha=np.sqrt(np.sqrt(pi)))
        pis.append(pi)
        keys.append(i)
        d = scipy.zeros(len(x))
        ax.fill_between(x,mlab.normpdf(x, mu, sigma),where=mlab.normpdf(x, mu, sigma)>=d, alpha=np.sqrt(np.sqrt(pi)))
    #ax[0].legend(keys, prop={'size': 12})
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-2, right=2)
    plt.tick_params(labelsize=12)
    #plt.show()
    f.savefig('./images/mdn/gaussian{}'.format(g), bbox_inches='tight')
    f.clear()


# In[ ]:


new_gauss = {}
for rang, mu, sigma, pi in zip(range(len(np_mu)), np_mu, np_sigma, np_pi):
    temp_df = pd.DataFrame([mu, sigma, pi], index=['mu', 'sigma', 'pi']).T
    temp_df['cum_pi'] = temp_df['pi'].cumsum()
    new_gauss[rang] = temp_df


# In[ ]:


mc_guess = np.random.uniform()

for h, i in enumerate(new_gauss[0].loc[:,'cum_pi']):
    if i < mc_guess:
        pass
    else:
        print(new_gauss[0].loc[h,:])
        sales = np.random.normal(loc=new_gauss[0].loc[h,'mu'], scale=new_gauss[0].loc[h,'sigma'])
        print(sales)
        break


# In[ ]:


predictions = []
for row in new_gauss:
    mc_guess = np.random.uniform()

    for h, i in enumerate(new_gauss[row].loc[:,'cum_pi']):
        if i < mc_guess:
            pass
        else:
            sales = np.random.normal(loc=new_gauss[row].loc[h,'mu'], scale=new_gauss[row].loc[h,'sigma'])
            predictions.append(sales)
            break


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))
pd.DataFrame(scaler.inverse_transform(y_test), columns=['true']).plot(ax=ax, linewidth=2)
pd.DataFrame(predictions, columns=['predicted']).plot(ax=ax, linewidth=2)


# Things to do before tomorrow:
# 
# Get MSE of this. 
# 
# Do blind predictions usign this other method previously developed. Then assess MSE and cumulative sales.

# In[ ]:


sum(scaler.inverse_transform(y_test))


# In[ ]:


sum  = 0
for i in predictions:
    if i != 0:
        sum+= i


# In[ ]:


sum


# In[ ]:


del sum


# In[ ]:


subset_df = full_sales[full_sales['sku_key']==48676]


# In[ ]:


subset_df.head()


# In[ ]:


ts = TimeSeriesModelling(subset_df)
df = ts.scale_data(subset_df, scaler=scaler, values_index='sales')
df = pd.DataFrame(df, columns=['sales'])
X, y = ts.preprocessing(df, lagshift=7)
X = ts.add_additional_feat(subset_df, cat, scale_add)
X = X.reshape(X.shape[0], X.shape[2])
X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)
ts.plot_timeseries('sales')


# In[ ]:


me = ModelEvaluation(model)


# In[ ]:


me.evaluate_model(X_train, X_test, y_train, y_test)


# In[ ]:


me.plot_evaluation(y_train, y_test)


# ## LSTM

# In[ ]:


early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)


# In[ ]:


K.clear_session()

batch_size = 1
neurons = 4

model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, 1, 215), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, batch_size=1, epochs=20, callbacks=[early_stop], validation_split=0.1)


# In[ ]:


ts.evaluate_model(model, X_train, X_test, y_train, y_test)


# In[ ]:


store_product = df_train.groupby(['sku_key', 'store_key']).sum().reset_index()[['sku_key', 'store_key']]


# In[ ]:


store_product.head()


# In[ ]:


full_sales.head()


# In[ ]:


start = time.time()
length = len(store_product)
count = 0
for i, j in store_product.values:
    count += 1
    subset_df = full_sales[(full_sales['sku_key']==i) & (full_sales['store_key'] == j)]
    subset_df.drop(['sku_key'],axis=1, inplace=True)
    ts = TimeSeriesModelling(subset_df)
    ts.set_date('tran_date')
    df = ts.scale_data(subset_df, scaler=scaler, values_index='sales')
    df = pd.DataFrame(ts.df, columns=['sales'])
    X, y = ts.preprocessing(df, lagshift=1)
    X = ts.add_additional_feat(subset_df, cat, scale_add)
    X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)
    model.fit(X_train, y_train, verbose=0, batch_size=1, epochs=40, callbacks=[early_stop], validation_split=0.1)
    if (count % 5) == 0:
        print('We have reached {}"%"'.format(count/length*100))
        print('So far we have taken {} minutes'.format((time.time()-start)/60))
end = time.time()
model.save('model_full_feat_allscale.h5')  # creates a HDF5 file 'my_model.h5'


# In[ ]:


#model.save('model_full_feat_allscale.h5')  # creates a HDF5 file 'my_model.h5'


# Load old models

# In[ ]:


#from keras.models import load_model

# returns a compiled model
# identical to the previous one
#model = load_model('model_full_feat.h5')


# # Blind predict

# In[ ]:


listof


# In[ ]:


subset_df = full_sales[full_sales['sku_key']==48676]


# In[ ]:


subset_df


# In[ ]:


ts = TimeSeriesModelling(subset_df)
ts.set_date('tran_date')


# Split into training

# In[ ]:


train = subset_df.iloc[:-90]
test = subset_df.iloc[-90:]


# In[ ]:


train


# In[ ]:


# BLIND 90 Days
import datetime
predicted = []

for i in range(90):
    df = ts.scale_data(train, scaler=scaler, values_index='sales')
    df = pd.DataFrame(df, columns=['sales'])
    X, y = ts.preprocessing(df, lagshift=7)
    X = ts.add_additional_feat(train, cat, scale_add)
    X = X.reshape(X.shape[0], X.shape[2])
    X_train, X_test, y_train, y_test = ts.train_test_split(X, y, 0.8)
    predict = model.predict(X_train[-1,:].reshape(1,X_train.shape[-1]))    
    
    predicted.append(float(scaler.inverse_transform(predict).reshape(-1)))
    
    changes = train.iloc[-1,:].copy()
    #This is where the add features go
    #Dates
    next_date = pd.to_datetime(train.index[-1] + datetime.timedelta(days=1))
    last_day = next_date.day
    last_week = next_date.week
    last_month = next_date.month
    last_year = next_date.year
    last_dayname = next_date.weekday_name
    
    if last_day in [25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5]:
        changes.loc['monthend_Not-Monthend'] = 0
        changes.loc['monthend_Monthend'] = 1
    else:
        changes.loc['monthend_Not-Monthend'] = 1
        changes.loc['monthend_Monthend'] = 0
    
    if last_dayname in ['Saturday', 'Sunday', 'Friday']:
        changes.loc['time_of_week_Weekday'] = 0
        changes.loc['time_of_week_Weekend'] = 1
    else:
        changes.loc['time_of_week_Weekday'] = 1
        changes.loc['time_of_week_Weekend'] = 0

    for i in range(1, 53):
        changes['week_{}'.format(i)] = 0
        if i == last_week:
            changes['week_{}'.format(i)] = 1
            
    for i in range(1, 13):
        changes['month_{}'.format(i)] = 0
        if i == last_week:
            changes['month_{}'.format(i)] = 1
    
    changes.loc['sales'] = float(scaler.inverse_transform(predict).reshape(-1))
    
    train = pd.concat([train, pd.DataFrame([changes], index=[next_date])])
    
    #Aggregates
    for i in range(2,12):
        train.loc[next_date,'agg{}'.format(i)] = train.iloc[-i:,1].rolling(i).mean()[-1]
    train.dropna(inplace=True)
    
    for i in range(10,11):
        train.loc[next_date,'max{}'.format(i)] = train.iloc[-i:,1].rolling(i).max()[-1]
    train.dropna(inplace=True)
    
    for i in range(10,1):
        train.loc[next_date,'min{}'.format(i)] = train.iloc[-i:,1].rolling(i).min()[-1]
    train.dropna(inplace=True)
    
    
    #Nuls
    notnuls = []
    
    nul_counter = 0
    nulidx = -2
    ntick = True
    while ntick == True:
        if train.iloc[nulidx, 2] == 0:
            nul_counter += 1
            nulidx -= 1
        else:
            ntick = False
            
    nnul_counter = 0
    nnulidx = -2
    tick = True
    while tick == True:
        if train.iloc[nnulidx, 2] != 0:
            nnul_counter += 1
            nnulidx -= 1
        else:
            tick = False
    
    train.loc[next_date, 'nuls'] = nul_counter
    train.loc[next_date, 'notnuls'] = nnul_counter
    
    


# In[ ]:


train.reset_index(inplace=True)
train['index'] = pd.to_datetime(train['index'])
train.set_index('index', inplace=True)

test.reset_index(inplace=True)
test['tran_date'] = pd.to_datetime(test['tran_date'])
test.set_index('tran_date', inplace=True)


# In[ ]:


train.tail()


# In[ ]:


train.columns[0:]


# In[ ]:


train['sales'] = train['sales'].astype('float')


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))

train['sales'].plot(ax=ax)
test['sales'].plot(ax=ax, color='r')
ax.set_title('55991 Sales')
#ax.set_xlim(right=pd.datetime(2018,1,1))
#ax.set_ylim(bottom=-5, top=10)


# One of the first things that we saw in this model it the model going rogue. I.e., it just goes off fully negative or fully positive. 
# What if we combine some sort of simulation into the model, which can pick lets say a zero for the predicted value with the same probability as its ocurrance in the dataset? This will bring the model back to the values most common in the dataset. 
# This could be combined with a general Monte Carlo approach, in which every prediction can be updated to a monte carlo version, or similar.
# 
# Other features maybe to add which could account for the going rogue, adding a rolling min and rolling max for a number of time periods.

# In[ ]:


plot_df = pd.DataFrame(train).join(test, lsuffix='train', rsuffix='test')


# In[ ]:


plot_df.head()


# In[ ]:


plot_df = plot_df[['salestrain', 'salestest']]
plot_df.columns = ['pred', 'true']


# In[ ]:


threshold = (1-plot_df['true'].value_counts()[0]/len(plot_df))
print('Parameterization threshold is {}'.format(threshold))


# In[ ]:


plot_df['pred'] = plot_df['pred'].apply(parameterize_output, args=[threshold])
pred = plot_df['pred'].apply(parameterize_output, args=[threshold]).values


# In[ ]:


plot_df.reset_index(inplace=True)
plot_df['index'] = pd.to_datetime(plot_df['index'])
plot_df.set_index('index', inplace=True)


# In[ ]:


plot_df.head()


# In[ ]:


f, ax = plt.subplots(figsize=(18,8),)
plot_df.plot(ax=ax, color=['darkviolet', 'orange'], linewidth=2.5)
ax.set_title('True and predicted sales', fontdict=font)
ax.set_xlabel('Date', fontdict=font)
ax.set_ylabel('Sales', fontdict=font)

ax.axvline('2017-10-27', color='black')
plt.tick_params(labelsize=14)

plt.show()
f.savefig('55991_unsmoothed_new_14days.png')


# In[ ]:


plot_df.index[-90].date()


# In[ ]:


test_df = plot_df[plot_df.index.date >= plot_df.index[-90].date()]


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(test_df.iloc[:,1], test_df.iloc[:,0]))
mse = metrics.mean_squared_error(test_df.iloc[:,1], test_df.iloc[:,0])
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)


# In[ ]:


test_df.iloc[:,0].sum()


# In[ ]:


rmse = np.sqrt(metrics.mean_squared_error(test_df.iloc[1:,1], test_df.iloc[:-1,1]))
mse = metrics.mean_squared_error(test_df.iloc[1:,1], test_df.iloc[:-1,1])
print('RMSE: %.3f' % rmse)
print('MSE: %.3f' % mse)


# In[ ]:


test_df.iloc[:,1].sum()

