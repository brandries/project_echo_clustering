
from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import col,lag
import time



# must do before importing pyplot or pylab
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

def show(fig):

    image = StringIO.StringIO()

    fig.savefig(image, format='svg')

    image.seek(0)

    print("%html <div style='width:1200px'>"+ image.buf +"</div>")



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
font = {'family' : 'sans-serif',
        'size'   : 24}

rc('font', **font)


# create entry points to spark
try:
    sc.stop()
except:
    pass
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
sc=SparkContext()
spark = SparkSession(sparkContext=sc)


df_pd = pd.read_csv('sampled_ts_train.csv', keep_default_na=False)


def make_min_max(df, sales_column, mins=True, maxes=True):
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



start_time = time.time()

from pyspark.sql.functions import *
from pyspark.sql.types import *
B = udf(make_min_max, ArrayType(IntegerType()))
df = df_spark.withColumn('min_max',B(struct([df_spark[sales] for sales in df_spark.columns])))
end_time = time.time()
duration = (end_time - start_time)
print("The process took: ",duration,'s')



start_time = time.time()
df_pd = make_min_max(df_pd,'sales')
end_time = time.time()
duration = (end_time - start_time)
print("The process took: ",duration,'s')



start_time = time.time()

from pyspark.sql.functions import *
from pyspark.sql.types import *
A = udf(periods_w_demand_new, ArrayType(IntegerType()))
df = df_spark.withColumn('data_id',A(struct([df_spark[sales] for sales in df_spark.columns])))
end_time = time.time()
duration = (end_time - start_time)
print("The process took: ",duration,'s')


def periods_w_demand_new(df, sales_column):
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


start_time = time.time()
df_pd = periods_w_demand_new(df_pd,'sales')
end_time = time.time()
duration = (end_time - start_time)
print("The process took: ",duration,'s')



def make_aggregates(df, sales_column, aggregate_range = 10):
       agg_index = []
       for i in range(2, 2 + aggregate_range):
           df.loc[:,'agg{}'.format(i)] = df.loc[:,sales_column].rolling(i).mean()
           agg_index.append('agg{}'.format(i))
       print('Aggregates done')
       return df



start_time = time.time()
df_pd = make_aggregates(df_pd,'sales')
end_time = time.time()
duration = (end_time - start_time)
print("The process took: ",duration,'s')


df_spark = spark.createDataFrame(df_pd)



def date_features(df,timestamp_column):
  
  import pyspark.sql.functions as F
  
  df = df.withColumn('date', F.to_date(F.col(timestamp_column), 'yyyy-mm-dd'))     .withColumn('day_of_week',F.date_format(F.col(timestamp_column),'u'))    .withColumn('month',F.month('date'))    .withColumn('day_of_month',F.dayofmonth('date'))    .withColumn('week_of_year',F.weekofyear('date'))
   
  df = df.withColumn('weekend',F.when(F.col('day_of_week') ==1,'weekend')                    .when(F.col('day_of_week') ==6,'weekend')                    .when(F.col('day_of_week') ==7,'weekend')                    .otherwise('Weekday'))
 
  df = df.withColumn('day',F.when(F.col('day_of_week') == 7,'Saturday')                     .when(F.col('day_of_week') == 2,'Monday')                     .when(F.col('day_of_week') == 3,'Tuesday')                     .when(F.col('day_of_week') == 4,'Wednesday')                     .when(F.col('day_of_week') == 5,'Thursday')                     .when(F.col('day_of_week') == 6,'Friday')                     .otherwise('Sunday'))
  df = df.withColumn('month_end', F.when(F.col('day_of_month') == 25,'month_end')                     .when(F.col('day_of_month') == 26,'month_end')                     .when(F.col('day_of_month') == 27,'month_end')                     .when(F.col('day_of_month') == 28,'month_end')                     .when(F.col('day_of_month') == 29,'month_end')                     .when(F.col('day_of_month') == 30,'month_end')                     .when(F.col('day_of_month') == 31,'month_end')                     .when(F.col('day_of_month') == 1,'month_end')                     .when(F.col('day_of_month') == 2,'month_end')                     .when(F.col('day_of_month') == 3,'month_end')                     .when(F.col('day_of_month') == 4,'month_end')                     .when(F.col('day_of_month') == 5,'month_end')                     .otherwise('not_month_end'))

  df = df.withColumn('christmas', F.when(((F.col('month') == 12) & (F.col('day_of_month') == 20)),'christmas')                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 21)),'christmas')                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 22)),'christmas')                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 23)),'christmas')                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 24)),'christmas')                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 25)),'christmas')                     .otherwise('not_christmas'))

  

  return df


df_spark = date_features(df_spark,'tran_date')


df_spark.select('date','month','day','day_of_week','weekend','day_of_month','week_of_year','month_end','christmas').show()



from pyspark.sql.window import Window



from pyspark.sql.functions import col,lag




def sales_lag_7(df,lag_count=7):
    from pyspark.sql.functions import col,lag
    from pyspark.sql.window import Window
    
    w_1 = Window().partitionBy().orderBy(col('date'))
    df = df.withColumn('sales_lag_7',lag('sales',count=lag_count).over(w_1))
  
    return df


df_spark = sales_lag_7(df_spark,7)



def shift_1(df,shift_count=1):
    from pyspark.sql.functions import col,lag
    from pyspark.sql.window import Window
    
    w_1 = Window().partitionBy().orderBy(col('date'))
    df = df.withColumn('shift_1',lag('sales',count=shift_count).over(w_1))
  
    return df


df_spark = shift_1(df_spark, 1)


df_spark.select("sales", "shift_1","sales_lag_7").show()


# Specifying the schema programmatically
import pyspark.sql.types as typ


df_spark = df_spark.withColumn('store_key', df_spark['store_key'].cast(typ.StringType()))


df_spark = df_spark.withColumn('sku_key', df_spark['sku_key'].cast(typ.StringType()))
df_spark = df_spark.withColumn('nuls', df_spark['nuls'].cast(typ.IntegerType()))
df_spark = df_spark.withColumn('notnuls', df_spark['notnuls'].cast(typ.IntegerType()))
df_spark = df_spark.withColumn('day_of_month', df_spark['day_of_month'].cast(typ.StringType()))
df_spark = df_spark.withColumn('week_of_year', df_spark['week_of_year'].cast(typ.StringType()))



'''#https://stackoverflow.com/questions/46767807/how-to-calculate-rolling-median-in-Pyspark-using-Window

w = (Window.orderBy(col('date').cast('long')).rangeBetween(-2,0))
mean_udf = udf(lambda x: float(np.mean(x)), FloatType())

df_spark.withColumn('list',collect_list('sales').over(w)) \
.withColumn('rolling_mean', mean_udf('list'))'''



def get_dummy(df,categoricalCols,continuousCols,labelCol):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col
    
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in categoricalCols ]
    
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in indexers ]
    
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")
    
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    
    model=pipeline.fit(df)
    data = model.transform(df)
    
    data = data.withColumn('label',col(labelCol))
    
    return data.select('features','label')


df = df_spark.na.drop()



# Deal with categorical data and convert the data to dense vector

catcols = ['store_key', 'sku_key','store_region','store_grading', 
           'sku_department', 'sku_subdepartment', 'sku_category', 
           'sku_subcategory','day_of_week', 'month', 'day_of_month', 
           'week_of_year', 'weekend','day', 'month_end', 'christmas']
num_cols = ['selling_price', 'avg_discount',  'sales_lag_7', 'shift_1'
           'max10','min10', 'nuls', 'notnuls', 'agg2', 'agg3', 'agg4',
            'agg5', 'agg6', 'agg7', 'agg8', 'agg9', 'agg10', 'agg11',]
labelCol = 'sales'



data = get_dummy(df,catcols,num_cols,labelCol)


# ## Process CSV



# https://dzone.com/articles/distributed-deep-learning-with-keras-on-apache-spa

def process_csv(fully_qualified_path, columns_renamed=tuple(),

                excluded_columns=tuple(), num_workers=None):

    if num_workers is None:

        raise NotImplementedError

    excluded_columns = frozenset(excluded_columns)

    data_frame = sqlContext.read.format('com.databricks.spark.csv') 
                           .options(header='true', inferSchema='true') 
                           .load(fully_qualified_path)

    for (old_name, new_name) in columns_renamed:

        data_frame = data_frame.withColumnRenamed(old_name, new_name)

    data_frame = data_frame.repartition(num_workers)

    feature_columns = tuple(frozenset(data_frame.columns) 
                            .difference(excluded_columns))

    transformer = feature.VectorAssembler(inputCols=feature_columns,

                                          outputCol='features')

    data_frame = transformer.transform(data_frame) 
                            .drop(*feature_columns).cache()

    return data_frame


# # Deep Learning


# automatically installs latest version of Keras as dependency

pip install dist-keras

# for GPU clusters, swap out default dependency tensorflow

# with tensorflow for GPU nodes

pip uninstall tensorflow

pip install tensorflow-gpu



from keras import layers, models, optimizers, regularizers, utils

from pyspark.ml import evaluation, feature, tuning

from distkeras import predictors, trainers

from pyspark.sql import functions, types

from pyspark import ml

import numpy as np 

import matplotlib 

from io import StringIO


class DistKeras(ml.Estimator):

    def __init__(self, *args, **kwargs):

        self.__trainer_klass = args[0]

        self.__trainer_params = args[1]

        self.__build_trainer(**kwargs)

        super(DistKeras, self).__init__()

    @classmethod

    def __build_keras_model(klass, *args, **kwargs):

        loss = kwargs['loss']

        metrics = kwargs['metrics']

        layer_dims = kwargs['layer_dims']

        hidden_activation, output_activation = kwargs['activations']

        hidden_init, output_init = kwargs['initializers']

        dropout_rate = kwargs['dropout_rate']

        alpha = kwargs['reg_strength']

        reg_decay = kwargs['reg_decay']

        reg = kwargs['regularizer']

        keras_model = models.Sequential()

        for idx in range(1, len(layer_dims)-1, 1):

            keras_model.add(layers.Dense(layer_dims[idx],

                                         input_dim=layer_dims[idx-1],

                                         bias_initializer=hidden_init,

                                         kernel_initializer=hidden_init,

                                         kernel_regularizer=reg(alpha)))

            keras_model.add(layers.Activation(hidden_activation))

            keras_model.add(layers.Dropout(dropout_rate))

            alpha *= reg_decay

        keras_model.add(layers.Dense(layer_dims[-1],

                                     input_dim=layer_dims[-2],

                                     bias_initializer=output_init,

                                     kernel_initializer=output_init,

                                     kernel_regularizer=reg(alpha)))

        keras_model.add(layers.Activation(output_activation))

        return keras_model

    def __build_trainer(self, *args, **kwargs):

        loss = kwargs['loss']

        learning_rate = kwargs['learning_rate']

        lr_decay = kwargs['lr_decay']

        keras_optimizer = optimizers.SGD(learning_rate, decay=lr_decay)

        keras_model = DistKeras.__build_keras_model(**kwargs)

        self._trainer = self.__trainer_klass(keras_model, keras_optimizer,

                                             loss, **self.__trainer_params)

    def _fit(self, *args, **kwargs):

        data_frame = args[0]

        if len(args) > 1:

            params = args[1]

            self.__build_trainer(**params)

        keras_model = self._trainer.train(data_frame)

        return DistKerasModel(keras_model)

class DistKerasModel(ml.Model):

    def __init__(self, *args, **kwargs):

        self._keras_model = args[0]

        self._predictor = predictors.ModelPredictor(self._keras_model)

        super(DistKerasModel, self).__init__()

    def _transform(self, *args, **kwargs):

        data_frame = args[0]

        pred_col = self._predictor.output_column

        preds = self._predictor.predict(data_frame)

        return preds.withColumn(pred_col,

                                cast_to_double(preds[pred_col]))

cast_to_double = functions.udf(lambda row: float(row[0]), types.DoubleType())


param_grid = tuning.ParamGridBuilder().baseOn(['regularizer', regularizers.l1_l2]) .addGrid('activations', [['tanh', 'relu']]) .addGrid('initializers', [['glorot_normal','glorot_uniform']]) .addGrid('layer_dims', [[input_dim, 2000, 300, 1]]) .addGrid('metrics', [['mae']]) .baseOn(['learning_rate', 1e-2]) .baseOn(['reg_strength', 1e-2]) .baseOn(['reg_decay', 0.25]) .baseOn(['lr_decay', 0.90]) .addGrid('dropout_rate', [0.20, 0.35, 0.50, 0.65, 0.80]) .addGrid('loss', ['mse', 'msle']).build()


estimator = DistKeras(trainers.ADAG,

                      {'batch_size': 256,

                       'communication_window': 3,

                       'num_epoch': 10,

                       'num_workers': 50},

                      **param_grid[0])


evaluator = evaluation.RegressionEvaluator(metricName='r2')

cv_estimator = tuning.CrossValidator(estimator=estimator,

                                     estimatorParamMaps=param_grid,

                                     evaluator=evaluator,

                                     numFolds=5)

cv_model = cv_estimator.fit(df_train)

df_pred_train = cv_model.transform(df_train)

df_pred_test  = cv_model.transform(df_test)

