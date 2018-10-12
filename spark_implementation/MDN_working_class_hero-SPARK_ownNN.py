
# coding: utf-8

# In[1]:


from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import col,lag
from pyspark.sql.types import IntegerType, DateType

# create entry points to spark
try:
    sc.stop()
except:
    pass
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
sc=SparkContext()
spark = SparkSession(sparkContext=sc)

results = query_job.result()
df = results.to_dataframe()

df = spark.read.csv('sampled_ts_train.csv',  header=True)

df = df.withColumn('tran_date', df['tran_date'].cast(DateType()))


def date_features(df,timestamp_column):
  
    import pyspark.sql.functions as F
  
    df = df.withColumn('date', F.to_date(F.col(timestamp_column), 'yyyy-mm-dd'))            .withColumn('day_of_week',F.date_format(F.col(timestamp_column),'u'))           .withColumn('month',F.month('date'))           .withColumn('day_of_month',F.dayofmonth('date'))           .withColumn('week_of_year',F.weekofyear('date'))
   
    df = df.withColumn('weekend',F.when(F.col('day_of_week') ==1,'weekend')                                  .when(F.col('day_of_week') ==6,'weekend')                                  .when(F.col('day_of_week') ==7,'weekend')                                  .otherwise('Weekday'))
 
    df = df.withColumn('day',F.when(F.col('day_of_week') == 7,'Saturday')                              .when(F.col('day_of_week') == 2,'Monday')                              .when(F.col('day_of_week') == 3,'Tuesday')                              .when(F.col('day_of_week') == 4,'Wednesday')                              .when(F.col('day_of_week') == 5,'Thursday')                              .when(F.col('day_of_week') == 6,'Friday')                              .otherwise('Sunday'))
    df = df.withColumn('month_end', F.when(F.col('day_of_month') == 25,'month_end')                                     .when(F.col('day_of_month') == 26,'month_end')                                     .when(F.col('day_of_month') == 27,'month_end')                                     .when(F.col('day_of_month') == 28,'month_end')                                     .when(F.col('day_of_month') == 29,'month_end')                                     .when(F.col('day_of_month') == 30,'month_end')                                     .when(F.col('day_of_month') == 31,'month_end')                                     .when(F.col('day_of_month') == 1,'month_end')                                     .when(F.col('day_of_month') == 2,'month_end')                                     .when(F.col('day_of_month') == 3,'month_end')                                     .when(F.col('day_of_month') == 4,'month_end')                                     .when(F.col('day_of_month') == 5,'month_end')                                     .otherwise('not_month_end'))

    df = df.withColumn('christmas', F.when(((F.col('month') == 12) & (F.col('day_of_month') == 20)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 21)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 22)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 23)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 24)),'christmas')                                     .when(((F.col('month') == 12) & (F.col('day_of_month') == 25)),'christmas')                                     .otherwise('not_christmas'))

  

    return df


df = date_features(df,'tran_date')

# Make sure all the dates are represented


for i in engineered:
    print(i, df.select(i).toPandas()[i].unique())


# You are shifting the sales by one day. thus transferring some of the data of one product into the next product. 
# You will thus have to split the dataframes into individual 10 000 frames, and then perform the processing on each of the data frames. 


w_1 = Window().partitionBy().orderBy(col('date'))
for i in range(2,8):
    df = df.withColumn('sales_lag_{}'.format(i),lag('sales',count=i).over(w_1).cast(IntegerType()))
    
df = df.withColumn('sales', df['sales'].cast(IntegerType()))
df = df.withColumn('avg_discount', df['avg_discount'].cast(DoubleType()))
df = df.withColumn('selling_price', df['selling_price'].cast(DoubleType()))

df = df.drop('tran_date')


df.createOrReplaceTempView("sales_table")
spark.catalog.listTables()

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


from pyspark.ml.feature import OneHotEncoder, StringIndexer

df = df.na.drop()

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
   

for i in new_cats:
    encoder = OneHotEncoder(inputCol=i, outputCol='{}_vector'.format(i))
    indexed_df = encoder.transform(indexed_df)
    indexed_df = indexed_df.drop(i)



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

#ALTERNATIVE METHOD

#input_data = indexed_df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feats, outputCol="features")

final_df = assembler.transform(indexed_df)

final_df.show(1)


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer, encoder, vector_indexer, assembler])
model = pipeline.fit(df)
transformed = model.transform(df)


final_df = final_df.select('sales', 'features')


from pyspark.ml.feature import StandardScaler


standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

scaler = standardScaler.fit(final_df)
scaled_df = scaler.transform(final_df)
scaled_df = scaled_df.drop('features')
scaled_df.take(2)

train_data, test_data = scaled_df.randomSplit([.8,.2])



from pyspark.ml.feature import OneHotEncoder
import keras
from keras import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


K.clear_session()
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

model = Sequential()
model.add(Dense(32, input_shape=(239,), activation='tanh'))
model.add(Dense(1))

opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)


model.summary()


from elephas.utils.rdd_utils import to_simple_rdd


rdd = train_data.rdd

from elephas.spark_model import SparkModel
from elephas.optimizers import Adam


spark_model = SparkModel(model, frequency='epoch', mode='synchronous', num_workers=4, elephas_optimizer=Adam())
spark_model.fit(rdd, epochs=20, batch_size=500, verbose=1, validation_split=0.1)


spark_model.fit()
