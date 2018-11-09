# Preprocessing time series data
import pandas as pd
import numpy as np
from tsfresh import extract_features

df = pd.read_csv('complete_df_7.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df['stock_open'] = df['stock_open'].astype(float)
# Create aggregate of sales down to product level
aggregate = df.groupby(['sku_key', 'tran_date']).agg({'sales':'sum',
                                                      'selling_price':'mean',
                                                      'avg_discount': 'mean',
                                                      'stock_open': 'sum'})
aggregate.reset_index(inplace=True)

# Create categorical to join to aggregates
categorical = df[['sku_key', 'sku_department', 'sku_subdepartment',
                  'sku_category', 'sku_subcategory', 'sku_label']]
nw_df = pd.DataFrame([], columns=['sku_key', 'sku_department',
                                  'sku_subdepartment', 'sku_category',
                                  'sku_subcategory', 'sku_label'])
for i in categorical['sku_key'].unique():
    cats = pd.DataFrame(categorical[categorical['sku_key'] == i].iloc[0]).T
    nw_df = pd.concat([nw_df, cats])

# Join categoricals and aggregates and write sku labels/joint table to csv
nw_df.reset_index(inplace=True, drop=True)
nw_df.to_csv('sku_labels.csv', index=False)
aggregate['sku_key'] = aggregate['sku_key'].astype(int)
nw_df['sku_key'] = nw_df['sku_key'].astype(int)
aggregate_df = aggregate.merge(nw_df, how='left', on='sku_key')
aggregate_df.to_csv('aggregate_products.csv', index=False)

# Extract features from TS using tsfresh and write
aggregate_df['tran_date'] = pd.to_datetime(df['tran_date'])
extracted_features = extract_features(aggregate_df[['sku_key',
                                                    'tran_date',
                                                    'sales']],
                                      column_id="sku_key",
                                      column_sort="tran_date")
extracted_features.to_csv('extracted_features.csv')
