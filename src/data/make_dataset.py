# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_features
from dynamic_time_warping import Preprocessing

def main():
    df = pd.read_csv('complete_df_7.csv')
    if df.columns[0] == 'Unnamed: 0':
        df.drop('Unnamed: 0', axis=1, inplace=True)
    df['stock_open'] = df['stock_open'].astype(float)
    #aggregate to the product level across stores
    aggregate = df.groupby(['sku_key', 'tran_date'])\
    .agg({'sales':'sum', 'selling_price':'mean',
          'avg_discount': 'mean', 'stock_open': 'sum'})
    aggregate.reset_index(inplace=True)

    #Get the categorical variables for each product
    categorical = df[['sku_key', 'sku_department', 'sku_subdepartment',
                  'sku_category', 'sku_subcategory', 'sku_label']]
    nw_df = pd.DataFrame([], columns=['sku_key','sku_department',
                                      'sku_subdepartment','sku_category',
                                      'sku_subcategory', 'sku_label'])
    for i in categorical['sku_key'].unique():
        nw_df = pd.concat([nw_df, pd.DataFrame(categorical[categorical['sku_key'] == i].iloc[0]).T])
    nw_df.reset_index(inplace=True, drop=True)
    nw_df.to_csv('sku_labels.csv', index=False)

    #Join aggregate product to the categorical variables
    aggregate['sku_key'] = aggregate['sku_key'].astype(int)
    nw_df['sku_key'] = nw_df['sku_key'].astype(int)
    aggregate_df = aggregate.merge(nw_df, how='left', on='sku_key')
    aggregate_df.to_csv('aggregate_products.csv', index=False)

    #Extract features from the time series using tsfresh
    aggregate_df['tran_date'] = pd.to_datetime(df['tran_date'])
    extracted_features = extract_features(aggregate_df[['sku_key', 'tran_date',
                                                        'sales']],
                                          column_id="sku_key",
                                          column_sort="tran_date")
    extracted_features.to_csv('extracted_features.csv')

    #Pivot Table and save dfs with and without nans
    pp = Preprocessing()
    pivot_table = pp.pivot_table(aggregate_df)
    sorted_df = pp.sort_nas(pivot_table)
    pp.make_diff_length_list(sorted_df)
    pivot_nans, nans, pivot_no_nans, no_nans = pp.split_nans(pivot_table,
                                                             extracted_features)

    pivot_table.to_csv('pivot_table.csv')
    pivot_nans.to_csv('nans_pivot_table.csv')
    pivot_no_nans.to_csv('no_nans_pivot_table.csv')
    nans.to_csv('extracted_feat_nans.csv')
    no_nans.to_csv('extracted_feat_no_nans.csv')

if __name__ == '__main__':
    main()
