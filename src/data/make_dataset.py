# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../../data/complete_df_7.csv')
    if df.columns[0] == 'Unnamed: 0':
        df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'stock_open' in df.columns:
        df['stock_open'] = df['stock_open'].astype(float)
    #aggregate to the product level across stores
    aggregate = df.groupby(['sku_key', 'tran_date'])\
    .agg({'sales':'sum', 'selling_price':'mean',
          'avg_discount': 'mean', 'stock_open': 'mean'})
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

if __name__ == '__main__':
    main()
