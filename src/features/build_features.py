# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tsfresh import extract_features
from dynamic_time_warping import Preprocessing

def main():
    aggregate_df = pd.read_csv('aggregate_products.csv')
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
