# Clustering of timeseries data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from VisualizationFunctions import plot_by_factor, AnalyzeClusters

print('Reading in the data...')
agg = pd.read_csv('sku_labels.csv')
df = pd.read_csv('extracted_features.csv')
df.dropna(axis=1, inplace=True)

scale = MinMaxScaler()
skus = df['id']
df.set_index('id', inplace=True)
X = scale.fit_transform(df)
names = df.columns

print('Running Dimentionality Reduction...')
dimred = TSNE(2)
tsnes = dimred.fit_transform(X)

#Merge tsne coordinates onto original df with sku_keys
plot_df = pd.DataFrame(tsnes).join(df.reset_index())

#Merge above tsne and features table to sku_key and categories
plot_df['sku_key'] = plot_df['id'].astype(int)
agg['sku_key'] = agg['sku_key'].astype(int)
plot_df = plot_df.merge(agg, how='left', on='sku_key')

colors=['b', 'r', 'g', 'y', 'm', 'orange', 'gold', 'skyblue']

plot_by_factor(plot_df, 'sku_department', colors)


#This is where the clusters come into play. here you have to read in
#the files which contain clusters from the different methods and
#assess them.

colors = ['darkblue', 'tomato', 'orchid', 'darkorange', 'lime', 'gold', 'dodgerblue', 'pink',
          'grey', 'darkgreen', 'y', 'slateblue', 'r', 'brown']
run_cont = ['sales', 'selling_price', 'avg_discount']
run_cats = ['sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory']

product_sales = pd.read_csv('aggregate_products.csv')
clusters = pd.read_csv('som_clusters.csv')
clusters.columns = ['sku_key', 'cluster']

analyze = AnalyzeClusters()
df_dict = analyze.make_dataset(product_sales, clusters)

analyze.plot_cluster_continuous(df_dict, run_cont)
analyze.plot_cluster_continuous_box(df_dict, run_cont)
analyze.plot_cluster_categorical(df_dict, run_cats)
