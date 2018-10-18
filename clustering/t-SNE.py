# Clustering of timeseries data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


agg = pd.read_csv('sku_labels.csv')
df = pd.read_csv('extracted_features.csv')
df.dropna(axis=1, inplace=True)

scale = MinMaxScaler()
skus = df['id']
df.set_index('id', inplace=True)
X = scale.fit_transform(df)

names = df.columns

dimred = TSNE()
plot_df = dimred.fit_transform(X)
plot_df = pd.DataFrame(plot_df).join(df.reset_index())

plot_df['sku_key'] = plot_df['id'].astype(int)
agg['sku_key'] = agg['sku_key'].astype(int)

plot_df = plot_df.merge(agg, how='left', on='sku_key')

listof = {}
for i, j in enumerate(plot_df['sku_department'].unique()):
    listof[j] = i

plot_df['sku_department'] = plot_df['sku_department'].map(listof)

f, ax = plt.subplots(figsize=(15,12))

colors=['b', 'r', 'g', 'y', 'm', 'orange', 'gold', 'skyblue']

for i in plot_df['sku_department'].unique():
    ax.scatter(plot_df[plot_df['sku_department'] == i][0], plot_df[plot_df['sku_department'] == i][1],
    color=colors[i], label=i)           
    
ax.legend()
ax.set_title('t-SNE colored by department')
plt.show()


kmeans = AgglomerativeClustering(n_clusters=8)

clusters_fit = kmeans.fit_predict(plot_df[[0,1]])

tsne_cluster = plot_df.join(pd.DataFrame(clusters_fit), rsuffix='clus')
tsne_cluster.rename(columns={'0':'tsne1', 1:'tsne2', '0clus':'cluster'}, inplace=True)
tsne_cluster.head()

f, ax = plt.subplots(figsize=(15,12))

colors=['darkblue', 'darkorange', 'purple', 'darkgreen', 'gold', 'darkred', 'black', 'lime']

for i in tsne_cluster['cluster'].unique():
    ax.scatter(tsne_cluster[tsne_cluster['cluster'] == i]['tsne1'], tsne_cluster[tsne_cluster['cluster'] == i]['tsne2'],
    color=colors[i], label=i)           
    
ax.legend()
ax.set_title('k-Means clusters on t-SNE')
plt.show()


product_sales = pd.read_csv('aggregate_products.csv')

product_sales['sku_key'] = product_sales['sku_key'].astype(int)
product_sales.drop(['sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory'], axis=1, inplace=True)

cluster_dfs = {}
for i in tsne_cluster['cluster'].unique():
    cluster_dfs['cluster_{}'.format(i)] = tsne_cluster[tsne_cluster['cluster'] == i]

for i in cluster_dfs.keys():
    cluster_dfs[i] = cluster_dfs[i].merge(product_sales, on='sku_key')

cluster_dfs['cluster_0'].head()

for j in ['sales', 'selling_price', 'avg_discount']:
    print('\n\n', j)
    for i in cluster_dfs.keys():
        print(i)
        plt.figure(figsize=(15,3))
        cluster_dfs[i][j].plot(kind='hist', bins=20, logy=True)
        plt.title(j)
        if j == 'sales':
            plt.xlim(-50, 800)
        elif j == 'selling_price':
            plt.xlim(-100, 8000)
        elif j == 'avg_discount':
            plt.xlim(-1500, 2000)
        
        plt.show()


