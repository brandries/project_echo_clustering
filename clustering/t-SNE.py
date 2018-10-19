# Clustering of timeseries data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

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
dimred = TSNE()
plot_df = dimred.fit_transform(X)
plot_df = pd.DataFrame(plot_df).join(df.reset_index())

print('Clusting...')
kmeans = AgglomerativeClustering(n_clusters=8)
clusters_fit = kmeans.fit_predict(plot_df[[0,1]])
tsne_cluster = plot_df.join(pd.DataFrame(clusters_fit), rsuffix='clus')
<<<<<<< HEAD
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

tsne_cluster.rename(columns={'0':'tsne1', 1:'tsne2', '0clus':'cluster'},
                    inplace=True)

print('Outputting...')
out_df = tsne_cluster[['id', 'cluster']]
out_df.to_csv('tsne_clusters.csv', index=False)
