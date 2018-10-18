import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from dtaidistance import dtw, dtw_visualisation, clustering
from dtaidistance import dtw_visualisation as dtwvis



agg = pd.read_csv('sku_labels.csv')
df = pd.read_csv('extracted_features.csv')
df.dropna(axis=1, inplace=True)

scale = MinMaxScaler()
skus = df['id']
df.set_index('id', inplace=True)
X = scale.fit_transform(df)

names = df.columns


product_sales = pd.read_csv('aggregate_products.csv')

product_sales['sku_key'] = product_sales['sku_key'].astype(int)
product_sales.drop(['sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory'], axis=1, inplace=True)


product_sales.head()

product_ts = pd.pivot_table(product_sales, values='sales', index='sku_key', columns='tran_date')

product_ts['nas'] = product_ts.apply(lambda x: x.isna()).sum(axis=1)
print('There are {} products with less than 50% entries'.format(len(product_ts[product_ts['nas'] > len(product_ts.columns)/2])))

product_ts = product_ts.sort_values('nas', ascending=True).drop('nas', axis=1)

plt.figure(figsize=(5,10))
plt.imshow(product_ts, cmap='hot', interpolation='nearest')
plt.show()

product_ts_fill = product_ts.fillna(0)

product_ts.head()

product_matrix_fill = product_ts_fill.as_matrix()
product_matrix = product_ts.as_matrix()

product_dict = {}
product_list = []

for i, j in zip(range(len(product_matrix)), product_ts.index):
    product_dict[j] = product_matrix[i][~np.isnan(product_matrix[i])]
    product_list.append(product_matrix[i][~np.isnan(product_matrix[i])])

subsample = 200
product_matrix_fill = product_matrix_fill[:subsample]

ds = dtw.distance_matrix_fast(product_matrix_fill)

f, ax = dtw_visualisation.plot_matrix(ds)
f.set_size_inches(12, 12)


model = clustering.LinkageTree(dtw.distance_matrix_fast, {})

clusters_dtw = model.fit(product_matrix_fill)

f, ax = model.plot()
f.set_size_inches(17, 20)



clusters = fcluster(model.linkage, 1.154)
np.unique(clusters)

fig = plt.figure(figsize=(20, 20))
dendrogram(model.linkage, orientation='left', leaf_font_size=15, color_threshold=100, labels=product_ts.index[:subsample])
plt.show()

dtw_df = new_df[:subsample]
dtw_df['cluster'] = clusters

cluster_dfs_dtw = {}
for i in dtw_df['cluster'].unique():
    cluster_dfs_dtw['cluster_{}'.format(i)] = dtw_df[dtw_df['cluster'] == i]

for i in cluster_dfs_dtw.keys():
    cluster_dfs_dtw[i] = cluster_dfs_dtw[i].merge(product_sales, on='sku_key')

for j in ['sales', 'selling_price', 'avg_discount']:
    print('\n\n', j)
    for i in cluster_dfs_dtw.keys():
        print(i)
        print('There are {} skus in this cluster'.format(len(cluster_dfs_dtw[i]['sku_key'].unique())))
        plt.figure(figsize=(15,3))
        cluster_dfs_dtw[i][j].plot(kind='hist', bins=20, logy=True)
        plt.title(j)
        if j == 'sales':
            plt.xlim(-50, 800)
        elif j == 'selling_price':
            plt.xlim(-100, 8000)
        elif j == 'avg_discount':
            plt.xlim(-1500, 2000)
        
        plt.show()
