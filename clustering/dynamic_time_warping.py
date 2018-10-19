import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from dtaidistance import dtw, dtw_visualisation, clustering
from dtaidistance import dtw_visualisation as dtwvis
from sklearn.preprocessing import MinMaxScaler

run_plots = False

print('Reading in the data...')
agg = pd.read_csv('sku_labels.csv')
df = pd.read_csv('extracted_features.csv')
product_sales = pd.read_csv('aggregate_products.csv')
df.dropna(axis=1, inplace=True)

scale = MinMaxScaler()
skus = df['id']
df.set_index('id', inplace=True)
X = scale.fit_transform(df)

names = df.columns

print('Manipulate and pivot table...')
product_sales['sku_key'] = product_sales['sku_key'].astype(int)
product_sales.drop(['sku_department', 'sku_subdepartment', 'sku_category', 'sku_subcategory'], axis=1, inplace=True)
product_ts = pd.pivot_table(product_sales, values='sales', index='sku_key', columns='tran_date')

product_ts['nas'] = product_ts.apply(lambda x: x.isna()).sum(axis=1)
print('There are {} products with less than 50% entries'.format(len(product_ts[product_ts['nas'] > len(product_ts.columns)/2])))

product_ts = product_ts.sort_values('nas', ascending=True).drop('nas', axis=1)

def plot_nas(df):
    plt.figure(figsize=(5,10))
    plt.imshow(df, cmap='hot', interpolation='nearest')
    plt.show()

if run_plots == True:
    plot_nas(product_ts)

product_ts_fill = product_ts.fillna(0)
product_matrix_fill = product_ts_fill.values
product_matrix = product_ts.values

product_dict = {}
product_list = []

for i, j in zip(range(len(product_matrix)), product_ts.index):
    product_dict[j] = product_matrix[i][~np.isnan(product_matrix[i])]
    product_list.append(product_matrix[i][~np.isnan(product_matrix[i])])

subsample = 200
product_matrix_fill = product_matrix_fill[:subsample]

print('Produce distance matrix...')
ds = dtw.distance_matrix_fast(product_matrix_fill)
if run_plots == True:
    f, ax = dtw_visualisation.plot_matrix(ds)
    f.set_size_inches(12, 12)

model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
clusters_dtw = model.fit(product_matrix_fill)

if run_plots == True:
    f, ax = model.plot()
    f.set_size_inches(17, 20)

clusters = fcluster(model.linkage, 1.154)
if run_plots == True:
    fig = plt.figure(figsize=(20, 20))
    dendrogram(model.linkage, orientation='left', leaf_font_size=15, color_threshold=100, labels=product_ts.index[:subsample])
    plt.show()

dtw_df = product_ts.reset_index()[:subsample]
dtw_df['cluster'] = clusters

output_df = dtw_df[['sku_key', 'cluster']]

print('Outputting...')
output_df.to_csv('dtw_clusters.csv', index=False)
