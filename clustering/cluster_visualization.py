# Pull in tsne or pca to visualize. 

# Get more visualization after looking at continuous variable distributions


# Visualize members of clusters

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


