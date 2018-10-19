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

dimred = TSNE(2)
tsnes = dimred.fit_transform(X)

#Merge tsne coordinates onto original df with sku_keys
plot_df = pd.DataFrame(tsnes).join(df.reset_index())

#Merge above tsne and features table to sku_key and categories
plot_df['sku_key'] = plot_df['id'].astype(int)
agg['sku_key'] = agg['sku_key'].astype(int)
plot_df = plot_df.merge(agg, how='left', on='sku_key')

colors=['b', 'r', 'g', 'y', 'm', 'orange', 'gold', 'skyblue']

def plot_by_factor(df, factor, colors, showplot=False):
    ''' Plot by factor on a already constructed
    t-SNE plot.
    '''
    listof = {}     # this gets numbers to get the colors right
    for i, j in enumerate(plot_df[factor].unique()):
        listof[j] = i
    plot_df[factor] = plot_df[factor].map(listof)

    f, ax = plt.subplots(figsize=(15,12))
    for i in df[factor].unique():
        ax.scatter(df[df[factor] == i][0],
                   df[df[factor] == i][1],
                   color=colors[i], label=i)
    ax.legend()
    ax.set_title('t-SNE colored by {}', factor)

    if showplot == True:
        plt.show()
    else:
        f.savefig('images/{}.png'.format(factor))

plot_by_factor(plot_df, 'sku_department', colors)


#This is where the clusters come into play. here you have to read in
#the files which contain clusters from the different methods and
#assess them.

class AnalyzeClusters(object):
    def __init__(self):
        pass

    def make_dataset(self, sales_df, clus_df):
        sales_df['sku_key'] = sales_df['sku_key'].astype(int)
        self.c_dfs = {}
        for i in clus_df['cluster'].unique():
            self.c_dfs['cluster_{}'.format(i)] = clus_df[clus_df['cluster'] == i]

        for i in self.c_dfs.keys():
            self.c_dfs[i] = self.c_dfs[i].merge(sales_df, on='sku_key')

        return self.c_dfs


    def plot_cluster_continuous(self, cluster_dfs, categories, showplot=False):
        for j in categories:
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
                if showplot == True:
                    plt.show()
                else:
                    f.savefig('images/{}.png'.format('{}-{}'.format(i, j)))

run_cats = ['sales', 'selling_price', 'avg_discount']
product_sales = pd.read_csv('aggregate_products.csv')
clusters = pd.read_csv('tsne_clusters.csv')

analyze = AnalyzeClusters()
df_dict = analyze.make_dataset(product_sales, clusters)

analyze.plot_cluster_continuous(df_dict, run_cats)
