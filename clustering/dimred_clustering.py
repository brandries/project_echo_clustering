# Clustering of timeseries data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler
import hdbscan

class DataPreprocess(object):
    def __init__(self):
        pass

    def read_data(self, sku_labels, features):
        print('Reading in the data...')
        self.labels = pd.read_csv(sku_labels)
        self.features = pd.read_csv(features)
        self.features.dropna(axis=1, inplace=True)
        return self.labels, self.features

    def scale_data(self, df, scaler):
        scale = scaler
        skus = df['id']
        df.set_index('id', inplace=True)
        X = scale.fit_transform(df)
        return X


class DimensionalityReduction(object):
    def __init__(self):
        pass

    def run_dimred(self, features, dimred):
        print('Running Dimentionality Reduction...')
        projection = dimred.fit_transform(features)
        return projection


class Clustering(object):
    def __init__(self):
        pass

    def cluster(self, dimred, clustering_algo):
        print('Clustering...')
        clusters_fit = clustering_algo.fit_predict(dimred[[0,1]])
        return clusters_fit


def main():
    dp = DataPreprocess()
    labels, df = dp.read_data('sku_labels.csv', 'extracted_features.csv')
    scaler = RobustScaler()
    scaled = dp.scale_data(df, scaler)
    names = df.columns

    dr = DimensionalityReduction()
    tsne = TSNE()
    tsne = dr.run_dimred(scaled, tsne)
    plot_df = pd.DataFrame(tsne).join(df.reset_index())
    cl = Clustering()
    clus_algo = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters_fit = cl.cluster(plot_df, clus_algo)
    tsne_cluster = plot_df.join(pd.DataFrame(clusters_fit), rsuffix='clus')
    tsne_cluster.rename(columns={'0':'tsne1', 1:'tsne2', '0clus':'cluster'},
                        inplace=True)

    print('Outputting...')
    out_df = tsne_cluster[['id', 'cluster']]
    out_df.to_csv('tsne_clusters.csv', index=False)

if __name__ == '__main__':
    main()
