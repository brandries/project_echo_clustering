# Clustering of timeseries data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

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

    def run_tsne(self, features):
        from sklearn.manifold import TSNE
        print('Running Dimentionality Reduction...')
        dimred = TSNE()
        tsne = dimred.fit_transform(features)
        return tsne


class Clustering(object):
    def __init__(self):
        pass

    def cluster(self, dimred, clustering_algo):
        print('Clusting...')
        clusters_fit = clustering_algo.fit_predict(dimred[[0,1]])
        tsne_cluster = plot_df.join(pd.DataFrame(clusters_fit), rsuffix='clus')
        return tsne_cluster


def main():
    dp = DataPreprocess()
    labels, df = dp.read_data('sku_labels.csv', 'extracted_features.csv')
    scaler = RobustScaler()
    scaled = dp.scale_data(df, scaler)
    names = df.columns

    dr = DimensionalityReduction()
    tsne = dr.run_tsne(scaled)
    plot_df = pd.DataFrame(tsne).join(df.reset_index())
    cl = Clusting()
    clus_algo = AgglomerativeClustering(n_clusters=6)
    plot_df = cl.cluster(plot_df, clus_algo)
    plot_df.rename(columns={'0':'tsne1', 1:'tsne2', '0clus':'cluster'},
                   inplace=True)

    print('Outputting...')
    out_df = plot_df[['id', 'cluster']]
    out_df.to_csv('tsne_clusters.csv', index=False)

if __name__ == '__main__':
    main()
