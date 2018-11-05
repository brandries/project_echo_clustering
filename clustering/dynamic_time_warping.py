import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from dtaidistance import dtw, dtw_visualisation, clustering
from dtaidistance import dtw_visualisation as dtwvis
from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle
from dimred_clustering import DataPreprocess
run_plots = False

class Preprocessing(object):
    def __init__(self):
        pass

    def pivot_table(self, df):
        print('Manipulate and pivot table...')
        df['sku_key'] = df['sku_key'].astype(int)
        df.drop(['sku_department', 'sku_subdepartment',
                 'sku_category', 'sku_subcategory', 'sku_label'],
                axis=1, inplace=True)
        product_ts = pd.pivot_table(df, values='sales',
                                    index='sku_key', columns='tran_date')
        return product_ts

    def sort_nas(self, df):
        df['nas'] = df.apply(lambda x: x.isna()).sum(axis=1)
        print('There are {} products with less than 50% entries'\
        .format(len(df[df['nas'] > len(df.columns)/2])))
        self.df_ordered = df.sort_values('nas', ascending=True).drop('nas', axis=1)
        return self.df_ordered

    def plot_nas(self, df):
        plt.figure(figsize=(5,10))
        plt.imshow(df, cmap='hot', interpolation='nearest')
        plt.show()

    def make_diff_length_list(self, df):
        self.product_ts_fill = df.fillna(0)
        self.product_matrix_fill = self.product_ts_fill.values
        self.product_matrix = df.values
        product_dict = {}
        product_list = []
        for i, j in zip(range(len(self.product_matrix)), df.index):
            product_dict[j] = self.product_matrix[i][~np.isnan(self.product_matrix[i])]
            product_list.append(self.product_matrix[i][~np.isnan(self.product_matrix[i])])

        self.product_dict = product_dict
        self.product_list = product_list


class DynamicTimeWarping(object):
    def __init__(self):
        pass

    def distance_matrix(self, df):
        print('Producing distance matrix...')
        ds = dtw.distance_matrix_fast(df)
        if run_plots == True:
            f, ax = dtw_visualisation.plot_matrix(ds)
            f.set_size_inches(12, 12)
        return ds

    def linkage_tree(self, df):
        print('Producing linkage Tree')
        self.model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        clusters_dtw = self.model.fit(df)
        return clusters_dtw
        pickle.dump(self.model, open('model.pkl', 'wb'))
        if run_plots == True:
            f, ax = self.model.plot()
            f.set_size_inches(17, 20)

    def cluster(self, model, cluster_nr):
        threshold = cluster_nr
        clusters = fcluster(model.linkage, threshold,
                            criterion='inconsistent', depth=10)
        return clusters

        if run_plots == True:
            fig = plt.figure(figsize=(20, 20))
            dendrogram(model.linkage, orientation='left', leaf_font_size=15,
                       color_threshold=100, labels=product_ts.index[:subsample])
            plt.show()




def main():
    dp = DataPreprocess()
    labels, df = dp.read_data('sku_labels.csv', 'extracted_features.csv')
    product_sales = pd.read_csv('aggregate_products.csv')
    scaler = RobustScaler()
    X = dp.scale_data(df, scaler)
    names = df.columns

    pp = Preprocessing()
    product_ts = pp.pivot_table(product_sales)
    product_ts = pp.sort_nas(product_ts)
    product_ts.to_csv('pivot_products.csv')
    if run_plots == True:
        pp.plot_nas(product_ts)
    pp.make_diff_length_list(product_ts)
    dtw = DynamicTimeWarping()
    clusters_dtw = dtw.linkage_tree(pp.product_matrix_fill)
    clusters = dtw.cluster(dtw.model, 6)
    dtw_df = product_ts.reset_index()
    dtw_df['cluster'] = clusters
    output_df = dtw_df[['sku_key', 'cluster']]
    print('Outputting...')
    output_df.to_csv('dtw_clusters.csv', index=False)

if __name__ == '__main__':
    main()
