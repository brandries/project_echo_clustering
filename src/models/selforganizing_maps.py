import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sompy
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView
from dimred_clustering import DataPreprocess
from sklearn.preprocessing import RobustScaler, StandardScaler
from src.models.dynamic_time_warping import Preprocessing

def knn_elbow(df, k_range=20, plot=False):
    from sklearn.cluster import KMeans
    scores = {}
    for i in range(1, k_range+1):
        kmeans = KMeans(n_clusters=i)
        clusters_fit = kmeans.fit_predict(df)
        scores[i] = kmeans.inertia_
    if plot == True:
        pd.DataFrame(scores, index=['score']).T.plot(figsize=(15,8))
        plt.title('Elbow KMeans')
        plt.xlabel('K')
        plt.show()
    return scores

def plot_figures(som):
    #BMU map
    vhts  = BmuHitsView(12,12,"Hits Map",text_size=12)
    #U matrix
    u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True,
                                  text_size=8, show_text=True)
    UMAT  = u.build_u_matrix(sm, distance=1, row_normalized=False)
    #Cluster map
    sm.cluster(6)
    hits  = HitMapView(10,10,"Clustering",text_size=12)
    #Show factor influence
    view2D  = View2D(15,15,"time-series",text_size=10, names=names)
    #Show plots
    view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)
    vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12,
              cmap="Greys", logaritmic=False)
    UMAT = u.show(sm, distance2=1, row_normalized=False, show_data=True,
                  contooor=True, blob=False)
    a = hits.show(sm)
    plt.show()

def make_class_dict(clusters):
    map_dict = {}
    for i, j in enumerate(clusters):
        map_dict[i] = j
    return map_dict

def assign_from_som(clusters, model):
    map_dict = make_class_dict(clusters)
    assignment = pd.DataFrame(model._bmu).T
    assignment[0] = assignment[0].astype(int)
    cluster_assignments = assignment[0].map(map_dict)
    df_assigned = pd.DataFrame(cluster_assignments)
    return df_assigned


class BuildSOM(object):
    def __init__(self):
        pass

    def build_som(self, X):
        print('Building SOM...')
        sm = SOMFactory().build(X, normalization = 'var',
                                mapsize=(15,15), initialization='pca')
        sm.train(n_job=1, verbose='info',
                 train_rough_len=100, train_finetune_len=200)

        topographic_error = sm.calculate_topographic_error()
        quantization_error = np.mean(sm._bmu[1])
        print ("Topographic error = {}; Quantization error = {}"\
        .format(topographic_error,quantization_error))
        return sm


def main():
    show_plots = False
    subset = 'none'
    df = pd.read_csv('extracted_features.csv')
    df.set_index('id', inplace=True)
    df.dropna(axis=1, inplace=True)
    print('There are {} samples'.format(len(use_df)))
    X = scaler.fit_transform(use_df)
    som = BuildSOM()
    model = som.build_som(X)
    if show_plots == True:
        plot_figures(sm)
    print('Getting optimal K-clusters...')
    nclus = 6
    clusters = model.cluster(n_clusters=nclus)
    df_assigned = assign_from_som(clusters, model)
    df_assigned.index = use_df.index
    print('Outputting...')
    df_assigned.to_csv('som_clusters.csv')

if __name__ == '__main__':
    main()

# Find a way to implement this to do hyperparameter search automatically
#som_scores = knn_elbow(model._normalizer.denormalize_by(model.data_raw,
#                                                        model.codebook.matrix),
#                                                        40)
