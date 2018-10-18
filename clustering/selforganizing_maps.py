import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sompy
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView


agg = pd.read_csv('sku_labels.csv')
df = pd.read_csv('extracted_features.csv')
df.dropna(axis=1, inplace=True)

print('''____________________________ \n Reading and scaling data 
    ____________________________''')

scale = MinMaxScaler()
skus = df['id']
df.set_index('id', inplace=True)
X = scale.fit_transform(df)

names = df.columns

print('''____________________________ \n Building SOM 
____________________________''')
sm = SOMFactory().build(X, normalization = 'var', mapsize=(15,15), initialization='pca')
sm.train(n_job=1, verbose='info', train_rough_len=20, train_finetune_len=20)

topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

view2D  = View2D(15,15,"time-series",text_size=10, names=names)
view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)

vhts  = BmuHitsView(12,12,"Hits Map",text_size=12)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

sm._calculate_ms_and_mpd()

u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
UMAT  = u.build_u_matrix(sm, distance=1, row_normalized=False)
UMAT = u.show(sm, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)

sm.cluster(4)
hits  = HitMapView(10,10,"Clustering",text_size=12)
a = hits.show(sm)

def knn_elbow(df, k_range=20, plot=True):
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
        
    return scores

som_scores = knn_elbow(sm._normalizer.denormalize_by(sm.data_raw, sm.codebook.matrix), 40)

print('''____________________________ \n Getting optimal K-clusters 
____________________________''')

clusters = sm.cluster(n_clusters=5)

map_dict = {}
for i, j in enumerate(clusters):
    map_dict[i] = j

assignment = pd.DataFrame(sm._bmu).T
assignment[0] = assignment[0].astype(int)


cluster_assignments = assignment[0].map(map_dict)
df_assigned = pd.DataFrame(cluster_assignments)
df_assigned.index = df.index

print('''____________________________ \n Outputting 
____________________________''')

df_assigned.to_csv('som_clusters.csv')