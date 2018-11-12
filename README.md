# Project-Echo_edsa
Project Echo repo for the edsa team members

# Clustering
To assist in model building as well as product segmentation, product are
clustered using various approaches, including self-organizing maps,
dimensionality reduction (t-SNE and UMAP) followed by Hierarchical Density
Based Clustering (HDBSCAN), or Dynamic Time Warping.

#Final model
The winning model was a self organizing map which is clustered using k-KMeans
clustering.

The code for producing the model can be found in the src folders.
Data should be organized into the data structure.
Overall folder structure inspired by Cookiecutter.

To run the code, install src and run modules as needed:
`pip3 install -editable .`
