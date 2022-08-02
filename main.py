#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
#Setup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import *


#Categories to focus clustering analysis on
categories = [
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'alt.atheism',
 'soc.religion.christian',
]

#Working with the 20 newsgroups dataset, a popular dataset for clustering 
dataset = fetch_20newsgroups(subset='train', categories=categories, 
                    shuffle=True, remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame(dataset.data, columns=["corpus"])

#Checking first few rows of dataset
df.head(10)

#Cleaning textual data
df['cleaned'] = df['corpus'].apply(lambda x: clean_text(x, remove_stopwords=True))

#Checking cleaned data
df.head(10)

#Calculating tfidf values, and setting upper and lower ceilings on term frequencies
#Removing words that are too common or too rare
#Standardizing data since Kmeans works on Euclidean distances of numeric data and not text
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X = vectorizer.fit_transform(df['cleaned'])

#Clustering documents based on text with Kmeans
# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=None)
# fitting the model
kmeans.fit(X)
# storing cluster labels 
clusters = kmeans.labels_


#Checking dimensions of DTM matrix
X.shape

#Reducing dimensions of DTM matrix using PCA
pca = PCA(n_components=2, random_state=42)
#storing the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
pc1 = pca_vecs[:, 0]
pc2 = pca_vecs[:, 1]


#Storing clusters and pca vectors in dataframe 
df['cluster'] = clusters
df['pc1'] = pc1
df['pc2'] = pc2


#Returning top terms for each cluster
def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    #grouping  the TF-IDF vector by cluster
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()
    #fetching tf-idf terms
    terms = vectorizer.get_feature_names() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        # for each row , finding terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
            
get_top_keywords(20)
#Clearly cluster 0 is rlated to sports, cluster 1 to tech and cluster 2 to religion

#mapping clusters to appropriate labels 
cluster_map = {0: "sport", 1: "tech", 2: "religion"}
df['cluster'] = df['cluster'].map(cluster_map)

#Visualizing clusters
plt.figure(figsize=(12, 7))
plt.title("TKMeans 20newsgroup clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("PC1", fontdict={"fontsize": 16})
plt.ylabel("PC2", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette="viridis")
plt.show()
