# Clustering_TextData_With_Document_Vectorization
Conducted clustering analysis on textual data using Kmeans, an unsupervised machine learning algorithm. Preprocessed data and vectorized the textual data since Kmeans calculates the Euclidean distance between fields of each of record and Euclidean distance cannot be calculated for text. 

Vectorized text by calculating tf-idf (term frequences-inverse document frequencies) values, creating a document term matrix. Filtered out terms that are too frequent or too rare, focusing on terms that add meaning to the data. 

Reduced the dimensionality of the document term matrix through PCA, to visualize the cluster solution across the principal components. 

Requirements: 
Python



