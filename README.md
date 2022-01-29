# RichB's Fan Club's Bill.com Submission
Robbie Kenworthy, Alex Holzach, Timofey Efimov, Mahmoud Al-Madi

Link to Google Colab for larger files that couldn't be uploaded: https://drive.google.com/drive/folders/16KLaNwlE7BfZfNMVMtA0a0MJ5YQ0vAQK?usp=sharing

Languages: Python

Libraries: json, random, collections, pandas, PIL, matplotlib, scipy, gensim, sklearn, tensorflow.keras, google.colab, numpy

Purpose: 
Bill.com presented a challenge relevant to their applications in the business domain. Businesses can be represented as nodes within a graph, and the connections between these businesses can be represented as edges. Bill.com is interested in helping their clients by using this graph structure to provide recommendation and searches in their database. Given a smaller emulation of their dataset, we can represent each node a webpage in a social media network. Each webpage has a description and type, and they have connections between each other. With this structure, we sought to predict whether or not a link should exist between nodes, when the data set was made more sparse. 

We accomplished this primarily by use of the node2vec algorithm, on both the word desciptions and node connections within the graph itself. Given these embeddings for both the nodes and words, we can create a unique feature vectors for each pair of nodes by finding the square difference between the the embeddings. With this, we implemented both a KNN and neural net model to predict whether or not there should exist an edge between two nodes. The KNN was a baseline model that we compared the neural net models performance against. 

Challenges:
Representing the different information in a similar dimension vector space was a challenge. We had to learn the intricacies of node2vec to process all of our different data. Another difficulty was how to represent the information we gather about each node as a pair of nodes to give a binary label for whether or not a node exists, and whether each method we explored would be beneficial. Finally, our models still need improvements. The accuracy is nowhere near what we had wanted, and varied massively compared to our validation set. There must be some overfitting somewhere that must be explored in order to gain higher performance. 

Relevant files Included:
PCA.ipynb
In this file  I did the PCA for the feature vectors for each node to see if the sentence as a whole is a useful feature for the node. The size of the vector is the same for all of them since we did windows for words that are convolved with the sentences to normalize the length.

PCAwords.ipynb
In this file I did the PCA analysis of the vocabulary of all unique words that are present in all nodes description combined. The idea was to see if distinguishing nodes by their descriptions is a useful idea and helps to predict the results better

Visualization.ipynb
In this file I performed the 3D visualization of the graph to see how nodes from different page types are clustered throughout the graph.

node2vec.ipynb
Inputs: training_graph.csv, node_features_text.json, node_classification.csv, isolated_nodes.csv, test_edges.csv
This file surrounded around processing the different graph structures given and creating .csv files to safe the information for later. This included the random walks for the training and testing data, and prunning of the training set to reduce overfitting.

Outputs: testing_walks.csv, testing_node2vec.csv, sparse_walks.csv, sparse_node2vec.csv, node2vec_walks.csv, node2vec_features.csv

node2vec_ml.ipynb
Inputs: sparse_node2vec.csv, word2vec_features.csv, node_classification.csv, training_graph.csv, testing_node2vec.csv, test_edges.csv, isolated_nodes.csv, test_labels.csv

Generates data matrix and inputs into KNN and Neural Network

Outputs: Accuracies for both KNN and Neural Network

word2vec_alex.ipynb
Inputs: training_graph.csv, node_features_text.json, node_classification.csv, isolated_nodes.csv, test_edges.csv, test_labels.csv.

Extracts word2vec features for the descriptions of every node in node_features_text.json. Uses word2vec features from every node to determine similarities between any two nodes. Uses KNN and a deep neural network classifier to estimate the labels from the edges listed in test_edges.csv.

Outputs: A KNN and a deep neural network classifier model for estimating the existence of an edge between any two nodes in the network.
