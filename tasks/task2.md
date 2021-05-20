# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# Use item features

* use ``tags`` data
* each item has ``number of tags`` features
* feature value of a tag is number of users who associated the item with that tag
* create a sparse item-feature (movie-tag) matrix

# Compute cosine distances between items

* compute pairwise distances between all items and get item-item distance matrix
* if it's too large (doesn't fit in memory), use smaller MovieLens dataset (10M e.g.)
* you can also leave only several thousands of most popular items

# Compute ratings by KNN according to the distance matrix

* simple matrix multiplication
* if doesn't fit into memory, predict for only a small number of users

# How to improve

* compute dense (IF-IDF or word2vec) representations of tag features instead of sparse ones
* compute other distances besides cosine
* use other datasets
* combine with popular recommendarions for cold users
