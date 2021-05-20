# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# Use only ratings

* for collaborative filtering we use `ratings` as features
* each item has `number of users` features
* feature value is rating
* create a sparse item-feature (movie-user) matrix

# Predict ratings using KNN based on cosine distance

* compute cosine distance matrix between items
* make predictions in the same way as for content-based recommendations

# How to improve

* center ratings around each user's average
* normalize ratings according to their distributions for each user 
* binarize ratings
* find the best value for `K` (the number of nearest neighbours)
* use SLIM to learn the best distance matrix
