# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# ALS

* use ALS from `implicit` package
* compare to our previous recommenders

# How to improve

* center and normalize ratings
* binarize ratings
* find the best value latent factors dimensionality
* implement pure SVD recommender
* try models from `implicit.approximate_als`. Are they faster? Is the quality drastically inferior?
