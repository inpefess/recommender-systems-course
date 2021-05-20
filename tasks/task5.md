# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# Bayesian Personalized Ranking

* use BPR from `implicit` package
* compare to our previous recommenders
* use BPR from `lightfm` package
* compare to our previous recommenders

# How to improve

* try using `warp` or other losses from `lightfm` package
* try using item features (tags) to improve quality
* try implementing BPR-like algorithm using LightGBM
