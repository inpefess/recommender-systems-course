# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# Cold Items Analytics

* compute the percentage of cold items in the test set
* compute the percentage of interactions with cold items (in the test set)
* compute the percentage of users having cold items in the test set
* compute the percentage of users having only cold items in the test set

# Recommend Cold Items

* build an ALS recommender
* build a neighbourhood-based content-based recommender
* define factors of cold items as weighted sums of ALS factors
* (use content-based item similarities as weights)
* compare quality of pure AlS and the ALS with added cold items factors

# How to improve

* change ALS and content-based parameters
* use a regressor instead of KNN
* use a DNN instead of KNN
