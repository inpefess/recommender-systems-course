# Getting and Understanding the Data

Download the MovieLens 25M dataset: https://grouplens.org/datasets/movielens/
Create charts of:
* number of unique users in the system by month
* number of ratings each month
* number of new users for each month
* number of new items for each month
* number of unique items rated for each month
* number of users by the total number of items rated
* number of items by the total number of users who rated them
* average, maximal, and minimal number of items rated by a user for each month
* most popular items by genre
* genres popularity chart

# Splitting on train and test

Write several functions for train/test split in recommendations.

Split the `ratings` dataset into two part by a `timestamp`:
* 95% of older ratings should go to the train
* 5% of newer ratings should go the test

Random split: 95%/5% train/test

Leave-one-out split:
* for each user put their last rating into test
* all others go in train

# Generating Recommendations

Create a dataset with columns user, item, predicted rating, where for each user there are ten most popular items recommended (same for all users).

As an alternative, recommend ten most popular items of the preferred genre of a user. Preferred genre is defined as such, in which the user rated most items.

# Evaluate recommendations

For each user compute:
* what percentage of items recommended she really reated (in testing period)
* what percentage of the items from the test set were recommended to the user
* aggregate those percentages using mean and median

Compare different strategies (with and without genres). Observe how the metrics change when changing the splitting scheme.
