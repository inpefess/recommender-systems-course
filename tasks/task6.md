# Use previous pipeline

* MovieLens 25M dataset
* 95/5 train/test split by timestamp
* evaluate Hit-rate@10

# Embedding-based architecture

* use ``pytorch`` or ``tensorflow``
* create a simple NN architecture (inner product of embedding layers for users and items)
* cross-entropy loss (implicit feedback)
* sample one negative at random for each positive

# How to improve

* change loss function, the number of epochs, optimizer, and learning rate
* use auto-encoder architecture
* use sparse tensors
* stack more layers
