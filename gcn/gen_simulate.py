from collections import defaultdict

import numpy as np

################################################
# The original Data format for reference
################################################
#  ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
# adj: unweighted undirected graph
# features: weighted fixed length vector
# y_train
# y_val
# y_test

################################################
# Important factors for simulated data
# 1, features(distributed or one-hot):
#   1.1, random initialization
#   1.2, random Gaussian, assign random mean/std to each class, and control overlap among classes
# 2, graph:
#   2.1, random connection, echo NN
#   2.2, KNN without initialization, strong connection within clusters, few link among clusters
#   2.3, K-means with initializing K clusters, strong connection within clusters, few link among clusters
#   2.4, randomly select connection level for each pair of classes, and for in-class
# 3, label:
#   3.1, Oracle: a random NN receiving features and graph, outputs softmax
#   3.2, K labels for K cluster (KNN, K-means)
#   3.3, random

# sets of configuration
# 1, feat: rand distributed; label: random; graph: random;
# 2, feat: rand distributed; label: random; graph: sample 2 link numbers for in-class and beteewn-class;
# 3, feat: rand distributed; label: random; graph: An oracle
# 4, feat: related to label; label: sample feat columns, mean and std; graph: sample 2 link numbers for in-class and beteewn-class;
# 5, feat: related to label; label: sample feat columns, mean and std; graph: An oracle
################################################
feat_num = 50
mean = [5, 8, 10, 12, 15]
cov = [[10, 5, 3, 2, 1],
       [5, 10, 3, 2, 1],
       [3, 3, 10, 2, 1],
       [2, 2, 2, 10, 1],
       [1, 1, 1, 1, 10]]
c1 = np.random.multivariate_normal(mean, cov, feat_num / 2)
mean = [10, 15, 12, 5, 8]
cov = [[10, 5, 3, 2, 1],
       [5, 10, 3, 2, 1],
       [3, 3, 10, 2, 1],
       [2, 2, 2, 10, 1],
       [1, 1, 1, 1, 10]]
c2 = np.random.multivariate_normal(mean, cov, feat_num / 2)

# print c1, c2

# init graph

adj = defaultdict(list)
# link = {}
# for k, v in link.iteritems():
s = [(1, 2),
     (1, 3),
     (1, 5),
     (1, 23),
     (2, 3),
     (2, 18),
     (2, 24),
     (3, 4),
     (3, 12),
     (3, 16),
     (5, 6),
     (5, 12),
     (5, 18),
     (5, 24),
     (5, 26),
     (6, 8),
     (6, 10),
     (6, 14),
     (6, 18),
     (6, 19),
     (7, 10),
     (7, 11),
     (7, 19),
     (7, 21),
     (7, 24),
     (8, 24),
     (9, 12),
     (9, 13),
     (9, 15),
     (9, 16),
     (9, 26),
     (9, 17),
     (9, 18),
     (10, 15),
     (10, 17),
     (10, 19),
     (10, 21),
     (10, 22),
     (10, 25),
     (11, 20),
     (11, 21),
     (12, 21),
     (13, 14),
     (14, 18),
     (14, 24),
     (15, 16),
     (15, 24),
     (15, 34),
     (16, 17),
     (16, 21),
     (16, 24),
     (17, 18),
     (17, 23),
     (18, 20),
     (18, 21),
     (18, 22),
     (19, 21),
     (19, 24),
     (20, 24),
     (21, 22),
     (21, 24),
     (22, 23),
     (22, 42),
     (23, 24),
     (25, 26),
     (25, 27),
     (25, 30),
     (25, 31),
     (25, 32),
     (25, 33),
     (26, 31),
     (26, 12),
     (26, 38),
     (26, 39),
     (26, 40),
     (26, 44),
     (26, 48),
     (26, 49),
     (27, 29),
     (27, 30),
     (27, 31),
     (27, 33),
     (27, 39),
     (27, 42),
     (28, 30),
     (28, 15),
     (28, 31),
     (28, 37),
     (29, 30),
     (29, 39),
     (29, 42),
     (30, 32),
     (30, 33),
     (30, 35),
     (30, 39),
     (30, 48),
     (31, 48),
     (31, 3),
     (32, 37),
     (32, 38),
     (32, 39),
     (32, 40),
     (32, 42),
     (32, 49),
     (33, 49),
     (34, 35),
     (34, 37),
     (34, 39),
     (34, 41),
     (34, 43),
     (34, 45),
     (35, 36),
     (35, 18),
     (35, 38),
     (35, 39),
     (35, 41),
     (35, 42),
     (35, 43),
     (35, 47),
     (36, 37),
     (36, 38),
     (36, 39),
     (36, 21),
     (36, 41),
     (36, 42),
     (36, 43),
     (37, 40),
     (37, 41),
     (37, 45),
     (38, 40),
     (39, 40),
     (40, 41),
     (40, 45),
     (40, 48),
     (41, 43),
     (41, 46),
     (42, 45),
     (43, 45),
     (44, 46),
     (44, 47),
     (45, 46),
     (45, 48),
     (45, 49),
     (46, 47),
     (47, 48),
     (48, 49),
     ]

for k, v in s:
    adj[k].append(v)
    adj[v].append(k)
for k, v in adj.iteritems():
    adj[k] = list(set(v))
print adj
