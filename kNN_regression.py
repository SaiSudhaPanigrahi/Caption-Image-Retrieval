#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import csv
import random
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

num_predict = 20
num_train = 8000
num_dev = 2000
num_test = 2000

split_idx = list(range(num_train + num_dev))
random.shuffle(split_idx)

# Description Parsing
ngram_range = (1,1)
min_df = 8
binary = True
norm = None

# ResNet Parsing
intermediate = False

# weighting
w_i = 13 # 23
w_t = 1


### Parse Descriptions and tags

from word_parser import word_parser
w_parser = word_parser(split_idx=split_idx, num_train=num_train, num_dev=num_dev, num_test=num_test, ngram_range=ngram_range, min_df=min_df, binary=binary, norm=norm)
d_train, d_dev, d_test = w_parser.parse_descriptions()
t_train, t_dev, t_test = w_parser.parse_tags()


# ### Parse ResNet Features

from parse_features import *
i_train, i_dev, i_test = parse_features(split_idx, num_train, num_dev, num_test, intermediate=intermediate)


# ### Regression

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

# Ridge
parameters = {"alpha": [10.0]}
reg = GridSearchCV(Ridge(), parameters, cv=10, verbose=1)

reg.fit(i_train, d_train)
# reg_best = reg.best_estimator_

print("Trained linear regression model!")
print("Summary of best model:")
print(reg)


# ### Cross-Validation

d_dev_pred = reg.predict(i_dev)
dist_i = cdist(d_dev, d_dev_pred, metric='sqeuclidean')
dist_t = cdist(d_dev, t_dev, metric='sqeuclidean')


### Scoring

from scoring import *

dist_all = dist_i * w_i + dist_t * w_t

dist_idx = np.argsort(dist_all, axis=1)
scoring(dist_idx, num_dev)


# ### Test and write to file

d_train_all = np.concatenate([d_train, d_dev])
i_train_all = np.concatenate([i_train, i_dev])
t_train_all = np.concatenate([t_train, t_dev])

reg.fit(i_train_all, d_train_all)

d_test_pred = reg.predict(i_test)
dist_i = cdist(d_test, d_test_pred, metric='sqeuclidean')
dist_t = cdist(d_test, t_test, metric='sqeuclidean')

dist_all = dist_i * w_i + dist_t * w_t

dist_idx = np.argsort(dist_all, axis=1)

from write_to_csv import *
write_to_csv(dist_idx, './pred_kNN_regression.csv', num_predict=num_predict, num_test=num_test)




