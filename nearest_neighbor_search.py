#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
import dill
import os
import csv
import random
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
print("Loaded word vectors successfully!")

# constants
num_train = 8000
num_dev = 2000
num_test = 2000
num_predict = 20

split_idx = list(range(num_train + num_dev))
random.shuffle(split_idx)

top_k_d = 10
top_k_i = 10


# ### Parse descriptions

from parse_descriptions import parse_descriptions

def doc_to_vec(sentence, word2vec):
    # get list of word vectors in sentence
    word_vecs = [word2vec.get_vector(w) for w in sentence.split() if w in word2vec.vocab]
    # return average
    return np.stack(word_vecs).mean(0)

# build x matrices
train_dev_desc = parse_descriptions("data/descriptions_train", num_doc=(num_train+num_dev))
test_desc = parse_descriptions("data/descriptions_test", num_doc=num_test)
d_train = np.array([doc_to_vec(train_dev_desc[i], word2vec) for i in split_idx[:num_train]])
d_dev = np.array([doc_to_vec(train_dev_desc[i], word2vec) for i in split_idx[num_train:]])
d_test = np.array([doc_to_vec(d, word2vec) for d in test_desc])

print("Built all x matrices!")
print("x_train shape:", d_train.shape)
print("x_dev shape:", d_dev.shape)
print("x_test shape:", d_test.shape)

# word preprocessing
import re
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

class Preprocess(object):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        # self.stopList = set(stopwords.words("english"))
        self.stopList = set([word.encode('ascii', 'ignore') for word in stopwords.words('english')])
        
    def preprocess(self, string, n_gram=1):
        
        # replace special character with space
        string = re.sub(r'[^a-zA-Z0-9 ]', r' ', string).encode('ascii', 'ignore')
        
        # Lemmatization (handles capitalization), ignoring stop word
        # turn output to ASCII and ignore special character
        ans = [self.stemmer.stem(word).encode('ascii', 'ignore') for word in string.split()]
        ans = [word for word in ans if word not in self.stopList]
        
        return ans

# load descriptions
descriptions_train = [set()] * n_train
processor = Preprocess()
for i in range(n_train):
    with open('data/descriptions_train/' + str(i) + '.txt') as f:
        words = f.read() # readlines()
        descriptions_train[i] = processor.preprocess(words)

descriptions_test = [set()] * n_test
for i in range(n_test):
    with open('data/descriptions_test/' + str(i) + '.txt') as f:
        words = f.read() # readlines()
        descriptions_test[i] = processor.preprocess(words)

# get bag of words features
def BagofWords(train, test):
    bag = set()
    for words in train:
        bag |= set(words)
    bag = list(bag)
    bag_idx = {x:i for i, x in enumerate(bag)}
    print(len(bag))
    # print(bag)
    
    # create feature vectors
    train_features = np.zeros((len(train), len(bag)))
    test_features = np.zeros((len(test), len(bag)))

    data = [train, test]
    features = [train_features, test_features]
    # '''
    for k in [0,1]:
        print('train/test: ', k)
        for i in xrange(len(data[k])):
            # if i%500 == 0: print(k, i)
            for word in data[k][i]:
                try:
                    features[k][i, bag_idx[word] ] += 1
                except KeyError: pass
    # '''
    return train_features, test_features, bag, bag_idx

train_features, test_features, bag, bag_idx = BagofWords(descriptions_train, descriptions_test)

# post-process: L2 normalization
from sklearn.preprocessing import normalize
train_features = normalize(train_features, norm='l2', axis=1)
test_features = normalize(test_features, norm='l2', axis=1)


# ### Parse ResNet Features

def parse_features(features_path):
    vec_map = {}
    with open(features_path) as f:
        for row in csv.reader(f):
            img_id = int(row[0].split("/")[1].split(".")[0])
            vec_map[img_id] = np.array([float(x) for x in row[1:]])
    return np.array([v for k, v in sorted(vec_map.items())])

i_train_dev = parse_features("data/features_train/features_resnet1000_train.csv")
i_train = i_train_dev[split_idx[:num_train]]
i_dev = i_train_dev[split_idx[num_train:]]
i_test = parse_features("data/features_test/features_resnet1000_test.csv") # @ is matrix multiplication for Python 3

print("Built all y matrices!")
print("y_train shape:", i_train.shape)
print("y_dev shape:", i_dev.shape)
print("y_test shape:", i_test.shape)

# load features
features_train_ff = pd.read_csv('data/features_train/features_resnet1000_train.csv', delimiter=',', index_col=0, header=None)
features_test_ff = pd.read_csv('data/features_test/features_resnet1000_test.csv', delimiter=',', index_col=0, header=None)

features_train_ff.index = features_train_ff.index.str.lstrip('images_train/').str.rstrip('.jpg')
features_train_ff.index = pd.to_numeric(features_train_ff.index, errors='coerce')
features_train_ff.sort_index(inplace=True)

features_test_ff.index = features_test_ff.index.str.lstrip('images_test/').str.rstrip('.jpg')
features_test_ff.index = pd.to_numeric(features_test_ff.index, errors='coerce')
features_test_ff.sort_index(inplace=True)


# ### do kNN

def kNN_prediction(train_index_img, test_index_img, train_index_caption, test_index_caption, dist_caption, dist_image, w_caption=2933, w_img=1):
    assert(len(features_train_ff) == len(train_features))
    assert(len(features_test_ff) == len(test_features))
    
    n_train = len(train_index_img)
    n_test = len(test_index_img)
    
    # find closest caption in train 
    # dist_caption = cdist(test_features, train_features, metric='sqeuclidean')
    closest_caption_idx = np.argpartition(dist_caption, top_k, axis=1)[:, :top_k]
    closest_caption_dist = np.asarray([dist_caption[i, closest_caption_idx[i]] for i in range(len(dist_caption))])
    
    # find closest image in test
    # dist_image = cdist(features_train_ff, features_test_ff, metric='sqeuclidean')
    closest_image_idx = np.argpartition(dist_image, top_k, axis=1)[:, :top_k]
    closest_image_dist = np.asarray([dist_image[i, closest_image_idx[i]] for i in range(len(dist_image))])
    
    # get 400 distances (dist_caption+dist_image) for each caption
    dist_final = np.empty((n_test, top_k*top_k))
    idx_final = np.empty((n_test, top_k*top_k))
    for i in range(n_test):
        for j in range(top_k):
            d_caption = closest_caption_dist[i, j]
            for k in range(top_k):
                d_img = closest_image_dist[closest_caption_idx[i, j], k]
                dist_final[i, j*top_k+k] = d_caption * w_caption + d_img * w_img
                idx_final[i, j*top_k+k] = closest_image_idx[closest_caption_idx[i, j], k] # need to fix
    
    # reassign the labels
    # print(test_index_img[3], inx_fina[i, j])
    '''
    for i in range(len(idx_final)):
        for j in range(len(idx_final[0])):
            idx_final[i, j] = test_index_img[int(idx_final[i, j])]
    '''

    # predict
    dist_final_arg = np.argsort(dist_final, axis=1)
    predict = [[] for _ in range(n_test)]
    for i in range(n_test):
        for j in range(top_k*top_k):
            if len(predict[i]) != n_predict and idx_final[i, dist_final_arg[i, j]] not in predict[i]:
                # print(idx_final[i, dist_final_arg[i, j]])
                predict[i].append(idx_final[i, dist_final_arg[i, j]])
                
    for i in range(len(predict)):
        for j in range(len(predict[0])):
            predict[i][j] = test_index_img[int(predict[i][j])]
    
    return predict

def scoring(predict, label):
    print(len(predict), len(label))
    assert(len(predict) == len(label))
    score = 0
    for i in range(len(predict)):
        try:
            idx = predict[i].index(label[i])
            score += (21 - idx) / 20
        except ValueError:
            # print(label[i], predict[i])
            pass
    score /= len(predict)
    print(score)
    return score

# precomputation
from sklearn.model_selection import KFold
n_splits = 3
kf = KFold(n_splits=n_splits)

train_index_img = [train_index for train_index, test_index in kf.split(features_train_ff)]
test_index_img = [test_index for train_index, test_index in kf.split(features_train_ff)]
train_index_caption = [train_index for train_index, test_index in kf.split(train_features)]
test_index_caption = [test_index for train_index, test_index in kf.split(train_features)]

# precompute the distances
dist_caption, dist_image = [], []
for i in range(n_splits):
    dist_caption.append( cdist(train_features[test_index_caption[i]], train_features[train_index_caption[i]], metric='sqeuclidean') )
    dist_image.append( cdist(features_train_ff.values[train_index_img[i]], features_train_ff.values[test_index_img[i]], metric='sqeuclidean') )

dist_caption[0].shape

# do 3 fold cross validation
w_captions = [1500]# , 2000, 2500, 3000, 3500] # modify this line
accuracy = np.empty(len(w_captions))

for i, w_caption in enumerate(w_captions):
    for j in range(1):
        cv_predict = kNN_prediction(train_index_img[j],
                                    test_index_img[j],
                                    train_index_caption[j],
                                    test_index_caption[j],
                                    dist_caption[j],
                                    dist_image[j],
                                    w_caption=w_caption)
        
        accuracy[i] += scoring(cv_predict, test_index_caption[j])
accuracy /= n_splits
plt.plot(accuracy)

from write_to_csv import *
write_to_csv(dist_idx, './pred_nearest_neighbor_search.csv', num_predict=num_predict, num_test=num_test)
