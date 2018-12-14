# get bag of words features
import numpy as np
from sklearn.preprocessing import normalize

class BagofWords(object):
    def __init__(self, train):
        self.bag = set()
        for words in train:
            self.bag |= set(words)
        self.bag = list(self.bag)
        self.bag_idx = {x:i for i, x in enumerate(self.bag)}
        
    def getFeatures(self, data, post='l2'):
        feature = np.zeros((len(data), len(self.bag)))
        for i in xrange(len(data)):
            for word in data[i]:
                try:
                    feature[i, self.bag_idx[word] ] += 1
                except KeyError: pass
        if post=='l2':
            feature = normalize(feature, norm='l2', axis=1)
        return feature