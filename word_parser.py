# import nltk
# nltk.download()
import os
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class word_parser(object):
    def __init__(self, split_idx, num_train, num_dev, num_test, ngram_range=(1,1), min_df=10, binary=False, norm='l2'):
        self.vectorizer = TfidfVectorizer(preprocessor=None, tokenizer=None, stop_words=None, ngram_range=ngram_range, min_df=min_df, binary=binary, norm=norm)
        self.split_idx = split_idx
        self.num_train = num_train
        self.num_dev = num_dev
        self.num_test = num_test

    def tokenize_keep_noun(self, doc):
        lemmatizer = WordNetLemmatizer()
        all_tokens = []
        for sentence in doc:
            tokens = []
            t = word_tokenize(sentence)
            tokens_pos = pos_tag(t)
            for token in tokens_pos:
                # only keep nouns
                if token[1] == 'NN':
                    tokens.append(token[0])
                # singularize plural nouns
                elif token[1] == 'NNS': 
                    tokens.append(lemmatizer.lemmatize(token[0], pos='n').encode('ascii', 'ignore'))
            all_tokens.append(' '.join(tokens))
        return all_tokens

    def parse_descriptions_doc(self, data_dir, num_doc):
        docs = []
        for i in range(num_doc):
            path = os.path.join(data_dir, "%d.txt" % i)
            with open(path) as f:
                docs.append(f.read())
        return docs
    
    def parse_descriptions(self):
        train_dev_desc = np.asarray( self.parse_descriptions_doc("data/descriptions_train", num_doc=(self.num_train+self.num_dev)) )
        test_desc = np.asarray( self.parse_descriptions_doc("data/descriptions_test", num_doc=self.num_test) )

        train_desc = train_dev_desc[self.split_idx[:self.num_train]]
        dev_desc = train_dev_desc[self.split_idx[self.num_train:]]

        train_tokens = self.tokenize_keep_noun(train_desc)
        dev_tokens = self.tokenize_keep_noun(dev_desc)
        test_tokens = self.tokenize_keep_noun(test_desc)

        d_train = self.vectorizer.fit_transform(train_tokens)
        d_dev = self.vectorizer.transform(dev_tokens)
        d_test = self.vectorizer.transform(test_tokens)

        d_train = np.asarray(d_train.toarray())
        d_dev = np.asarray(d_dev.toarray())
        d_test = np.asarray(d_test.toarray())
        
        print("Built all d matrices!")
        print("d_train shape:", d_train.shape)
        print("d_dev shape:", d_dev.shape)
        print("d_test shape:", d_test.shape)
        
        return d_train, d_dev, d_test
    
    def parse_tags_doc(self, data_dir, num_doc):
        tags = []
        for i in range(num_doc):
            with open(data_dir + str(i) + '.txt') as f:
                lines = f.readlines()
                tag = []
                for line in lines:
                    tag.append(line.split(':')[1].rstrip())
                tags.append(' '.join(tag))
        return tags
    
    def parse_tags(self):
        train_dev_tags = np.asarray(self.parse_tags_doc('data/tags_train/', num_doc=(self.num_train+self.num_dev)))
        test_tags = np.asarray(self.parse_tags_doc('data/tags_test/', num_doc=(self.num_test)))
        
        train_tags = train_dev_tags[self.split_idx[:self.num_train]]
        dev_tags = train_dev_tags[self.split_idx[self.num_train:]]

        train_tokens = self.tokenize_keep_noun(train_tags)
        dev_tokens = self.tokenize_keep_noun(dev_tags)
        test_tokens = self.tokenize_keep_noun(test_tags)

        t_train = self.vectorizer.transform(train_tokens)
        t_dev = self.vectorizer.transform(dev_tokens)
        t_test = self.vectorizer.transform(test_tokens)


        t_train = np.asarray(t_train.toarray())
        t_dev = np.asarray(t_dev.toarray())
        t_test = np.asarray(t_test.toarray())
        
        # bow = BagofWords(t_train)
        # t_train = bow.getFeatures(t_train)
        # t_dev = bow.getFeatures(t_dev)
        # t_test = bow.getFeatures(t_test)

        print("Built all t matrices!")
        print("t_train shape:", t_train.shape)
        print("t_dev shape:", t_dev.shape)
        print("t_test shape:", t_test.shape)
        
        return t_train, t_dev, t_test
