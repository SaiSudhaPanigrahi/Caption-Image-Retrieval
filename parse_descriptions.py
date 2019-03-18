import os
from sklearn.feature_extraction.text import TfidfVectorizer

class parse_words(object):

	def __init__(self):
		vectorizer = TfidfVectorizer()

def parse_descriptions(data_dir, num_doc):
    docs = []
    for i in range(num_doc):
        path = os.path.join(data_dir, "%d.txt" % i)
        with open(path) as f:
            docs.append(f.read())
    return docs