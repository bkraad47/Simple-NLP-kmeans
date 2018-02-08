# Solution to https://github.com/Isentia/Coding-Challenge/blob/master/NLP-ML-Architect-Challenge.md
# Reference - http://www.artfact-online.fr/blog/blog-post/6
# Simple K-means classifier by Badruddin Kamal

import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pandas as pd

# Download nltk packages

nltk.download('stopwords')
nltk.download('punkt')


def tokenizer(text):

    # Tokenizes and stems the text using PorterStemmer

    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t
              not in stopwords.words('english')]
    return tokens


df = \
    pd.read_json('https://raw.githubusercontent.com/Isentia/Coding-Challenge/master/NLP-test.json'
                 , orient='records')
searchable_text = []
headline = []
for i in df['hits']['hits']:
    headline.append(i['_source']['headline'])
    searchable_text.append(i['_source']['searchabletext'].lower())
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                   stop_words=stopwords.words('english'
                                   ), max_df=0.8, min_df=0.2)

# Builds a tf-idf matrix for the sentences

tfidf_matrix = tfidf_vectorizer.fit_transform(searchable_text)

# Set cluster size

nclusters = 60
kmeans = KMeans(n_clusters=nclusters, max_iter=600)
kmeans.fit_predict(tfidf_matrix)
clusters = collections.defaultdict(list)
for (i, label) in enumerate(kmeans.labels_):
    clusters[label].append(i)

for cluster in range(nclusters):
    for (i, item) in enumerate(clusters[cluster]):
        if i == 0:
            print ('cluster ', cluster,
                   ' - closest centroid document headline : ', headline[item])
        print ('\tDocument index ', i, ' - headline : ', headline[item])
