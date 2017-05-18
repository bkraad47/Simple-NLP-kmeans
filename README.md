# Simple NLP document  - Badruddin Kamal (Raad)

## Solution to https://github.com/Isentia/Coding-Challenge/blob/master/NLP-ML-Architect-Challenge.md

A simple NLP application that takes documents as JSON and runs a K-means(Default 60 clusters) clustering algorithm. Documents are indexed based on their closest centroids(Which are randomly selected).

It uses nltk nltk stopwords and porterstemmer

## Installation of dependencies and environment

`conda env create -f environment.yml`

`activate isentiaNLP`

## Running / Deployment

`python isential_NLP_main.py`

## Reference

http://www.artfact-online.fr/blog/blog-post/6