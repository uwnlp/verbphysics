#!/bin/bash

#
# verbphysics
#
# Data retrieval script.
#
# author: mbforbes
#

# Get and extract ngramdb (cached query and pmi) data
mkdir data/ngramdb/
cd data/ngramdb/
curl https://storage.googleapis.com/ai2-mosaic-public/projects/verb-physics/ngramdb-cache.tar.gz > ngramdb-cache.tar.gz
tar -xzf ngramdb-cache.tar.gz
rm ngramdb-cache.tar.gz
cd ../..

# Get and convert GloVe (word embedding) data
mkdir data/glove/
curl https://nlp.stanford.edu/data/wordvecs/glove.6B.zip > data/glove/glove.6B.zip
unzip data/glove/glove.6B.zip -d data/glove/
python src/glove.py
cd data/glove/
rm glove.6B.100d.txt
rm glove.6B.200d.txt
rm glove.6B.300d.txt
rm glove.6B.50d.txt
rm glove.6B.zip
cd ../..

# Get embedding-trained unary factor weights
mkdir data/emb/
cd data/emb/
curl https://storage.googleapis.com/ai2-mosaic-public/projects/verb-physics/emb-trained-weights.tar.gz > emb-trained-weights.tar.gz
tar -xzf emb-trained-weights.tar.gz
rm emb-trained-weights.tar.gz
cd ../..

# Get wordnet data for NLTK
python -m nltk.downloader wordnet
