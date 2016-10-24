# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle
from tweet_tokenizer import Tokenizer
# numpy
import numpy

import pandas as pd
# csv
import csv
# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys
import io
import re

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)



class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        self.tok = Tokenizer(preserve_case=False)
        with open('stop_words.txt', 'r') as f:
            self.stopwords = [line.strip() for line in f]
    def __iter__(self):
        for prefix, source in self.sources.items():
            reader = pd.read_csv(source, names=['class', 'screen_name', 'description', 'name', 'sc', 'fc', 'fric', 'ID'])
            # reader.next()
            for line_id, line in enumerate(reader['description']):
                line = re.sub(r'[^\w]', ' ', line)
                words = self.tok.tokenize(line)
                words = [word for word in words if word not in self.stopwords]
                yield TaggedDocument(words, [prefix+ '_' + reader['class'][line_id]+ '_%s' % line_id])
    def to_array(self):
        self.sentences = []
        for prefix, source in self.sources.items():
            reader = pd.read_csv(source, names=['class', 'screen_name', 'description', 'name', 'sc', 'fc', 'fric', 'ID'])
            for line_id, line in enumerate(reader['description']):
                line = re.sub(r'[^\w]', ' ', line)
                words = self.tok.tokenize(line)
                words = [word for word in words if word not in self.stopwords]
                self.sentences.append(TaggedDocument(words, [prefix+'_' + reader['class'][line_id]+ '_%s' % line_id]))
        return self.sentences
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


log.info('source load')
# sources = {'test':'1884_test.csv', 'train':'1800_train.csv'}
sources = {'train':'annotated_English_0-4000(3683nurses).csv'}
# sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)
sentences.to_array()
# print sentences[0]
log.info('D2V')
# model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model = Doc2Vec(min_count=1, window=5, size=300, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
    log.info('EPOCH: {}'.format(epoch))
    model.train(sentences.sentences_perm())

log.info('Model Save')
model.save('./nurses.d2v')
model = Doc2Vec.load('./nurses.d2v')

log.info('Sentiment')
# train_arrays = numpy.zeros((1800, 100))
# train_labels = numpy.zeros(1800)
# test_arrays = numpy.zeros((1884, 100))
# test_labels = numpy.zeros(1884)
# train_arrays = []
# train_labels = []
# test_arrays = []
# test_labels = []

# for instance in sentences.to_array():
#     if instance.tags[0].split('_')[0] == 'test':
#         test_arrays.append(model.docvecs[instance.tags[0]])
#         if instance.tags[0].split('_')[1] == 'No':
#             test_labels.append(0)
#         else:
#             test_labels.append(1)
#     else:
#         train_arrays.append(model.docvecs[instance.tags[0]])
#         if instance.tags[0].split('_')[1] == 'No':
#             train_labels.append(0)
#         else:
#             train_labels.append(1)

X = []
y = []
for instance in sentences.to_array():
    X.append(model.docvecs[instance.tags[0]])
    if instance.tags[0].split('_')[1] == 'No':
        y.append(0)
    else:
        y.append(1)



# for i in range(12500):
#     prefix_train_pos = 'TRAIN_POS_' + str(i)
#     prefix_train_neg = 'TRAIN_NEG_' + str(i)
#     train_arrays[i] = model.docvecs[prefix_train_pos]
#     train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
#     train_labels[i] = 1
#     train_labels[12500 + i] = 0

print X
print y

# test_arrays = numpy.zeros((1884, 100))
# test_labels = numpy.zeros(1884)

# for i in range(12500):
#     prefix_test_pos = 'TEST_POS_' + str(i)
#     prefix_test_neg = 'TEST_NEG_' + str(i)
#     test_arrays[i] = model.docvecs[prefix_test_pos]
#     test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
#     test_labels[i] = 1
#     test_labels[12500 + i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(X, y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(X, y)
