import csv
import logging
import re
import gensim
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NurseBioSentences(object):
    def __init__(self, file_name):
        self.file_name = file_name
    def __iter__(self):
        with open('stop_words.txt', 'r') as f:
            stopwords = [line.strip() for line in f]
        with open(self.file_name, 'rb') as csvFile:
            nursereader = csv.reader(csvFile)
            counter = 0
            for row in nursereader:
                bio = row[2]
                bio = re.sub(r'[^\w]', ' ', bio)
                bio = bio.lower().strip()
                yield [word for word in bio.split()
                       if word not in stopwords]

sentences = NurseBioSentences('nurse_annotated_2000.csv')

model = gensim.models.doc2vec.Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4)
with open('annotatedNurseBioVecs.csv', 'wb') as csvfile:
    vecwriter = csv.writer(csvfile)
    header = []
    for i in range(1,101):
        header.append('value' + str(i))
    vecwriter.writerow(header)
    totalnurses = 0
    for bio in sentences:
        totalwords = 0
        vecsum = np.zeros((100,))
        for word in bio:
            if word in model.vocab:
                totalwords += 1
                vecsum = np.add(model[word], vecsum)
        if (totalwords != 0):
            vecave = vecsum / totalwords
        else:
            vecave = vecsum
        vecwriter.writerow(vecave)
        totalnurses += 1
    print totalnurses

            # print model[word]
