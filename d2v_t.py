from os import listdir
from os.path import isfile, join

# posfiles = []
# posfiles = [f for f in listdir("train/pos") if f.endswith('.txt')]

# negfiles = []
# negfiles = [f for f in listdir("train/neg") if f.endswith('.txt')]

pos_reviews = []
for f in listdir("train/pos"):
    with open("train/pos/"+f, 'r') as f:
        pos_reviews.extend(f.read()+'\n')

with open('pos.txt', 'w') as outfile:
        outfile.writelines(pos_reviews)


neg_reviews = []
for f in listdir("train/neg"):
    with open("train/neg/"+f, 'r') as f:
        neg_reviews.extend(f.read()+'\n')

with open('neg.txt', 'w') as outfile:
        outfile.writelines(neg_reviews)


unsup_reviews = []
for f in listdir("train/unsup"):
    with open("train/unsup/"+f, 'r') as f:
        unsup_reviews.extend(f.read()+'\n')

with open('unsup.txt', 'w') as outfile:
        outfile.writelines(unsup_reviews)
