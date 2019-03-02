#this is to create a small dataset; for ELMo LM learing test purpose
import pandas as pd
import numpy as np

#read csv
data = pd.read_csv("./NER/ner_dataset.csv", encoding = "latin1")

#fill na 
data = data.fillna(method="ffill")

#observe data
data.tail(10)
 
words = list(set(data["Word"].values))
specials = ["ENDPAD", "<S>", "</S>"]
for s in specials:
    words.append(s)


n_words = len(words); n_words

tags = list(set(data["Tag"].values))

n_tags = len(tags); n_tags


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent +=1 
            return s
        except:
            return None 




getter = SentenceGetter(data)

sent = getter.get_next()

#view data 
print(sent)
    
sentences = getter.sentences

max_len = 50

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


from keras.preprocessing.sequence import pad_sequences

#extract words from sentences
X = [[w[0] for w in s] for s in sentences]

#pad sentence , composed with words, with 'ENDPAD'.
#remove X pad
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append('ENDPAD')
    new_X.append(new_seq)
X = new_X

#pad tag len:max_len(50) 
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

from keras.utils import to_categorical
new_y=[]
for i in y:
    new_y.append(to_categorical(i, num_classes=n_tags)) 
y = new_y

#remove_index = [45404, 41770, 10910, 61] which has strange tokens . To make them work, these strange tokens must be washed out and relevant tage be removed.
X=np.delete(X, (45404, 41770, 10910,61),axis=0)
y=np.delete(y, (45404, 41770, 10910,61),axis=0)
