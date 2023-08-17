#!/usr/bin/env python
# coding: utf-8

# importing required libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string
import nltk

from operator import index

import gensim
import gensim.corpora as corpora
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from collections import Counter
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ignore warning
import warnings
warnings.filterwarnings('ignore')


#initialization
stop = stopwords.words('english')
symbols = list(string.punctuation)


# loading datasets
fakeDataset = pd.read_csv("./dataset/Fake.csv")
trueDataset = pd.read_csv("./dataset/True.csv")


# top-5 rows
display(fakeDataset.head())

display(trueDataset.head())


# adding label column
fakeDataset["label"] = "1"
trueDataset["label"] = "0"


# merging datasets
data = [fakeDataset, trueDataset]
dataset = pd.concat(data, ignore_index=True, sort=False)

# dimensions
print(
    "Total number of rows in Fake Dataset {} and True Dataset {} and combined dataset {}"
    .format(fakeDataset.shape, trueDataset.shape, dataset.shape))

print(dataset.columns)

# dropping unneccessary columns
columns = ['title', 'subject', 'date']
dataset.drop(columns=columns, inplace=True)


# after dropping columns
display(dataset.head())

# checking for NaN values
display(dataset.isna().any())

# checking for null values
display(dataset.isnull().any())

# Dataset preprocessing
def cleaning(raw):
    htmlFree = BeautifulSoup(raw, "html.parser")  # removing html tags
    # removing numbers and others except small and capital alphabets
    letters = re.sub("[^a-zA-Z ]", " ", htmlFree.get_text())
    low = letters.lower()  # Converting everything to lower case
    words = low.split()  # spiliting sentences into words
    cleaned = [w for w in words if not w in stop]  # removing stopping words
    return ' '.join(cleaned)

# cleaning the dataset and segregating input and output
x = list(map(cleaning, dataset['text']))
y = dataset['label']

# dataset visualisation
def get_list_of_words(document):
    Document = [a.split(" ") for a in document]
    return Document


document = get_list_of_words(x[:1000])

id2word = corpora.Dictionary(document)
corpus = [id2word.doc2bow(text) for text in document]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                            update_every=3, chunksize=100, passes=50, alpha='auto',
                                            per_word_topics=True)



def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.append(
                    [i, int(topic_num), round(prop_topic, 4) * 100, topic_keywords])
            else:
                break

    return sent_topics_df

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  collocations=False,
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# bar chart for labels
cnt = Counter(y)
yval = list(cnt.values())
xval = list(cnt.keys())
sns.barplot(x=['Fake', "True"], y=yval, palette="Blues_d")
plt.title("Label Counts")
plt.show()

# vector embedding
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 200

# padding
maxlen = 0
word_freqs = Counter()
num_recs = 0

for line in x:
    sentence = line.strip()
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1

# vector embedding initailsation
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

X = np.empty((num_recs, ), dtype=list)
Y = np.zeros((num_recs, ))

# vectorisations
i = 0
for idx, line in enumerate(x):
    sentence = line.strip()
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    Y[i] = int(y[idx])
    i += 1


# padding the sequences
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# splitting the dataset for train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

# defining model
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

# summary of the model
model.summary()


# training the model
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))


# evaluating the model
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print(("Test score: %.3f, accuracy: %.3f" % (score, acc)))


# graph to represent the training and validation
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")
plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


from prettytable import PrettyTable
  
myTable = PrettyTable(["Predicted", "Actual", "Sentence"])
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, 40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    myTable.add_row([int(ypred), ylabel, sent])
print(myTable)

print(classification_report(ytest, [round(i[0]) for i in model.predict(Xtest)]))

print(accuracy_score(ytest, [round(i[0]) for i in model.predict(Xtest)]))


