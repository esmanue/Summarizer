import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 
import re
from nltk.corpus import stopwords
from string import punctuation
from datasets import load_dataset
import gensim
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

# Hugginface data yükledim ve pandas serisine dönüştürdüm
ds = load_dataset("ashraq/financial-news-articles")
train_data = ds["train"]
df = pd.DataFrame(train_data)




print(df.head())
print(df.isnull().sum())


stop_words = set(stopwords.words('english'))


def clean_data(data):
    data = data.lower()
    data = re.sub(r'\s+', ' ', data).strip() #sağ ve soldaki ve ortadaki bosluklari kaldirir
    data = data.split()
    filtered_words = [word for word in data if word not in stop_words] #stopwordsleri filtreliyor
    return ' '.join(filtered_words)


text = str(df["text"][3])
text = clean_data(text)
sentences = text.split()

tokenized = []
for sentence in sentences:
    print(sentence)
    tokenized.append(simple_preprocess(sentence, deacc=True))

# tokenize edıldıkten spnra sozluge yerlestiriyoruz
my_dictionary = corpora.Dictionary(tokenized)
print(my_dictionary)

# once kelimeleri sayisina gore bow olusturuyoruz
def converter_to_BoW(my_dictionary):
    BoW_corpus = [my_dictionary.doc2bow(doc, allow_update=True) for doc in tokenized]
    return BoW_corpus

BoW_corpus = converter_to_BoW(my_dictionary)

# bowdaki  kelimelerin frekansını cıkariyoruz cikarilan frekansa gore tdıdf yapılır
def BoW_List(BoW_corpus):
    word_weight = []
    for doc in BoW_corpus:
        for id, freq in doc:
            word_weight.append([my_dictionary[id], freq])
    return word_weight

#normalizasyonu burada yapiyoruz
tfIdf = models.TfidfModel(BoW_corpus, smartirs='ntc')

# kelimeleri ve normalizasyon yapilmis frekansları listeliyoruz
weight_tfidf = []
for doc in tfIdf[BoW_corpus]:
    for id, freq in doc:
        weight_tfidf.append([my_dictionary[id], np.around(freq, decimals=3)])
print(weight_tfidf)



sentences = sent_tokenize(text)
sentence_scores = []

for summarize in sentences:
    summarize = clean_data(summarize)
    BoW_c = my_dictionary.doc2bow(simple_preprocess(summarize))
    tfidf = tfIdf[BoW_c]
