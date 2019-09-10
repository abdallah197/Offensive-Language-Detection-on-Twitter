import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import twokenize

np.random.seed(500)


df = pd.read_csv('data/train.tsv', sep ='\t', names = ['label', 'tweet'])
df['label'] = df['label'].map({'NOT':0, 'OFF': 1})
df['tweet'].dropna(inplace=True)
df['tweet'] =df['tweet'].map(lambda x: x.lower())
df['tweet'] = df['tweet'].map(lambda x: twokenize.tokenizeRawTweetText(x))
