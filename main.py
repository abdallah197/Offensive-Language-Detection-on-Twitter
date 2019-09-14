import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import  naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import twokenize

# %%

df = pd.read_csv('data/train.tsv', sep='\t', names=['label', 'tweet'])
df['label'] = df['label'].map({'NOT': 0, 'OFF': 1})
df['tweet'].dropna(inplace=True)
dft = pd.read_csv('data/dev.tsv', sep='\t', names=['label', 'tweet'])
dft['label'] = dft['label'].map({'NOT': 0, 'OFF': 1})
dft['tweet'].dropna(inplace=True)
corpus = df + dft
trainx, testx, trainy, testy = df['tweet'], dft['tweet'], df['label'], dft['label']
cv = CountVectorizer(tokenizer=lambda text: twokenize.tokenizeRawTweetText(text), vocabulary=None)
X_train_cv = cv.fit_transform(trainx)
X_test_cv = cv.transform(testx)

def preprocessing(text):
    return text.replace('@USER', '')


cv = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                     encoding='utf-8', input='content',
                     lowercase=True, max_df=1.0, max_features=None, min_df=1,
                     ngram_range=(1, 1), preprocessor=None, stop_words=None,
                     strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                     tokenizer=twokenize.tokenizeRawTweetText, vocabulary=None)
X_train_cv = cv.fit_transform(trainx)
X_test_cv = cv.transform(testx)
naive_bayes = MultinomialNB().map(lambda x: twokenize.tokenizeRawTweetText(x))
corpus = df+dft

naive_bayes.fit(X_train_cv, trainy)
predictions = naive_bayes.predict(X_test_cv)
#%%
print("Accuracy score: ", accuracy_score(testy, predictions))
print("Precision score: ", precision_score(testy, predictions))
print("Recall score: ", recall_score(testy, predictions))


#%%
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_cv,trainy)
predictions_SVM = SVM.predict(X_test_cv)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, testy)*100)
