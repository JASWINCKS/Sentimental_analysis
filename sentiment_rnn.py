# Sentimental_Analysis_RNN
# load Libraries
# 
# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import re 
import pickle as pic
import string

# tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

#nltk

import  nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#plotting 
import seaborn as sns
from wordcloud import WordCloud
# == mathplotlib imported above


# load data
data = pd.read_csv("C://Users/cks/Desktop/FILES/Projects/Code/Gen/training.1600000.processed.noemoticon.csv", encoding = 'latin', names =['polarity', 'id', 'date', 'query', 'user', 'text'])

data = data.sample(frac=1)
data = data[:20000]

# print dataset shape
print("Dataset shape is",data.shape)

# print top 5 rows
print(data.head(5))

data['polarity'].unique()

# data[polarity] -->0 = negative
# data[polarity] -->1 = positive

data['polarity'] = data['polarity'].replace(4,1)
print(data.head(5))

print("**"*1000)
# describe the  data
print(data.describe())
print("**"*100)
print("\n")
# count of positive and negative reviews
postive = data[data['polarity'] == 1]
negative = data[data['polarity'] == 0]

print("positive reviews",len(postive))
print("negative reviews",len(negative))

def word_count(words):
    return len(words.split())

# plot word count distribution of positive and negative comments
data['word count'] = data['text'].apply(word_count)

# Filter positive and negative comments based on polarity
pos = data[data['polarity'] == 1]['word count']
neg = data[data['polarity'] == 0]['word count']

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.xlim(0, 45)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.hist([pos, neg], color=['green', 'red'], alpha=0.5, label=['positive', 'negative'])
plt.legend(loc='upper right')
plt.show()

# get common words for training
'''
from collections import Counter
words = []
for l in list(data['text']):
    words = l.split()
    for word in words:
        if len(word) < 2:
            words.append(word.lower())

words_most = Counter(words).most_common(10)
print(words_most)
'''
# Data Preprocessing
import matplotlib.pyplot as plt
plt.ion()
sns.counterplot(data['polarity'])

# remove unwanted attributes
data.drop(['id','date','query','user','word count'],axis=1,inplace=True)
print(data.head(10))

# check for null
print(data.isnull().sum()/ len(data)*100)

# convert data[text] into str type
data['text'] = data['text'].astype(str)

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
print(stopwords)

# nltk download punkt and wordnet
nltk.download('punkt')
nltk.download('wordnet')

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
def process_tweets(tweet):
  # Lower Casing
    tweet = tweet.lower()
    tweet=tweet[1:]
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet) 
    #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #tokenizing words
    tokens = word_tokenize(tweet)
    #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopwords]
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
      if len(w)>1:
        word = wordLemm.lemmatize(w)
        finalwords.append(word)
    return ' '.join(finalwords)


data['processed_tweets'] = data['text'].apply(lambda x: process_tweets(x))
print('Text Preprocessing complete.')
print(data.head(10))


# Vectorizing the data

X = data['processed_tweets'].values
y = data['polarity'].values

print(X.shape,y.shape)

# Convert collection of raw documents to a matrix of TF-IDF features.
vector = TfidfVectorizer(sublinear_tf=True)
X = vector.fit_transform(X)
print('Vector complete.')
print("No of feature_words",len(vector.get_feature_names()))

print(X.shape)
print(y.shape)



# Rnn Model Building
# Define the RNN model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizerfrom keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data.processed_tweets)
sequences = tokenizer.texts_to_sequences(data.processed_tweets)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
model2 = Sequential()
model2.add(layers.Embedding(max_words, 128))
model2.add(layers.LSTM(64,dropout=0.5))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(8, activation='relu'))
model2.add(layers.Dense(1,activation='sigmoid'))
model2.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
checkpoint2 = ModelCheckpoint("rnn_model.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=10,validation_data=(X_test, y_test),callbacks=[checkpoint2])

# Evaluating the model
loss, accuracy = model2.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)


user_input = input("Enter a tweet: ")
sequences = tokenizer.texts_to_sequences([user_input])
test = pad_sequences(sequences, maxlen=max_len)
prediction = model2.predict(test)
if prediction[0][0] > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")

