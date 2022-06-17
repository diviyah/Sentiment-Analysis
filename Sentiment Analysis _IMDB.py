# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:19:41 2022

@author: ASUS
"""

import pandas as pd


CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

#EDA
#STEP 1 - Data Loading

df = pd.read_csv(CSV_URL)

df_copy = df.copy()  #backup

# Let's say if we mess up the data right, 
# we can copy the df_copy again and again, so really a back up one ah! smart!
# df_copyII = df_copy.copy()

#%% STEP 2 - Data Inspection
df.head(10)
df.tail(10)

df.info()
df.describe().T #cant get anything much though due to its non-numeric


df['sentiment'].unique() #to get the unique target variables
                         #only got positive and negative sentiment
                         #Kalau use Deep Learning, need to do OHE for this

df['review'][0] #cant do much
                #but can do slicing to see
df['sentiment'][0] # 0 indicates positive reviews

df['sentiment'][3] #Negative sentiments
df['review'][1]    #Negative reviews
# Is there anything there to remove? Funny Characters! (HTML's ones), slashes
# So need to remove because later under tokenization, it will consider them as a char
# So we need to remove..

df.isna().sum() # no missing values

df.duplicated().sum() #We have 418 duplicated datas
df[df.duplicated()] #Extracting the duplicated data 

## STUFFS TO REMOVE!
# <br /> HTMLS tags have to be removed    
# Numbers can be filtered
# need to remove duplicated datas


#%% STEP 3 - Data Cleaning

#3.1 Removing duplicated datas
df = df.drop_duplicates() # so now we have (49582,2) data from (50000,2)

#3.2 Removing HTML tags
# # Method 1
# '<br /> dhgsajklfgfdhsjka <br />'.replace('br />',' ') #replacing with nothing  
# # Method 2
# df['review'][0].replace('br />',' ') #no more br already now
#                                 # need to do for loop because we need to do for all row
# Method 3 
# Even faster way - remove whatever in <>
review = df['review'].values # Features: X #Extract the values to make review=sentiment len
sentiment = df['sentiment'] # Target: y

import re

for index,rev in enumerate(review):
    # remove html tags
    # re.sub('<.*?>',rev) #? means dont be greedy
                        # so it wont convert anything anything beyong diamond bracket
                        # . means any characters
                        # * means zero or more occurences
                        # Any character except new lines (/n)
                        
    review[index] = re.sub('<.*?>',' ',rev) #'<.*?>' means we wanna remove all these
                            #' ' to replace with empty spac

                            # rev is our data
#3.3 Removing Numerics
    # convert into lower case
    # remove number
    # ^ means NOT alphabet
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()
                                # substituting that is not a-z and A-Z 
                                # .. will be replaced with a space
                                # Hence, all numeric will be removed
                                # so now we have changed every word into lower 
                                # and splitted them into a list of words
                                
review[10] #so can see all the words has been split into a list of words
    
# 'ABC def hij'.lower().split()   # this is how we split each words
# 'I am a Data Scientist'.lower().split()   
    # Why need to use enumerate?
# but review and sentiment is not equal lor!    
# review has more data than sentiment
# extract the values.. 
# BUT WHY? because the datas are linked to df together

# watch the video and relate with the concept of a, b and pop


#%% STEP 4 - Features Selection

# Nothing to select since this NLP data

#%% STEP 5 - Preprocessing
#           1) Convert into lower case - done in Data Cleaning

#           2) Tokenization
# can never guess suitable amount of vocabulary size
# do scientific way ya
# np.unique() doesnt work on NLP like it would on numeric data
# temp = df['review'].unique() # see doesnt work!
#  BUT MR WARREN RUN SOMETHING WITH temp THAT IT SHOWED 8000 DATAS
vocab_size = 10000
oov_token = 'OOV'

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) #only the reviews have to be tokenized
                               # Sentiments --> OHE
word_index = tokenizer.word_index
print(word_index)            

# so need to encode all this into numbers to fit the review
train_sequences = tokenizer.texts_to_sequences(review)     
# so all the words now are in numerics              

#           3) Padding  & Truncating
# len(train_sequences[0])
# len(train_sequences[1])
# len(train_sequences[2])
# len(train_sequences[3])  # The number of words user has commented

# for padding we can choose either mean or median

length_of_review =[len(i) for i in train_sequences]

import numpy as np

# np.mean(length_of_review) # to get the number of mean words => 238 words
np.median(length_of_review) # to get the median words => 178 words
# can use the size of the train_sequences words as well

# pick the reasonable padding value
# we are choosing median for our padding values
# Padding is to make each length to be similar
max_len = 180

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_review = pad_sequences(train_sequences,
                              maxlen=max_len,
                              padding='post',
                              truncating='post')
                                # so now all in equal length already now
                                # 1 is OOV 

#           4) One Hot Encoding for the Target - Sentiment

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

#           5) Train-test-split because this is a classification problem

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development

#Try ourself ne sollitare
# USE LSTM layers, dropout, dense, input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional, Embedding

# embedding_dim = 64
# # np.shape(X_train)[1] = 180

# model = Sequential()
# model.add(Input(shape=(np.shape(X_train)[1],1))) #11 colums from X but need to apply in tuple
# model.add(Embedding(vocab_size,embedding_dim))
# # Make sure input layer is there, then put it in embedding layer. Need to have input dimension-vocab_size
# model.add(Bidirectional(LSTM(128,return_sequences=(True))))
# # model.add(LSTM(128,return_sequences=(True)))#Once added the Bidirectional..no need LSTM anymore for the line
# model.add(Dropout(0.2))
# model.add(LSTM(128))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(np.shape(sentiment)[1], activation='softmax')) #Number of output layer=2 from y column value
#             #activation function='softmax' for output layer is for classification problem 
# model.summary()

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics='acc')

# hist=model.fit(X_train,y_train,
#                validation_data=(X_test,y_test),
#                batch_size=128,
#                epochs=10)

## for bidirectional - 
embedding_dim = 64
# np.shape(X_train)[1] = 180

model = Sequential()
model.add(Input(shape=(180))) #11 colums from X but need to apply in tuple
model.add(Embedding(vocab_size,embedding_dim))
# Make sure input layer is there, then put it in embedding layer. Need to have input dimension-vocab_size
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
# model.add(LSTM(128,return_sequences=(True)))#Once added the Bidirectional..no need LSTM anymore for the line
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(np.shape(sentiment)[1], activation='softmax')) #Number of output layer=2 from y column value
            #activation function='softmax' for output layer is for classification problem 
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

hist=model.fit(X_train,y_train,
               validation_data=(X_test,y_test),
               batch_size=128,
               epochs=10)
#%%

hist.history.keys()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(hist.history['loss'],'r--', label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

# NOW after embedding and bidirectional, 
# .. can see that training accuracy is high.. but validation acc is low!

#underfitting 
# because both losses are very small

#%% Model Evaluation
# acheive > 90% F1 score

from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


y_true = y_test
y_pred = model.predict(X_test)

#%%
y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))


# 85% after we embedded the words

# labels = ['Positive', 'Negative']
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()

#%% Model Saving

import os
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

# tokenizer_sentiment is our dictionary now

import pickle
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)


#%% Discussion/Reporting

# Discuss your result on why we couldnt reach 90% but managed to reach 84% instead
# Model acheived around 85% accuracy during training
# Recall(Sensitivity) and f1-score reported 89 and 84% respectively
# However the model starts to overfit after 2nd epoch
# Earlystopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different DL architecture can enhance the model performance as well
# For example, BERT model, transformer
# model, GPT3 model may help to improve the model


# =============================================================================
# POSSESIONS OF A GOOD DISCUSSION
# =============================================================================

# Results ---> discussion on the results
# Gives suggestions ---> how to improve your model
# Gather Evidences ---> what actually went wrong during training/model development

# How many percent was positive, how many was negative? - we didnt even check that hahaha

# (df['review'] == 'positive').sum()  #Make correction on your own!





#%% Deployment unusually done on another PC/mobile phone

from tensorflow.keras.models import load_model

loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))
loaded_model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%% ADDITIONAL UNDERSTANDING!
input_review = 'The movie intrigued me, the way the it was taken. \
                    whuii, mesmerized. I love it so much!'
    
# input_review = input('type your review here')
input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()                   
    
    
from tensorflow.keras.preprocessing.text import tokenizer_from_json

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                     maxlen=180,
                                     padding='post',
                                     truncating='post')

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)

print(ohe.inverse_transform(outcome)) #positive 
                #so the model has determined that input_review is positive
                # So the model is doing a good job here
                    
# loaded_model.predict(input_review)
#list index out of range because our input is in string not numeric

# model is saved but didnt save the tokenizer
# so save the tokenizer as well