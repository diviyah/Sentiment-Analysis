# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:51:56 2022

@author: ASUS
"""

#%% Deployment unusually done on another PC/mobile phone

from tensorflow.keras.models import load_model
import os
import json
import pickle

# 1. Trained Model -->
# 2. tokenizer --> loading from json
# 3. MMS/OHE --> loading from pickle


TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_sentiment.json')

# to load trained model

loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))

loaded_model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
# to load OHE
with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)
    
#%% ADDITIONAL UNDERSTANDING!

import re
from tensorflow.keras.models import load_model
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# input_review = 'The movie intrigued me, the way the it was taken. \
#                     whuii, mesmerized. I love it so much!'
    
input_review = input('type your review here')
input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()                   
    
    
tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                     maxlen=180,
                                     padding='post',
                                     truncating='post')

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

# with open(OHE_PATH,'rb') as file:
#     loaded_ohe = pickle.load(file)

print(loaded_ohe.inverse_transform(outcome)) #positive 
                # so the model has determined that input_review is positive
                # So the model is doing a good job here
                
                
