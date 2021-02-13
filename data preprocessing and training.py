# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:07:25 2021

@author: DELL
"""
import numpy as np
import pandas as pd
import gc
import sys
import joblib as jb

comments = pd.read_csv('E:/personal projects/jigsaw-toxic-comment-classification-challenge/train.csv')

comments = comments.sample(n=75000)
comments.reset_index(inplace=True, drop=True)

# sys.getsizeof(comments)

##########Preprocessing the data########################
import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = list([])

for i in range(len(comments)):
    review = re.sub('[^a-zA-Z]', ' ', comments['comment_text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

##########Creating Bag of words model####################


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(corpus).toarray()

y_toxic = comments['toxic']
y_severe_toxic = comments['severe_toxic']
y_obscene = comments['obscene']
y_threat = comments['threat']
y_insult = comments['insult']
y_identity_hate = comments['identity_hate']

def check_data_imbalance(df, cat):
    print('1'+':'+str(len(df[df[cat]==1])), '0'+':'+str(len(df[df[cat]==0])), cat) 

comment_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


for i in comment_types:
    check_data_imbalance(comments, i)
    
##As we can see that there is some imbalance in the dataset for the +ve and -ve classes in all toxicities. So we introduce class weights########

class_weights = dict()

class_weights['toxic'] = dict({1:10, 0:1})
class_weights['severe_toxic'] = dict({1:100, 0:1})
class_weights['obscene'] = dict({1:20, 0:1})
class_weights['threat'] = dict({1:350, 0:1})
class_weights['insult'] = dict({1:15, 0:1})
class_weights['identity_hate'] = dict({1:100, 0:1})


    
########Train models for 6 toxicity types#####################

from sklearn.ensemble import RandomForestClassifier

def train_models(df, comment_type, class_weight):
    model = RandomForestClassifier(class_weight=class_weight, n_estimators=300)
    print('Training model for toxicity:'+' '+str(comment_type))
    model.fit(df, comments[comment_type])
    return model


model_dict = dict()

for comment_type in comment_types:
    model_dict[comment_type] = train_models(X, comment_type, class_weights[comment_type])
    
for comment_type in model_dict.keys():
    joblib_file = 'model_'+str(comment_type)
    jb.dump(model_dict[comment_type], joblib_file)
    
    











