# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:53:10 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import gc
import sys
import joblib as jb

#########importing the test input and output data#####################

test_df = pd.read_csv('E:/personal projects/jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels = pd.read_csv('E:/personal projects/jigsaw-toxic-comment-classification-challenge/test_labels.csv')


#########Filtering out the data which was not used for scoring########

test_labels = test_labels[test_labels['toxic']!=-1]


comment_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for comment in comment_types:
    print(test_labels[comment].value_counts())

#########We checked here that all the data which was not used for scoring has been filtered out######

test_df = pd.merge(test_df, test_labels, how = 'right', on = 'id')
test_df = test_df.sample(n=1000)

test_df.reset_index(inplace=True)

#########We have merged the labels along with the input data and taken a sample of 1000 rows only for our test set######


import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = list([])

for i in range(len(test_df)):
    review = re.sub('[^a-zA-Z]', ' ', test_df['comment_text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(corpus).toarray()


###########Creating predicted classes for metrics calculation#############################

for comment in comment_types:
        joblib_file = 'model_'+str(comment)
        model = jb.load(joblib_file)
        test_df[str(comment)+'_pred'] = model.predict(X)
        
        
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_metrics(comment):
    print('These are the metrics for the comment: '+str(comment))
    print(confusion_matrix(test_df[comment], test_df[str(comment)+'_pred']))
    print(accuracy_score(test_df[comment], test_df[str(comment)+'_pred']))
    print(classification_report(test_df[comment], test_df[str(comment)+'_pred']))
    

##########Printing out the metrics#########################################################

for comment in comment_types:
    print_metrics(comment)









