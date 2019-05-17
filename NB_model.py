#!/usr/bin/env python
# coding: utf-8

# In[1]:


# some functions cited from Anti Novartis's code
import pandas as pd
import numpy as np
import re, string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
from sklearn.metrics import f1_score


# In[ ]:


print("load data")
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
subm = pd.read_csv('./input/sample_submission.csv')
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
train_ds, validation_ds = train_test_split(train, test_size=0.1)

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[ ]:


N = 50000 # max trigrams
vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, max_features=N)

trn_term_doc = vec.fit_transform(train_ds['question_text'])
val_term_doc = vec.transform(validation_ds['question_text'])
test_term_doc = vec.transform(test['question_text'])
print("model fit")
model = MultinomialNB()
model.fit(trn_term_doc, train_ds['target'])

preds_validation = model.predict_proba(val_term_doc)[:,1]
preds_test = model.predict_proba(test_term_doc)[:,1]

print("Use validation dataset to find optimal threshold")
best_threshold = threshold_search(y_true=validation_ds['target'], y_proba=preds_validation)

print("F1 Score")
best_threshold

