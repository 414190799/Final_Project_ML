#!/usr/bin/env python
# coding: utf-8

# In[3]:

# some functions cited from Anti Novartis's code
import pandas as pd
import numpy as np
import re, string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy import sparse
from sklearn.metrics import f1_score


# In[4]:


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

class SvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1, solver='sag'):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.solver = solver
        
    def predict(self, x):
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(x)

    def predict_proba(self, x):
        check_is_fitted(self, ['_clf'])
        return self._clf.predict_proba(x)

    def fit(self, x, y):
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs, solver=self.solver).fit(x, y)
        return self


# In[5]:


N = 50000 # max trigrams
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, max_features=N)

trn_term_doc = vec.fit_transform(train_ds['question_text'])
val_term_doc = vec.transform(validation_ds['question_text'])
test_term_doc = vec.transform(test['question_text'])
print("model fit")
# model = SVC(C = 1.0)
model = SvmClassifier(dual=True, solver='liblinear', C = 1e1)
model.fit(trn_term_doc, train_ds['target'])

# ####################################Grid Search
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# scores = ['precision', 'recall']

# from sklearn.model_selection import GridSearchCV
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#     clf = GridSearchCV(SvmClassifier(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(trn_term_doc, train_ds['target'])

#     print("Best parameters set found on development set:")
#     print()
    
#     print(clf.best_params_)
    
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
    
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
              
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)

#     print(classification_report(y_true, y_pred))
# #########

preds_validation = model.predict_proba(val_term_doc)[:,1]
preds_test = model.predict_proba(test_term_doc)[:,1]

print("Use validation dataset to find optimal threshold")
best_threshold = threshold_search(y_true=validation_ds['target'], y_proba=preds_validation)

print("F1 Score")
best_threshold


# In[6]:


# pred_test_y = (preds_test > best_threshold['threshold']).astype(int)
# test_df = pd.read_csv("./input/test.csv", usecols=["qid"])
# out_df = pd.DataFrame({"qid":test_df["qid"].values})
# out_df['prediction'] = pred_test_y
# out_df.to_csv("submission.csv", index=False)

