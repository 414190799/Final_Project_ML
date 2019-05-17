#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
from scipy import sparse
from sklearn.metrics import f1_score


# In[12]:


print("load data")
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
subm = pd.read_csv('./input/sample_submission.csv')
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
train_ds, validation_ds = train_test_split(train, test_size=0.1)


# In[13]:


print("build NN_SVM classifier")
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

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1, solver='sag'):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.solver = solver
        
    def predict(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs, solver=self.solver).fit(x_nb, y)
        return self


# In[14]:


N = 50000 # max trigrams
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, max_features=N)

trn_term_doc = vec.fit_transform(train_ds['question_text'])
val_term_doc = vec.transform(validation_ds['question_text'])
test_term_doc = vec.transform(test['question_text'])
print("model fit")
model = NbSvmClassifier(dual=True, solver='liblinear', C = 1e1)
model.fit(trn_term_doc, train_ds['target'])

#################################Plot
# X_train = vec.fit_transform(train_ds['question_text'])
# y_train = train_ds['target']

# X_test = vec.transform(validation_ds['question_text'])
# y_test = validation_ds['target']

# test_term_doc = vec.transform(test['question_text'])

# # train_ds, validation_ds = train_test_split(train, test_size=0.1)
# y_score = model.fit(X_train, y_train).decision_function(X_test)
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

#################################


# In[ ]:


preds_validation = model.predict_proba(val_term_doc)[:,1]
preds_test = model.predict_proba(test_term_doc)[:,1]

print("Use validation dataset to find optimal threshold")
best_threshold = threshold_search(y_true=validation_ds['target'], y_proba=preds_validation)

print("F1 Score")
best_threshold

