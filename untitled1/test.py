#coding:utf-8

import pandas as pd

train_df = pd.read_pickle("soma_goods_train.df") # images load

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.externals import joblib
from konlpy.utils import pprint
from collections import OrderedDict
from konlpy.tag import Twitter
import re

vectorizer = CountVectorizer()
hannanum = Twitter()

d_list = []
cate_list = []
tmp_list = []
r_list = []

for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    d_list.append(each[1]['name'])

    #print "origin :" + each[1]['name']
    tmp = re.sub('[/,.+☆●■╋┏━%+┓!@#\]$^&?*(\[)_♣┕▶◀┙ΛΟ▩]', ' ', each[1]['name'])
    tmp_list = hannanum.morphs(tmp)
    tmp_list = OrderedDict.fromkeys(tmp_list)
    tmp = " ".join(tmp_list)

    #print "re :" + tmp
    #print each[1]['name']

    r_list.append(tmp)

    cate_list.append(cate)



# #print "original : " + d_list[0]/,.+☆●■╋┏━┓
#
# temp = hannanum.nouns(d_list[0])
# #pprint(temp)
#
# temp = OrderedDict.fromkeys(temp)
#
# result = " ".join(temp)
# result = re.sub('[!@#$%^&*()_+-]','',result)
# #print result


print len(set(cate_list))

cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))

print cate_dict[u'디지털/가전;네트워크장비;KVM스위치']
print cate_dict[u'패션의류;남성의류;정장']

y_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    y_list.append(cate_dict[cate])

x_list = vectorizer.fit_transform(r_list)

svc_param = {'C':np.logspace(-2,0,20)}
gs_svc = GridSearchCV(LinearSVC(loss='squared_hinge'),svc_param,cv=5,n_jobs=4)
gs_svc.fit(x_list, y_list)

print gs_svc.best_params_, gs_svc.best_score_
clf = LinearSVC(C=gs_svc.best_params_['C'])
clf.fit(x_list,y_list)

joblib.dump(clf,'classify.model',compress=3)
joblib.dump(cate_dict,'cate_dict.dat',compress=3)
joblib.dump(vectorizer,'vectorizer.dat',compress=3)

