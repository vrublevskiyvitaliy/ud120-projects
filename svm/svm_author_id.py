#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

for c in [10000]:
    print c
    clf = SVC(kernel='rbf', C=c)
    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features_test)

    print "Chris:", sum(pred)

    print "prediction time:", round(time()-t0, 3), "s"
    acu = accuracy_score(labels_test, pred)
    print acu

