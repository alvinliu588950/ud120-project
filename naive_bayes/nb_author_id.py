#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn.naive_bayes import GaussianNB
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


nb_classifier = GaussianNB()

###training start time
t_start_train = time()

###start training
nb_classifier.fit(features_train, labels_train)
print "training time:", round(time()-t_start_train, 3), "s"

t_start_predict = time()
print "accuracy:", nb_classifier.score(features_test, labels_test)
print "predict time:", round(time()-t_start_predict, 3), "s"
