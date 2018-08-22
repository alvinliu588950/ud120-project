#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn import tree

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print features_train.shape


#########################################################
### your code goes here ###
decision_tree_classifier = tree.DecisionTreeClassifier(min_samples_split=40)

###training start time
t_start_train = time()
print "------start training------"
###start training
decision_tree_classifier.fit(features_train, labels_train)
print "training time:", round(time()-t_start_train, 3), "s"

t_start_predict = time()
print "accuracy:", decision_tree_classifier.score(features_test, labels_test)
print "predict time:", round(time()-t_start_predict, 3), "s"
#########################################################


