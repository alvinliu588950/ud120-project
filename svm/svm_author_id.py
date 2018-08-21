    #!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time
from sklearn.svm import SVC
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### speed up traning speed by using smaller traning set
"""
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
"""
###

#########################################################
svm_classifier = SVC(kernel='rbf', C=10000.0)

###training start time
t_start_train = time()
print "------start training------"
###start training
svm_classifier.fit(features_train, labels_train)
print "training time:", round(time()-t_start_train, 3), "s"

t_start_predict = time()
predictions = svm_classifier.predict(features_test)
print "accuracy:", svm_classifier.score(features_test, labels_test)
print "predict time:", round(time()-t_start_predict, 3), "s"

print "some prediction of elements"
print "10:" , predictions[10]
print "26:" , predictions[26]
print "50:" , predictions[50]

unique, counts = np.unique(predictions, return_counts=True)

print "All classes predictions counts:"
print dict(zip(unique, counts))
