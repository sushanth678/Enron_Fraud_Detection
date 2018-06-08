#!/usr/bin/python


import pickle
import csv 
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV 
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


################################################################################
###Preprocessing functions

"""

#Use only to upload datafile to Google Colaboratory
from google.colab import files
uploaded = files.upload()

"""

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features

################################################################################
###Preprocessing
#Target Labels
classes = ['NON_POI', 'POI']                                       
### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', "r") as data_file:
    data_dict = pickle.load(data_file)
features_list = []
#To collect all the features in the data
for k,v in data_dict.items():         
    features_list.extend(v.keys())
    break
#Omitting Bugs
features_list.insert(0,features_list.pop(features_list.index('poi')))
features_list.pop(features_list.index('email_address'))
features_list.pop(features_list.index('shared_receipt_with_poi'))
#Converting data into suitable format
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#Feature Selection using KBest
features_new = SelectKBest(k=15).fit_transform(features, labels)
################################################################################
# Split into a training and testing set
features_train, features_test, labels_train, labels_test = train_test_split(features_new, labels, test_size=0.1)
################################################################################
###Processing
#Principal Component Analysis
pca = PCA(svd_solver='randomized', n_components=2).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

#Parameters to tune classifiers
param_grid_svc = [{'C': range(1,1000,100), 'gamma': range(1,1000,100), 'kernel': ['rbf', 'linear', 'linear_svc', 'polynomial']}]
param_grid_knn = [{'n_neighbors' : range(1,15), 'weights' : ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}]
param_grid_rf = [{'n_estimators' : range(1,15), 'criterion' : ['gini', 'entropy']}]

clf_gnb = GaussianNB()
clf_svc = SVC()
clf_dt = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier()
#Tuning the classifiers
clf_svc_tuned = GridSearchCV(clf_svc, param_grid_svc)
clf_knn_tuned = GridSearchCV(clf_knn, param_grid_knn)
clf_rf_tuned = GridSearchCV(clf_rf, param_grid_rf)
#Training the classifiers
clf_gnb.fit(features_train_pca,labels_train)
clf_svc.fit(features_train_pca,labels_train)
clf_dt.fit(features_train_pca,labels_train)
clf_knn_tuned.fit(features_train_pca,labels_train)
clf_rf_tuned.fit(features_train_pca,labels_train)
#Predicting the labels
pred_gnb = clf_gnb.predict(features_test_pca)
pred_svc = clf_svc.predict(features_test_pca)
pred_dt = clf_dt.predict(features_test_pca)
pred_knn = clf_knn_tuned.predict(features_test_pca)
pred_rf = clf_rf_tuned.predict(features_test_pca)

#F1_score
f1_score_all = {}
f1_score_all['clf_gnb'] = f1_score(labels_test, pred_gnb)
f1_score_all['clf_svc'] = f1_score(labels_test, pred_svc)
f1_score_all['clf_dt'] = f1_score(labels_test, pred_dt)
f1_score_all['clf_knn_tuned'] = f1_score(labels_test, pred_knn)
f1_score_all['clf_rf_tuned'] = f1_score(labels_test, pred_rf)
#Finding the best Classifier
classifiers = {'clf_gnb':'GaussianNB', 'clf_svc':'SVC', 'clf_dt': 'DecisionTreeClassifier', 'clf_knn_tuned':'KNearestNeighbors', 'clf_rf_tuned':'RandomForest'}
best_classifier = max(f1_score_all.iteritems(), key=operator.itemgetter(1))[0]
best_classifier = classifiers[best_classifier]
#dump_classifier_and_data(clf, data_dict, features_list)

################################################################################
###Scoring and Plotting and Displaying
print "Best classifier:",best_classifier
#GaussianNaiveBayes
visualizer_gnb = ClassificationReport(clf_gnb, classes = classes)
visualizer_gnb.fit(features_train_pca,labels_train)  # Fit the visualizer and the model
visualizer_gnb.score(features_test_pca,labels_test)  # Evaluate the model on the test data
g = visualizer_gnb.poof()
#SVM
visualizer_svc = ClassificationReport(clf_svc, classes = classes)
visualizer_svc.fit(features_train_pca,labels_train)  # Fit the visualizer and the model
visualizer_svc.score(features_test_pca,labels_test)  # Evaluate the model on the test data
g = visualizer_svc.poof()
#DecisionTree
visualizer_dt = ClassificationReport(clf_dt, classes = classes)
visualizer_dt.fit(features_train_pca,labels_train)  # Fit the visualizer and the model
visualizer_dt.score(features_test_pca,labels_test)  # Evaluate the model on the test data
g = visualizer_dt.poof()
#KNearestNeighbors
visualizer_knn = ClassificationReport(clf_knn_tuned, classes=classes)
visualizer_knn.fit(features_train_pca,labels_train)  # Fit the visualizer and the model
visualizer_knn.score(features_test_pca,labels_test)  # Evaluate the model on the test data
print "                    KNearestNeighborsClassifier"
g = visualizer_knn.poof()
#RandomForest
visualizer_rf = ClassificationReport(clf_rf_tuned, classes=classes)
visualizer_rf.fit(features_train_pca,labels_train)  # Fit the visualizer and the model
visualizer_rf.score(features_test_pca,labels_test)  # Evaluate the model on the test data
print "                    RandomForestClassifier"
g = visualizer_rf.poof()

print 'True Labels:',labels_test
###Classification reports in table format
#print classification_report(labels_test, pred_gnb, target_names = classes), classification_report(labels_test, pred_svc, target_names = classes), classification_report(labels_test, pred_dt, target_names = classes), classification_report(labels_test, pred_knn, target_names = classes), classification_report(labels_test, pred_rf, target_names = classes)
################################################################################
