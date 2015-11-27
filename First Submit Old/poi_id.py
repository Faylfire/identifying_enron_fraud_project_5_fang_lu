#!/usr/bin/python
#Fang Lu
#Enron poi_id.py progression and testing
#

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import pprint

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import RandomForestClassifier as RFC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','exercised_stock_options'] #With DTC gives .32 and .36 precision and recall
#features_list = ['poi','salary', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options','long_term_incentive', 'restricted_stock']
#features_list = ['poi','salary','exercised_stock_options', 'bonus','total_stock_value']# With DTC and PCA default gives precision .43 and recall .39
#features_list = ['poi','salary','exercised_stock_options', 'bonus','total_stock_value']# With tuned DTC(15) and PCA gives .53 and .49
#feature_list= ['poi','exercised_stock_options','shared_receipt_with_poi','to_poi_ratio','expenses']#With tuned DTC(min_samples = 16) gives .48 and .49

#Features Selected for final analysis, removed all features with over 50% missing values
features_list = ['poi',
                 #'salary',
                 'exercised_stock_options',
                 #'bonus',
                 #'total_stock_value',
                 'shared_receipt_with_poi',
                 #'from_poi_ratio',
                 'to_poi_ratio',
                 'expenses',
                 #'restricted_stock',
                 #'total_payments',
                 ]
#Function to get dataset statistics/Exploratory
def getDataSetStat(data_dict, *args):
    poiCount = 0
    poilist = []
    for i in data_dict:
        if data_dict[i]['poi'] == True:
            poilist.append(i)
            poiCount += 1
    flist = []
    featureCount = len(data_dict['HANNON KEVIN P'])
    for k in data_dict['HANNON KEVIN P']:
        flist.append(k)
    
    fmissing = {}
    for l in flist:
        count = 0
        for i in data_dict:
            if data_dict[i][l] == 'NaN':
                count += 1
        fmissing[l] = count
    setDict = {}
    setDict['poiCount']= poiCount
    setDict['poilist'] = poilist
    setDict['flist'] = flist
    setDict['fmissing'] = fmissing
    setDict['featureCount'] = featureCount
    for i in args:
        if i == 0:
            pprint.pprint(setDict)
        else:
            return setDict

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

getDataSetStat(data_dict, 0)

### Task 2: Remove outliers
### Removed the obvious outlier of TOTAL from the dataset.
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
#Created the ratio features from and to poi feature divided by the total the total recieved and sent messages
for i in data_dict:
    #avoid divide by zero
    if data_dict[i]['from_messages'] != 'NaN' and data_dict[i]['from_messages'] != 0:
        data_dict[i]['to_poi_ratio'] = float(data_dict[i]['from_this_person_to_poi'])/data_dict[i]['from_messages']
    else:
        data_dict[i]['to_poi_ratio']='NaN'
    
    if data_dict[i]['to_messages'] != 'NaN' and data_dict[i]['to_messages'] != 0:
        data_dict[i]['from_poi_ratio'] = float(data_dict[i]['from_poi_to_this_person'])/data_dict[i]['to_messages']
    else:
        data_dict[i]['from_poi_ratio']='NaN'
    
    if data_dict[i]['to_messages'] != 'NaN' and data_dict[i]['to_messages'] != 0:
        data_dict[i]['shared_poi_ratio'] = float(data_dict[i]['shared_receipt_with_poi'])/data_dict[i]['to_messages']
    else:
        data_dict[i]['shared_poi_ratio']='NaN'
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=None)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def classifiertest():
    clfvalid = GaussianNB()
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "GuassianNB:-----"
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    clfvalid = DTC()
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "DTC:-----"
    print "Importance: ", clfvalid.feature_importances_
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    clfvalid = RFC()
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "RFC:-----"
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    clfvalid = ABC(DTC())
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "ABC:-----"
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    estimators = [('reduce_dim', PCA()), ('svm', DTC())]
    clfvalid = Pipeline(estimators)
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "PCA-DTC:-----"
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    estimators = estimators = [('scaling', MinMaxScaler()),('dtc', SVC())]
    clfvalid = Pipeline(estimators)
    clfvalid.fit(features_train, labels_train)
    score = clfvalid.score(features_test, labels_test)
    precision, recall = getRelevance(clfvalid)
    print "MinMax-SVC:-----"
    print "Accuracy: ", score
    print "Precision: ",precision, " Recall: ", recall
    
    

def getRelevance(clfvalid):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    predictions = clfvalid.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        else:
            true_positives += 1

    precision = 1.0*true_positives/(true_positives+false_positives+.001)
    recall = 1.0*true_positives/(true_positives+false_negatives+.001)
    return precision, recall

classifiertest()

##-------------------
##Below are tests ran with tester.py with K-fold cross-validation
##-------------------

#estimators = [('reduce_dim', PCA()), ('svm', SVC())]
#estimators = [('reduce_dim', MinMaxScaler()), ('svm', SVC())]
#clf = Pipeline(estimators)
#clf = SVC(kernel = 'linear',gamma = .01)
#clf= SVC()
clf = DTC(min_samples_split=16)
#clf = ABC(DTC())
#clf = DTC()
#clf=RFC()
#clf = RFC(n_estimators=20, min_samples_split=16)
#estimators = [('reduce_dim', PCA()), ('dtc', DTC(min_samples_split=16))]
#estimators = [('scaling', MinMaxScaler()),('reduce_dim', PCA()), ('dtc', SVC())]
#estimators = [('scaling', StandardScaler()),('reduce_dim', PCA(n_components=2)), ('dtc', DTC(min_samples_split=16))]
#clf = Pipeline(estimators)
#estimators = [('reduce_dim', PCA()), ('dtc', ABC(DTC(),n_estimators=600))]
#clf = Pipeline(estimators)
#estimators = [('reduce_dim', PCA()), ('gnb', GaussianNB())]
#clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

def iterPipe(num1,num2):
    for i in range(num1,num2+1):
        #estimators = [('scaling', StandardScaler()),('reduce_dim', PCA()), ('dtc', DTC(min_samples_split=i*2))]
        #estimators = [('reduce_dim', PCA(n_components=2)), ('dtc', DTC(min_samples_split=i))]
        #clfIter = Pipeline(estimators)
        #clfIter.set_params(reduce_dim__n_components=3)
        clfIter = DTC(min_samples_split=i)
        test_classifier(clfIter, my_dataset, features_list)
        
#get DTC feature importance 
def getDTCimportance():
    clf2 = DTC(min_samples_split=16)
    clf2.fit(features, labels)
    imp = clf2.feature_importances_
    f_importance = {}
    c = 0
    for i in features_list[1:]:
        f_importance[i]=imp[c]
        c +=1
    #pprint.pprint(f_importance)
    return f_importance


#iterPipe(2,20)
#iterPipe(14,18)
f_importance = getDTCimportance()

#Final Classifier and Result
clf = DTC(min_samples_split=16)
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)