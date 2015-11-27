#!/usr/bin/python
#Fang Lu
#Enron poi_id.py progression and testing
#

import sys
import pickle
sys.path.append("../tools/")
from scipy.stats import pearsonr as Pearson
from scipy.stats import pointbiserialr as Biserial
import numpy as np
import operator

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import pprint

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Feature selections performed by removing non-correlated features and performing exhaustive search on remaining

#Features Selected Originally DTC(min_samples_split=15) Precision = 0.48, Recall = 0.49
#features_list = ['poi','exercised_stock_options','shared_receipt_with_poi','to_poi_ratio','expenses']

#Final Features List 
features_list = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']

#Function to get dataset statistics/Exploratory Calculates NaN
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


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


#Feature Selection and Classifier Selection are Combined in Task 4

##--First Get Correlations--##

#Creates X and Y array for correlation given the raw data from featureFormat
def getXY(corrData):
    x = []
    y = []
    for item in corrData:
        y.append( item[0] )
        x.append( item[1] )
    return y, x
            
#Calculates the Point Biserial Correlation (Pearson) of Features to 'poi'
def corrPOI(myData):
    flist = []
    for k in myData['HANNON KEVIN P']:
        flist.append(k)
    flist.remove('email_address')
    
    pbsDict = {}
    for i in flist:
        corrList = ['poi', i]      
        pbsCorr = getCorr(myData, corrList)
        pbsDict[i] = pbsCorr

    correlations = pbsDict
    #Prints the Sorted Correlations Starting with the Highest Correlation
    for w in sorted(correlations, key=correlations.get, reverse=True):
        print w, correlations[w][0], correlations[w][1]
            
    return pbsDict

#Performs Pearsons Correlation test (same as PointBiserial Mathematically)
def getCorr(myData, corrList):
    corrData = featureFormat(myData, corrList, remove_all_zeroes = False, sort_keys = True)
    y, x = getXY(corrData)
    
    #Using pearsons makes getCorr more robust for feature correlation   
    return Pearson(y,x)

#Performs correlations on between all Features Results
def corrAll(myData):
    flist = []
    for k in myData['HANNON KEVIN P']:
        flist.append(k)
    flist.remove('email_address')
    
    #Creates a dictionary to store all the correlations between features
    corrDict = {}

    for i in flist:
        corrDict[i] = {}
        for j in flist:
            corrList = [i,j]
            pbsCorr = getCorr(myData, corrList)
            corrDict[i][j] = pbsCorr
            
    #filters out highly correlated feature pairs      
    uncorr = {}
    for i in corrDict:
        uncorr[i]={}
        for j in corrDict[i]:
            if abs(corrDict[i][j][0]) <= 0.2:
                uncorr[i][j]=corrDict[i][j]
    
    return corrDict, uncorr

#Utility function for reading correlations
def readCorr(corrDict, f1, f2=None):
    print '--------'
    if f2:       
        print 'r and p-values for:',f1,'and',f2, corrDict['shared_receipt_with_poi']['to_poi_ratio']
    else:
        print 'All Correlations with ',f1
        pprint.pprint(corrDict[f1])

#Call Correlation Functions

poiCorr = corrPOI(my_dataset)
allCorr, unCorr = corrAll(my_dataset)

#Examples on How to Access the Feature Correlation Results
featureOne = 'to_poi_ratio'
featureTwo = 'exercised_stock_options'

readCorr(allCorr, featureOne, featureTwo)
readCorr(allCorr, featureOne)
readCorr(poiCorr, featureOne)
readCorr(unCorr, featureTwo)


#PCA Analysis for insight into the features
#Also creates feature set to be tested

def pcaGet(myData):
    #Builds the feature_list for all of the features
    flist = []
    for k in myData['HANNON KEVIN P']:
        flist.append(k)
    flist.remove('email_address')
    flist.remove('poi')
    flist.insert(0, 'poi')
    
    #pprint.pprint(flist)
    
    #Obtain the features in array format from featureFormat and split out 'poi'
    pcaData = featureFormat(myData, flist , remove_all_zeroes = False, sort_keys = True)
    labels, features = targetFeatureSplit(pcaData)  
    
    #Run PCA showing the first 5 components, change n_components to see more
    pca = PCA(n_components=5, whiten=False)
    pca.fit(features)
    print '-----No StandardScalling-----'
    pprint.pprint(pca.explained_variance_ratio_) 
    #uncomment to see breakdown of PC contributions by features
    #pprint.pprint(pca.components_)    
    var = pca.explained_variance_ratio_
    print 'Total Variance Captured: ', sum(var[0:5])
    #newFeatures = pca.transform(features)
    
    #With StandardScaler
    stdScaler = StandardScaler()
    scaledFeatures = stdScaler.fit_transform(features)
    pcaStd = PCA(n_components=22, whiten=True)
    pcaStd.fit(scaledFeatures)
    
    print '-----With StandardScalling-----'
    pprint.pprint(pcaStd.explained_variance_ratio_) 
    varStd = pcaStd.explained_variance_ratio_
    numPC = 14
    print 'Total Variance Captured: ', sum(varStd[0:14])
    #pprint.pprint(pcaStd.components_)
    newFeatures = pcaStd.transform(features)

    
    return var, labels, newFeatures, features


#Call PCA functions for Analysis
variance, lab, newFeat, oldFeat = pcaGet(my_dataset)

#Cross-Validation and Exhaustive Search


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

#Modified test_classifier from tester.py, to be able to reduce the folds and use different random_state
#This modified classifier also allows for preloading of labels and features, thus can perform preprocessing such as PCA
def test_classifier_mod(clf, dataset, feature_list, folds = 1000, preload = False, lab = [], feat = [], printYes = True):
    
    #Used to run preloaded feature set as in for PCA Analysis
    if preload:
        #print 'in preload'
        labels = lab
        features = feat
    else:
        data = featureFormat(dataset, feature_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
    
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    #cv = StratifiedShuffleSplit(labels, folds, random_state = None)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/(total_predictions)
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        
        #Can turn off Printing
        if printYes:
            print clf
            print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
            print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
            print ""
        
        #returns Precision and Recall for easier access to results
        return precision, recall
    except:
        print "Got a divide by zero when trying out:", clf
        precision = 0
        recall = 0
        return precision, recall

#Test Multiple Classifiers 
def test_classifiers(testOption, feat_list, use_pca_features = False):
    print 'Feature List: ',feat_list

    if testOption == 0: 
        clfvalid = GaussianNB()
        print "GuassianNB:-----"
    elif testOption == 1:
        clfvalid = DTC(min_samples_split=2)
        print "DTC:-----"
    elif testOption == 2:
        clfvalid = RFC()
        print "RFC:-----"
    elif testOption == 3:
        clfvalid = ABC(DTC())
        print "AdaBoostC:-----"
    elif testOption == 4:
        estimators = [('reduce_dim', PCA()), ('dtc', DTC())]
        clfvalid = Pipeline(estimators)
        print "PCA-DTC:-----"
    elif testOption == 5:
        estimators = [('reduce_dim', PCA(n_components=2)), ('dtc', DTC(min_samples_split=17))]
        clfvalid = Pipeline(estimators)
        print "Tuned-PCA-DTC:-----"
    
    #Option to Use PCA features
    if use_pca_features:
        pre, re = test_classifier_mod(clfvalid, my_dataset, feat_list, preload = True, lab=lab, feat = newFeat)
    else:
        pre, re = test_classifier_mod(clfvalid, my_dataset, feat_list, printYes = True)
    
    return pre, re


#Sample Call to test_classifiers
f_minPlusSharedR = ['poi',
                'exercised_stock_options',
                'to_poi_ratio',
                'shared_receipt_with_poi']

p, r = test_classifiers(4, f_minPlusSharedR)

#Feature Testing functions using K-Fold cross validation and different feature sets
def featureTest(use_ftest = False, ftest = []):
    
    #All Features
    f_all = ['poi',
             'exercised_stock_options',
             'total_stock_value', 
             'bonus', 
             'salary', 
             'to_poi_ratio', 
             'deferred_income',
             'long_term_incentive', 
             'shared_poi_ratio', 
             'restricted_stock', 
             'total_payments', 
             'shared_receipt_with_poi', 
             'loan_advances', 
             'expenses', 
             'from_poi_to_this_person', 
             'other', 
             'from_poi_ratio', 
             'from_this_person_to_poi', 
             'to_messages', 
             'restricted_stock_deferred', 
             'from_messages', 
             'deferral_payments',
             'director_fees']
    
    #13 Most Correlated Features that are Significant ~98% Confidence
    f_correlated = ['poi',
             'exercised_stock_options',
             'total_stock_value', 
             'bonus', 
             'salary', 
             'to_poi_ratio',
             'deferred_income',
             'long_term_incentive', 
             'shared_poi_ratio', 
             'restricted_stock', 
             'total_payments', 
             'shared_receipt_with_poi', 
             'loan_advances', 
             'expenses']
    
    #Financial Only
    f_financial = ['poi',
             'exercised_stock_options',
             'total_stock_value', 
             'bonus', 
             'salary', 
             'long_term_incentive', 
             'restricted_stock', 
             'total_payments', 
             'loan_advances', 
             'expenses']
    
    
    #E-mail Only
    f_email_only = ['poi',
             'to_poi_ratio', 
             'shared_poi_ratio', 
             'shared_receipt_with_poi']
    
    f_email_2 = ['poi',
             'to_poi_ratio', 
             'shared_poi_ratio', 
             'shared_receipt_with_poi']
    
    f_email_1 = ['poi',
             'to_poi_ratio']
    
    f_email_original = ['poi',
             'shared_receipt_with_poi', 
             'from_poi_to_this_person', 
             'from_this_person_to_poi', 
             'to_messages', 
             'from_messages']
    
    f_email_created = ['poi',
             'to_poi_ratio', 
             'shared_poi_ratio',  
             'from_poi_ratio']
    
    #Misc Tests, By Selecting Top Correlations for Financial and E-mail
    f_min = ['poi',
             'exercised_stock_options',
             'to_poi_ratio']
    
    f_minPlus = ['poi',
                'exercised_stock_options',
                'to_poi_ratio',
                'bonus',
                'expenses']
    
    f_minPlusExp = ['poi',
                'exercised_stock_options',
                'to_poi_ratio',
                'expenses']
    
    f_minPlusSharedR = ['poi',
                'exercised_stock_options',
                'to_poi_ratio',
                'shared_poi_ratio']
    
    f_minPlusShared = ['poi',
                'exercised_stock_options',
                'to_poi_ratio',
                'shared_receipt_with_poi']
    
    #Random Tests By Hand
    f_test = ['poi',
                'exercised_stock_options',
                'shared_receipt_with_poi']
    
    f_c_selected = ['poi',
             'exercised_stock_options', 
             'bonus',
             'to_poi_ratio',
             'deferred_income',
             'shared_receipt_with_poi',  
             'expenses']
    
    f_c_selected_2 = ['poi',
             'exercised_stock_options', 
             'bonus',
             'shared_receipt_with_poi']
    
    prStr = []
    pre = 0
    re = 0
    if use_ftest:
        for i in range(6):
            pre, re = test_classifiers(i, ftest)
            prStr.append(pre)
            prStr.append(re)
    else:
        for i in range(6):
            pre, re = test_classifiers(i, f_c_selected_2)
            prStr.append(pre)
            prStr.append(re)
            #classifer_stratified_test(i, features_list, use_pca_features = True)

    print prStr
    return prStr

#Exhaustive feature testing after selection down to 6 variables
#function tests feature sets created by removing features individually
#Produces an array of arrays of the precision and recall scores for the 6 classifiers
def featIter(num=0):
    f_c = ['poi',
             'exercised_stock_options',
             'total_stock_value', 
             'bonus', 
             'salary', 
             'to_poi_ratio',
             'deferred_income',
             'long_term_incentive', 
             'shared_poi_ratio', 
             'restricted_stock', 
             'total_payments', 
             'shared_receipt_with_poi', 
             'loan_advances', 
             'expenses']
    f_c_selected = ['poi',
             'exercised_stock_options', 
             'bonus',
             'to_poi_ratio',
             'deferred_income',
             'shared_receipt_with_poi',  
             'expenses']
    f_c_selected_2 = ['poi',
             'exercised_stock_options', 
             'bonus',
             'shared_receipt_with_poi']
    #Expense down to 3 variables
    f_sans_expense = ['poi', 'exercised_stock_options', 'bonus', 'to_poi_ratio', 'deferred_income', 'shared_receipt_with_poi']
    
    f_sans_expense_def_inc = ['poi', 'exercised_stock_options', 'bonus', 'to_poi_ratio', 'shared_receipt_with_poi']
    
    #Deferred_income down to 3 variables
    f_sans_def_inc = ['poi', 'exercised_stock_options', 'bonus', 'to_poi_ratio', 'shared_receipt_with_poi', 'expenses']
    
    f_sans_def_inc_tpr = ['poi', 'exercised_stock_options', 'bonus', 'shared_receipt_with_poi', 'expenses']
    
    #Bonus down to 3 variables
    f_sans_bonus = ['poi', 'exercised_stock_options', 'to_poi_ratio', 'deferred_income', 'shared_receipt_with_poi', 'expenses']
    
    f_sans_bonus_shared = ['poi', 'exercised_stock_options', 'to_poi_ratio', 'deferred_income', 'expenses']

    
    #Final
    f_final = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']
    
    #For performing a reverse test, by adding to Final features and see if the model improves
    f_remaining1 = ['total_stock_value', 
             'bonus', 
             'salary', 
             'to_poi_ratio', 
             'long_term_incentive', 
             'shared_poi_ratio', 
             'restricted_stock', 
             'total_payments', 
             'shared_receipt_with_poi', 
             'loan_advances',  
             'from_poi_to_this_person', 
             'other', 
             'from_poi_ratio', 
             'from_this_person_to_poi', 
             'to_messages', 
             'restricted_stock_deferred', 
             'from_messages', 
             'deferral_payments',
             'director_fees']
    
    f_remaining = []
    
    pr_Arr = []
    #Removes 
    topRemove = False
    #Test Final Feature Set by Addition of remaining features individually
    final = True
    
    
    if topRemove:
        for i in range(num):
            f_c_selected 
            f_c_selected.pop(1)
            pr = featureTest(use_ftest = True, ftest = f_c_selected)
            pr_Arr.append(pr)
    elif final:
        f_c_selected = f_final
        pr = featureTest(use_ftest = True, ftest = f_c_selected)
        pr_Arr.append(pr)
    
        for i in range(len(f_remaining)):
            f_c_selected = f_final+[f_remaining[i]]
            pr = featureTest(use_ftest = True, ftest = f_c_selected)
            pr_Arr.append(pr)
    else:
        #Change f_c_selected with desired feature list to perform removal of individual features
        f_c_selected = f_sans_bonus_shared
        pr = featureTest(use_ftest = True, ftest = f_c_selected)
        pr_Arr.append(pr)
        num = len(f_c_selected)-1
        for i in range(num):
            ftest = f_c_selected[0:(num-i)] + f_c_selected[(num-i+1):]
            pr = featureTest(use_ftest = True, ftest = ftest)
            pr_Arr.append(pr)
            print ftest
        
        
    print 'Tests Done...'
    return pr_Arr

#Sample Call to iterFeat() uses the Final feature set

pr_Arr = featIter()

#Parameter Tuning GridSearchCV and Manual

#GridSearchCV for Classifier Parameter tuning

#PCA-Decision Tree GridSeachCV
def pcadtcGrid():
    #features_list = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']
    features_list = ['poi',
                     'exercised_stock_options',
                     'to_poi_ratio',
                     'shared_receipt_with_poi']

    estimators = [('reduce_dim', PCA()), ('dtc', DTC())]
    pipe = Pipeline(estimators)

    param_grid = dict(reduce_dim__n_components=[1,2],
                      dtc__min_samples_split=np.arange(2,20))
    #print param_grid

    d = featureFormat(my_dataset, features_list, sort_keys = True)

    y, X = targetFeatureSplit(d)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, verbose=False)
    grid_search.fit(X, y)
    print '----PCA-DTC-GridSeachCV----'
    print(grid_search.best_estimator_)

#Decision Tree GridSearchCV 
def dtcGrid():
    
    features_list = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']

    estimators = [('dtc', DTC())]
    pipe = Pipeline(estimators)

    param_grid = dict(dtc__min_samples_split=np.arange(2,36))
    d = featureFormat(my_dataset, features_list, sort_keys = True)

    y, X = targetFeatureSplit(d)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, verbose=False)
    grid_search.fit(X, y)
    print '----DTC-GridSeachCV----'
    print(grid_search.best_estimator_)
    
#Manual Parameter Tuning using DTC
def paramTune(start,end):
    scores= {}
    for i in range(start,end+1):
        #Uncomment to test
        
        #Parameter Tune Pipelined classifiers
        #estimators = [('scaling', StandardScaler()),('reduce_dim', PCA()), ('dtc', DTC(min_samples_split=i*2))]
        #estimators = [('reduce_dim', PCA(n_components=2)), ('dtc', DTC(min_samples_split=i))]
        #clfIter = Pipeline(estimators)
        #clfIter.set_params(reduce_dim__n_components=3)
        
        #Paramter Tune for simple classifiers
        #clfIter = DTC(min_samples_leaf=i, min_samples_split=3)
        #clfIter = DTC(min_samples_split=3, max_depth = i)
        #test_classifier(clfIter, my_dataset, features_list)
        
        clfIter = DTC(min_samples_split=i)
        
        p,r = test_classifier_mod(clfIter, my_dataset, features_list, printYes = False)
        scores[i]=p+r
        
    print '----ParamTune----'    
    print 'Max Precision and Recall Combined Score: ', max(scores.values())
    print 'Tuned Parameter: ', max(scores.iteritems(), key=operator.itemgetter(1))[0] 
    
    return scores

#Call GridSearchCV functions
pcadtcGrid()
dtcGrid()

#Change the features to tune different feature set
#features_list = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']
start = 2
end = 36
scoreDict = paramTune(start,end)

#Function to get feature importance for DTC
def getDTCimportance(features_list):
    
    #features_list = ['poi', 'exercised_stock_options', 'deferred_income', 'expenses']
    
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
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

f_importance = getDTCimportance(features_list)
print '----------'
print 'Feature Importances: ', f_importance

#Final Classifier and Result
clf = DTC(min_samples_split=3)
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print 'Pickle Files Generated...'
