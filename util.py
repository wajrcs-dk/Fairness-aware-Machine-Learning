import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from themis_ml.preprocessing import relabelling as pp
from themis_ml.metrics import (mean_difference, normalized_mean_difference)
from themis_ml.linear_model import LinearACFClassifier
from themis_ml.postprocessing import reject_option_classification as ppd

def calculateAuc(case, clf, x_test, y_test, s_test):
	# Calculates AUC
	if case == 'ROC':
		prob = clf.predict_proba(x_test, s_test)[:,1]
	else:
		if case == 'CFM':
			prob = clf.predict_proba(x_test, s_test)[:,1]
		else:
			prob = clf.predict_proba(x_test)[:,1]
	lb = preprocessing.LabelBinarizer()
	y_test_bin = lb.fit_transform(y_test)
	return metrics.roc_auc_score(y_test_bin, prob)

def meanDifference(y_train, y_pred, s_train, s_test):
	# Calculates MD
	md_y_true = mean_difference(y_train, s_train)[0]
	md_y_pred = mean_difference(y_pred, s_test)[0]
	return md_y_pred - md_y_true

def accuracyScore(y_test, y_pred):
	# Calucates Accuracy
	return accuracy_score(y_test, y_pred)

def printResult(case, auc, md, ac):
	# Prints result
	print case
	print 'AUC:', auc
	print 'MD:', md
	print 'Acuracy:', ac

# entropy | gini
def exeCase(inputFile, columnNames, case, sClass, pClass, usecols, clf):
	dataframe = pd.read_csv(inputFile, header=0, names=columnNames)
	dataX = pd.read_csv(inputFile, header=0, usecols=usecols)

	x = np.array(dataX)
	y = np.array(dataframe[pClass])
	s = np.array(dataframe[sClass])

	if sClass == 'race':
		s[s >= 1] = 1
	elif sClass == 'personal_status_and_sex':
		s[s == 0] = 0
		s[s == 1] = 1
		s[s == 2] = 0
		s[s == 3] = 0
		s[s == 4] = 1
	elif sClass == 'Age_in_years':
		s[s <= 25] = 0
		s[s > 25] = 1

	if case == 'RTV':
		massager = pp.Relabeller(ranker=tree.DecisionTreeClassifier(criterion='gini', random_state = 100, max_depth=7, min_samples_leaf=5))
		# obtain a new set of labels
		y = massager.fit(x, y, s).transform(x)

	# Do the slpitting
	x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=0.5, random_state=42)

	if case == 'CFM':
		clf = LinearACFClassifier()
		y_pred = clf.fit(x_train, y_train, s_train).predict(x_test, s_test)
	elif case == 'ROC':
		clf = ppd.SingleROClassifier(estimator=tree.DecisionTreeClassifier(criterion='gini', random_state = 100, max_depth=7, min_samples_leaf=5))
		y_pred = clf.fit(x_train, y_train).predict(x_test, s_test)
	else:
		y_pred = clf.fit(x_train, y_train).predict(x_test)

	# Calculating AUC
	auc = calculateAuc(case, clf, x_test, y_test, s_test)

	# Calculating MD
	md = meanDifference(y_train, y_pred, s_train, s_test)

	# Calucating Accuracy
	ac = accuracyScore(y_test, y_pred)

	printResult(case, auc, md, ac)
