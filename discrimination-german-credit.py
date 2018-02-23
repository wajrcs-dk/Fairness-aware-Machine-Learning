"""

Module for experimenting with Themis-ml.
Implementation of discrimination and mitigation for german credit test case.

@author Waqar Alamgir <w.alamgir@tu-braunschweig.de>

"""

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import util as u

classifiers = {
	'DecisionTree': tree.DecisionTreeClassifier(criterion='entropy', random_state = 100, max_depth=7, min_samples_leaf=5),
	'LogisticRegression': LogisticRegression(),
	'RandomForest': RandomForestClassifier()
}

# Read input file
input_dir = 'data/german-credit'
output_dir = 'output'
input_file = input_dir + '/german-credit.numeric.data'

column_names = [
	"Status_of_existing_checking_account",
	"Duration_in_month",
	"Credit_history",
	"Purpose",
	"Credit_amount",
	"Savings_account/bonds",
	"Present_employment_since",
	"Installment_rate_in_percentage_of_disposable_income",
	"personal_status_and_sex",
	"Other_debtors_guarantors",
	"Present_residence_since",
	"Property",
	"Age_in_years",
	"Other_installment_plans",
	"Housing",
	"Number_of_existing_credits_at_this_bank",
	"Job",
	"Number_of_people_being_liable_to_provide_maintenance_for",
	"Telephone",
	"foreign_worker",
	"credit_risk"
]

# Data set 1
print 'Data set 1'

for i in classifiers:

	print ''
	print i

	clf = classifiers[i]

	# Case 1
	print ''
	print 'personal_status_and_sex:'
	
	u.exeCase(input_file, column_names, 'B', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19], clf, 0)

	# Case 2
	print ''
	print 'foreign_worker:'
	u.exeCase(input_file, column_names, 'B', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	
	# Case 3
	print ''
	print 'Age_in_years:'
	u.exeCase(input_file, column_names, 'B', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf, 0)



