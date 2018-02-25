

```python
# Install "numpy scipy matplotlib ipython pandas sympy" libraries via pip
import sys
!pip install --user numpy scipy pandas sympy
```

    Requirement already satisfied: numpy in /usr/local/lib/python2.7/dist-packages
    Requirement already satisfied: scipy in /usr/local/lib/python2.7/dist-packages
    Requirement already satisfied: pandas in /usr/local/lib/python2.7/dist-packages
    Requirement already satisfied: sympy in /usr/lib/python2.7/dist-packages
    Requirement already satisfied: pytz>=2011k in /usr/lib/python2.7/dist-packages (from pandas)
    Requirement already satisfied: python-dateutil in /usr/lib/python2.7/dist-packages (from pandas)
    Requirement already satisfied: six in /usr/local/lib/python2.7/dist-packages (from python-dateutil->pandas)



```python
# Install scikit-learn via pip
import sys
!pip install -U scikit-learn
```

    Requirement already up-to-date: scikit-learn in /usr/local/lib/python2.7/dist-packages



```python
# Install themis-ml via pip
import sys
!pip install --user themis-ml
```

    Requirement already satisfied: themis-ml in /usr/local/lib/python2.7/dist-packages/themis_ml-0.0.4-py2.7.egg
    Requirement already satisfied: scikit-learn>=0.19.1 in /usr/local/lib/python2.7/dist-packages (from themis-ml)
    Requirement already satisfied: numpy>=1.9.0 in /usr/local/lib/python2.7/dist-packages (from themis-ml)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python2.7/dist-packages (from themis-ml)
    Requirement already satisfied: pandas>=0.22.0 in /usr/local/lib/python2.7/dist-packages (from themis-ml)
    Requirement already satisfied: pathlib2 in /usr/local/lib/python2.7/dist-packages (from themis-ml)
    Requirement already satisfied: pytz>=2011k in /usr/lib/python2.7/dist-packages (from pandas>=0.22.0->themis-ml)
    Requirement already satisfied: python-dateutil in /usr/lib/python2.7/dist-packages (from pandas>=0.22.0->themis-ml)
    Requirement already satisfied: six in /usr/local/lib/python2.7/dist-packages (from pathlib2->themis-ml)
    Requirement already satisfied: scandir; python_version < "3.5" in /usr/local/lib/python2.7/dist-packages (from pathlib2->themis-ml)



```python
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
```

    Data set 1
    
    DecisionTree
    
    personal_status_and_sex:
    	AUC 	MD 	Acuracy
    B 	0.64 	0.03 	0.66
    RPA 	0.65 	0.07 	0.7
    RTV 	1.0 	0.0 	1.0
    CFM 	0.74 	0.15 	0.74
    ROC 	0.64 	-0.07 	0.66
    
    foreign_worker:
    	AUC 	MD 	Acuracy
    B 	0.64 	0.06 	0.66
    RPA 	0.64 	0.04 	0.68
    RTV 	0.66 	0.35 	0.69
    CFM 	0.75 	0.1 	0.75
    ROC 	0.64 	-0.03 	0.66
    
    Age_in_years:
    	AUC 	MD 	Acuracy
    B 	0.64 	0.02 	0.66
    RPA 	0.64 	-0.05 	0.67
    RTV 	0.71 	-0.01 	0.72
    CFM 	0.73 	-0.27 	0.73
    ROC 	0.64 	0.04 	0.66
    
    LogisticRegression
    
    personal_status_and_sex:
    	AUC 	MD 	Acuracy
    B 	0.74 	0.13 	0.74
    RPA 	0.74 	0.12 	0.74
    RTV 	0.69 	0.72 	0.7
    CFM 	0.74 	0.15 	0.74
    ROC 	0.74 	0.15 	0.72
    
    foreign_worker:
    	AUC 	MD 	Acuracy
    B 	0.74 	0.11 	0.74
    RPA 	0.74 	0.03 	0.73
    RTV 	0.78 	0.36 	0.75
    CFM 	0.75 	0.1 	0.75
    ROC 	0.74 	0.1 	0.72
    
    Age_in_years:
    	AUC 	MD 	Acuracy
    B 	0.74 	-0.01 	0.74
    RPA 	0.75 	-0.09 	0.74
    RTV 	0.81 	0.01 	0.8
    CFM 	0.73 	-0.27 	0.73
    ROC 	0.74 	-0.02 	0.72
    
    RandomForest
    
    personal_status_and_sex:
    	AUC 	MD 	Acuracy
    B 	0.71 	0.19 	0.74
    RPA 	0.71 	0.16 	0.73
    RTV 	1.0 	0.03 	0.99
    CFM 	0.74 	0.15 	0.74
    ROC 	0.71 	0.08 	0.69
    
    foreign_worker:
    	AUC 	MD 	Acuracy
    B 	0.7 	-0.04 	0.72
    RPA 	0.74 	0.01 	0.74
    RTV 	0.72 	0.33 	0.73
    CFM 	0.75 	0.1 	0.75
    ROC 	0.71 	-0.09 	0.69
    
    Age_in_years:
    	AUC 	MD 	Acuracy
    B 	0.73 	-0.13 	0.74
    RPA 	0.75 	-0.15 	0.74
    RTV 	0.75 	-0.03 	0.73
    CFM 	0.73 	-0.27 	0.73
    ROC 	0.73 	-0.08 	0.7



```python
"""

Module for experimenting with Themis-ml.
Implementation of discrimination and mitigation for census income test case.

@author Waqar Alamgir <w.alamgir@tu-braunschweig.de>

"""

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import util as u

classifiers = {'DecisionTree': tree.DecisionTreeClassifier(criterion='entropy', random_state = 100, max_depth=7, min_samples_leaf=5), 'LogisticRegression': LogisticRegression(), 'RandomForest': RandomForestClassifier()}

# Read input file
input_dir = 'data/census-income'
output_dir = 'output'
input_file = input_dir + '/census-income.numeric.data'

column_names = [
	"Age_in_years",
	"classofworker",
	"industrycode",
	"occupationcode",
	"education",
	"wageperhour",
	"enrolledineduinstlastwk",
	"maritalstatus",
	"majorindustrycode",
	"majoroccupationcode",
	"race",
	"hispanicOrigin",
	"sex",
	"memberofalaborunion",
	"reasonforunemployment",
	"fullorparttimeemploymentstat",
	"divdendsfromstocks",
	"federalincometaxliability",
	"taxfilerstatus",
	"regionofpreviousresidence",
	"stateofpreviousresidence",
	"detailedhouseholdandfamilystat",
	"detailedhouseholdsummaryinhousehold",
	"instanceweight",
	"migrationcode-changeinmsa",
	"migrationcode-changeinreg",
	"migrationcode-movewithinreg",
	"liveinthishouse1yearago",
	"migrationprevresinsunbelt",
	"numpersonsworkedforemployer",
	"totalpersonearnings",
	"countryofbirthfather",
	"countryofbirthmother",
	"countryofbirthself",
	"citizenship",
	"totalpersonincome",
	"ownbusinessorselfemployed",
	"taxableincomeamount",
	"fillincquestionnaireforveteransadmin",
	"veteransbenefits",
	"weeksworkedinyear",
	"IncomeLevel"
]

# Data set 2
print 'Data set 2'

for i in classifiers:

	print ''
	print i

	clf = classifiers[i]

	# Case 1
	print ''
	print 'race:'
	u.exeCase(input_file, column_names, 'B'  , 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)

	# Case 2
	print ''
	print 'sex:'
	u.exeCase(input_file, column_names, 'B', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)

	# Case 3
	print ''
	print 'Age_in_years:'
	u.exeCase(input_file, column_names, 'B', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 1)
	u.exeCase(input_file, column_names, 'RPA', 'Age_in_years', 'IncomeLevel', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'RTV', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'CFM', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
	u.exeCase(input_file, column_names, 'ROC', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf, 0)
```

    Data set 2
    
    DecisionTree
    
    race:
    	AUC 	MD 	Acuracy
    B 	0.92 	-0.01 	0.95
    RPA 	0.92 	-0.01 	0.95
    RTV 	0.95 	-0.01 	0.96
    CFM 	0.92 	-0.03 	0.95
    ROC 	0.92 	-0.01 	0.95
    
    sex:
    	AUC 	MD 	Acuracy
    B 	0.92 	0.03 	0.95
    RPA 	0.92 	0.05 	0.95
    RTV 	1.0 	-0.02 	0.99
    CFM 	0.92 	0.04 	0.95
    ROC 	0.92 	0.03 	0.95
    
    Age_in_years:
    	AUC 	MD 	Acuracy
    B 	0.92 	0.05 	0.95
    RPA 	0.92 	0.06 	0.95
    RTV 	1.0 	-0.0 	1.0
    CFM 	0.89 	0.06 	0.95
    ROC 	0.92 	0.05 	0.95
    
    LogisticRegression
    
    race:
    	AUC 	MD 	Acuracy
    B 	0.92 	-0.01 	0.95
    RPA 	0.92 	-0.01 	0.95
    RTV 	0.95 	0.01 	0.95
    CFM 	0.92 	-0.03 	0.95
    ROC 	0.92 	-0.01 	0.95
    
    sex:
    	AUC 	MD 	Acuracy
    B 	0.92 	0.03 	0.95
    RPA 	0.91 	0.05 	0.95
    RTV 	1.0 	-0.04 	0.98
    CFM 	0.92 	0.04 	0.95
    ROC 	0.92 	0.02 	0.95
    
    Age_in_years:
    	AUC 	MD 	Acuracy
    B 	0.92 	0.06 	0.95
    RPA 	0.92 	0.06 	0.95
    RTV 	0.99 	0.04 	0.96
    CFM 	0.89 	0.06 	0.95
    ROC 	0.92 	0.05 	0.95
    
    RandomForest
    
    race:
    	AUC 	MD 	Acuracy
    B 	0.89 	-0.01 	0.95
    RPA 	0.89 	-0.01 	0.95
    RTV 	0.9 	0.04 	0.95
    CFM 	0.92 	-0.03 	0.95
    ROC 	0.89 	-0.01 	0.94
    
    sex:
    	AUC 	MD 	Acuracy
    B 	0.89 	0.03 	0.95
    RPA 	0.88 	0.04 	0.95
    RTV 	1.0 	-0.02 	0.99

