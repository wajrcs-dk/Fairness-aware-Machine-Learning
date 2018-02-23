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