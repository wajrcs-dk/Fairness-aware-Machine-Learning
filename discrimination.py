from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import util as u

classifiers = {'DecisionTree': tree.DecisionTreeClassifier(criterion='entropy', random_state = 100, max_depth=7, min_samples_leaf=5), 'LogisticRegression': LogisticRegression(), 'RandomForest': RandomForestClassifier()}

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
	u.exeCase(input_file, column_names, 'B', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'RPA', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'RTV', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'CFM', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'ROC', 'personal_status_and_sex', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)

	# Case 2
	print ''
	print 'foreign_worker:'
	u.exeCase(input_file, column_names, 'B', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'RPA', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], clf)
	u.exeCase(input_file, column_names, 'RTV', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'CFM', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'ROC', 'foreign_worker', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)

	# Case 3
	print ''
	print 'Age_in_years:'
	u.exeCase(input_file, column_names, 'B', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'RPA', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'RTV', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'CFM', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)
	u.exeCase(input_file, column_names, 'ROC', 'Age_in_years', 'credit_risk', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], clf)


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
print ''
print 'Data set 2'

for i in classifiers:

	print ''
	print i

	clf = classifiers[i]

	# Case 1
	print ''
	print 'race:'
	u.exeCase(input_file, column_names, 'B'  , 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RPA', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RTV', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'CFM', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'ROC', 'race', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)

	# Case 2
	print ''
	print 'sex:'
	u.exeCase(input_file, column_names, 'B', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RPA', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RTV', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'CFM', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'ROC', 'sex', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)

	# Case 3
	print ''
	print 'Age_in_years:'
	u.exeCase(input_file, column_names, 'B', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RPA', 'Age_in_years', 'IncomeLevel', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'RTV', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'CFM', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)
	u.exeCase(input_file, column_names, 'ROC', 'Age_in_years', 'IncomeLevel', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], clf)