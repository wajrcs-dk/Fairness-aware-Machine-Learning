"""

Module for experimenting with Themis-ml.
Implementation of discrimination and mitigation for census income test case.

@author Waqar Alamgir <w.alamgir@tu-braunschweig.de>

"""

# Importing all libraries
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from themis_ml.preprocessing.relabelling import Relabeller
from themis_ml.meta_estimators import FairnessAwareMetaEstimator
from themis_ml.linear_model.counterfactually_fair_models import LinearACFClassifier
from themis_ml.postprocessing.reject_option_classification import SingleROClassifier
from themis_ml import datasets
from themis_ml.datasets.census_income_data_map import preprocess_census_income_data
import util as u

census_income = datasets.census_income(True)
data = census_income[["income_gt_50k", "citizenship", "marital_stat", "education"]].head()
print data

census_income_preprocessed = (
    preprocess_census_income_data(census_income)
    # .assign(foreign=lambda df: df["foreign_born-_not_a_citizen_of_u_s"])
    .assign(age_below_25=lambda df: df["age"] <= 25)
    .assign(widowed=lambda df: df["widowed"])
)

print 'income_gt_50k'
income_gt_50k = census_income_preprocessed.income_gt_50k
income_gt_50k.value_counts()

print 'widowed'
is_widowed = census_income_preprocessed.widowed
is_widowed.value_counts()

print 'foreign'
is_foreign = census_income_preprocessed.foreign_worker
is_foreign.value_counts()

print 'age_below_25'
age_below_25 = census_income_preprocessed.age_below_25
age_below_25.value_counts()

# Establish Baseline Metrics

# specify feature set. Note that we're excluding the `is_widowed`
# and `age_below_25` columns that we created above.

feature_set_1 = [
    'age',
    'class_of_worker',
    'detailed_industry_recode',
    'detailed_occupation_recode',
    'education',
    'wage_per_hour',
    'enroll_in_edu_inst_last_wk',
    'marital_stat',
    'major_industry_code',
    'major_occupation_code',
    'race',
    'hispanic_origin',
    'sex',
    'member_of_a_labor_union',
    'reason_for_unemployment',
    'full_or_part_time_employment_stat',
    'capital_gains',
    'capital_losses',
    'dividends_from_stocks',
    'tax_filer_stat',
    'region_of_previous_residence',
    'state_of_previous_residence',
    'detailed_household_and_family_stat',
    'detailed_household_summary_in_household',
    'instance_weight',
    'migration_code_change_in_msa',
    'migration_code-change_in_reg',
    'migration_code-move_within_reg',
    'live_in_this_house_1_year_ago',
    'migration_prev_res_in_sunbelt',
    'num_persons_worked_for_employer',
    'family_members_under_18',
    'country_of_birth_father',
    'country_of_birth_mother',
    'country_of_birth_self',
    'citizenship',
    'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veteran\'s_admin',
    'veterans_benefits',
    'weeks_worked_in_year',
    'year'
]

# 'income_gt_50k',
# 'dataset_partition'

#################################################
# Case 1: Baseline
#################################################

# training and target data
X = census_income_preprocessed[feature_set_1].values
y = census_income_preprocessed["income_gt_50k"].values
s_widowed = census_income_preprocessed["widowed"].values
s_foreign = census_income_preprocessed["foreign_worker"].values
s_age_below_25 = census_income_preprocessed["age_below_25"].values

estimators = u.get_estemators()

experiment_baseline_widowed = u.cross_validation_experiment(estimators, X, y, s_widowed, "widowed")
experiment_baseline_foreign = u.cross_validation_experiment(estimators, X, y, s_foreign, "foreign_worker")
experiment_baseline_age_below_25 = u.cross_validation_experiment(estimators, X, y, s_age_below_25, "age_below_25")

experiment_baseline = u.generate_summary(experiment_baseline_widowed, experiment_baseline_foreign, experiment_baseline_age_below_25)
print experiment_baseline[1]

u.plot_experiment_results(experiment_baseline[0])

#################################################
# Naive Fairness-aware Approach: Remove Protected Class
#################################################

# create feature sets that remove variables with protected class information
feature_set_no_sex = [
    f for f in feature_set_1 if
    f not in [
        'personal_status_and_sex_widowed_divorced/separated/married',
        'personal_status_and_sex_male_divorced/separated',
        'personal_status_and_sex_male_married/widowed',
        'personal_status_and_sex_male_single']]
feature_set_no_foreign = [f for f in feature_set_1 if f != "foreign_worker"]
feature_set_no_age = [f for f in feature_set_1 if f != "age"]

# training and target data
X_no_sex = census_income_preprocessed[feature_set_no_sex].values
X_no_foreign = census_income_preprocessed[feature_set_no_foreign].values
X_no_age = census_income_preprocessed[feature_set_no_age].values

experiment_naive_widowed = u.cross_validation_experiment(estimators, X_no_sex, y, s_widowed, "widowed")
experiment_naive_foreign = u.cross_validation_experiment(estimators, X_no_foreign, y, s_foreign, "foreign_worker")
experiment_naive_age_below_25 = u.cross_validation_experiment(estimators, X_no_age, y, s_age_below_25, "age_below_25")

experiment_naive = u.generate_summary(experiment_naive_widowed, experiment_naive_foreign, experiment_naive_age_below_25)
print experiment_naive[1]

u.plot_experiment_results(experiment_naive[0])

#################################################
# Fairness-aware Method: Relabelling
#################################################

# here we use the relabeller class to create new y vectors for each of the
# protected class contexts.

# we also use the FairnessAwareMetaEstimator as a convenience class to
# compose together different fairness-aware methods. This wraps around the
# estimators that we defined in the previous
relabeller = Relabeller()
relabelling_estimators = [
    (name, FairnessAwareMetaEstimator(e, relabeller=relabeller))
    for name, e in estimators]

experiment_relabel_widowed = u.cross_validation_experiment(relabelling_estimators, X_no_sex, y, s_widowed, "widowed")
experiment_relabel_foreign = u.cross_validation_experiment(relabelling_estimators, X_no_foreign, y, s_foreign, "foreign_worker")
experiment_relabel_age_below_25 = u.cross_validation_experiment(relabelling_estimators, X_no_age, y, s_age_below_25, "age_below_25")

experiment_relabel = u.generate_summary(experiment_relabel_widowed, experiment_relabel_foreign, experiment_relabel_age_below_25)
print experiment_relabel[1]

u.plot_experiment_results(experiment_relabel[0])

#################################################
# Fairness-aware Method: Additive Counterfactually Fair Model
#################################################

LINEAR_REG = LinearRegression()
DECISION_TREE_REG = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10)
RANDOM_FOREST_REG = RandomForestRegressor(
    n_estimators=50, max_depth=10, min_samples_leaf=10)

# use the estimators defined above to define the linear additive
# counterfactually fair models
linear_acf_estimators = [
    (name, LinearACFClassifier(
         target_estimator=e,
         binary_residual_type="absolute"))
    for name, e in estimators]

experiment_acf_widowed = u.cross_validation_experiment(linear_acf_estimators, X_no_sex, y, s_widowed, "widowed")
experiment_acf_foreign = u.cross_validation_experiment(linear_acf_estimators, X_no_foreign, y, s_foreign, "foreign_worker")
experiment_acf_age_below_25 = u.cross_validation_experiment(linear_acf_estimators, X_no_age, y, s_age_below_25, "age_below_25")

experiment_acf = u.generate_summary(experiment_acf_widowed, experiment_acf_foreign, experiment_acf_age_below_25)
print experiment_acf[1]

u.plot_experiment_results(experiment_acf[0])

#################################################
# Reject-option Classification
#################################################

# use the estimators defined above to define the linear additive
# counterfactually fair models
single_roc_clf_estimators = [
    (name, SingleROClassifier(estimator=e))
    for name, e in estimators]

experiment_single_roc_widowed = u.cross_validation_experiment(single_roc_clf_estimators, X_no_sex, y, s_widowed, "widowed")
experiment_single_roc_foreign = u.cross_validation_experiment(single_roc_clf_estimators, X_no_foreign, y, s_foreign, "foreign_worker")
experiment_single_roc_age_below_25 = u.cross_validation_experiment(single_roc_clf_estimators, X_no_age, y, s_age_below_25, "age_below_25")

experiment_single_roc = u.generate_summary(experiment_single_roc_widowed, experiment_single_roc_foreign, experiment_single_roc_age_below_25)
print experiment_single_roc[1]

u.plot_experiment_results(experiment_single_roc[0])

#################################################
# Comparison
#################################################

compare_experiments = u.comparison(experiment_baseline, experiment_naive, experiment_relabel, experiment_acf, experiment_single_roc)

u.compare_experiment_results_multiple_model(compare_experiments.query("estimator == 'LogisticRegression'"))