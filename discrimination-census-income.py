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

census_income.head()

# census_income[["credit_risk", "purpose", "age_in_years", "foreign_worker"]].head()

