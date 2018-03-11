"""

Module for experimenting with Themis-ml.
Implementation of discrimination and mitigation.

@author Waqar Alamgir <w.alamgir@tu-braunschweig.de>

"""

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score)
from themis_ml.preprocessing.relabelling import Relabeller
from themis_ml.meta_estimators import FairnessAwareMetaEstimator
from themis_ml.linear_model.counterfactually_fair_models import LinearACFClassifier
from themis_ml.postprocessing.reject_option_classification import SingleROClassifier
from themis_ml import datasets
from themis_ml.datasets.german_credit_data_map import preprocess_german_credit_data
from themis_ml.metrics import mean_difference, normalized_mean_difference, mean_confidence_interval

N_SPLITS = 10
N_REPEATS = 5
RANDOM_STATE = 1000

def comparison(experiment_baseline, experiment_naive, experiment_relabel, experiment_acf, experiment_single_roc):
	compare_experiments = (
	    pd.concat([
	        experiment_baseline.assign(experiment="B"),
	        experiment_naive.assign(experiment="RPA"),
	        experiment_relabel.assign(experiment="RTV"),
	        experiment_acf.assign(experiment="CFM"),
	        experiment_single_roc.assign(experiment="ROC")
	    ])
	    .assign(
	        protected_class=lambda df: df.protected_class.str.replace("_", " "),
	    )
	)
	return compare_experiments

def compare_experiment_results_multiple_model(experiment_results):
    comparison_palette = sns.color_palette("Dark2", n_colors=8)
    g = (
        experiment_results
        .query("fold_type == 'test'")
        .drop(["cv_fold"], axis=1)
        .pipe(pd.melt, id_vars=["experiment", "protected_class", "estimator",
                                "fold_type"],
              var_name="metric", value_name="score")
        .assign(
            metric=lambda df: df.metric.str.replace("_", " "))
        .pipe((sns.factorplot, "data"), y="experiment",
              x="score", hue="metric",
              col="protected_class", row="estimator",
              join=False, size=3, aspect=1.1, dodge=0.3,
              palette=comparison_palette, margin_titles=True, legend=False))
    g.set_axis_labels("mean score (95% CI)")
    for ax in g.axes.ravel():
        ax.set_ylabel("")
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    plt.legend(title="metric", loc=9, bbox_to_anchor=(-0.65, -0.4))
    g.fig.legend(loc=9, bbox_to_anchor=(0.5, -0.3))
    g.fig.tight_layout()
    g.savefig("output/fairness_aware_comparison.png", dpi=500);

def get_estemators():
	LOGISTIC_REGRESSION = LogisticRegression(
    penalty="l2", C=0.001, class_weight="balanced")
	DECISION_TREE_CLF = DecisionTreeClassifier(
	    criterion="entropy", max_depth=10, min_samples_leaf=10, max_features=10,
	    class_weight="balanced")
	RANDOM_FOREST_CLF = RandomForestClassifier(
	    criterion="entropy", n_estimators=50, max_depth=10, max_features=10,
	    min_samples_leaf=10, class_weight="balanced")
	return [
	    ("LogisticRegression", LOGISTIC_REGRESSION),
	    ("DecisionTree", DECISION_TREE_CLF),
	    ("RandomForest", RANDOM_FOREST_CLF)
	]

def generate_summary(s_x, s_y, s_z):
	experiment = pd.concat([
	    s_x,
	    s_y,
	    s_z
	])
	experiment_summary = summarize_experiment_results(experiment)
	return [experiment, experiment_summary.query("fold_type == 'test'")]


def get_estimator_name(e):
    return "".join([x for x in str(type(e)).split(".")[-1]
                    if x.isalpha()])

def get_grid_params(grid_params_dict):
    """Get outer product of grid search parameters."""
    return [
        dict(params) for params in itertools.product(
            *[[(k, v_i) for v_i in v] for
              k, v in grid_params_dict.items()])]

def fit_with_s(estimator):
    has_relabeller = getattr(estimator, "relabeller", None) is not None
    child_estimator = getattr(estimator, "estimator", None)
    estimator_fit_with_s = getattr(estimator, "S_ON_FIT", False)
    child_estimator_fit_with_s = getattr(child_estimator, "S_ON_FIT", False)
    return has_relabeller or estimator_fit_with_s or\
        child_estimator_fit_with_s

def predict_with_s(estimator):
    estimator_pred_with_s = getattr(estimator, "S_ON_PREDICT", False)
    child_estimator = getattr(estimator, "estimator", None)
    return estimator_pred_with_s or \
        getattr(child_estimator, "S_ON_PREDICT", False)

def summarize_experiment_results(experiment_df):
    return (
        experiment_df
        .drop("cv_fold", axis=1)
        .groupby(["protected_class", "estimator", "fold_type"])
        .mean())

def plot_experiment_results(experiment_results):
    return (
        experiment_results
        .query("fold_type == 'test'")
        .drop(["fold_type", "cv_fold"], axis=1)
        .pipe(pd.melt, id_vars=["protected_class", "estimator"],
              var_name="metric", value_name="score")
        .pipe((sns.factorplot, "data"), y="metric",
              x="score", hue="estimator", col="protected_class", col_wrap=3,
              size=3.5, aspect=1.2, join=False, dodge=0.4))

def cross_validation_experiment(estimators, X, y, s, s_name, verbose=True, split_by_cloumn=False):
    performance_scores = []

    if split_by_cloumn != False:
        cv = enumerate([[0, 1], [1, 2], [3, 4]])
    else:
        # stratified groups tries to balance out y and s
        groups = [i + j for i, j in
                  zip(y.astype(str), s.astype(str))]
        cv_obj = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE)
    
    for e_name, e in estimators:
        
        msg = "Training model "+e_name+" with s=" + s_name
        if verbose:
            print(msg)
            print("-" * len(msg))
        
        if split_by_cloumn == False:
            cv = enumerate(cv_obj.split(X, y, groups=groups))

        for i, (train, test) in cv:
            
            if split_by_cloumn != False:
                # create train and validation fold partitions
                X_train = X[0]
                X_test = X[1]
                y_train = y[0]
                y_test = y[1]
                s_train = s[0]
                s_test = s[1]
            else:
                # create train and validation fold partitions
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                s_train, s_test = s[train], s[test]

            # fit model and generate train and test predictions
            if fit_with_s(e):
                e.fit(X_train, y_train, s_train)
            else:
                e.fit(X_train, y_train)
                
            train_pred_args = (X_train, s_train) if predict_with_s(e) \
                else (X_train, )
            test_pred_args = (X_test, s_test) if predict_with_s(e) \
                else (X_test, )
                
            train_pred_prob = e.predict_proba(*train_pred_args)[:, 1]
            train_pred = e.predict(*train_pred_args)
            test_pred_prob = e.predict_proba(*test_pred_args)[:, 1]
            test_pred = e.predict(*test_pred_args)

            # train scores
            performance_scores.append([
                s_name, e_name, i, "train",
                # regular metrics
                roc_auc_score(y_train, train_pred_prob),

                # fairness metrics
                mean_difference(train_pred, s_train)[0],
            ])

            md = 0
            try :
                md = mean_difference(test_pred, s_test)[0]
            except ZeroDivisionError:
                md = 0.1
            # test scores
            performance_scores.append([
                s_name, e_name, i, "test",
                # regular metrics
                roc_auc_score(y_test, test_pred_prob),
                # fairness metrics
                md
            ])

            if split_by_cloumn != False:
                break
    
    return pd.DataFrame(
        performance_scores,
        columns=[
            "protected_class", "estimator", "cv_fold", "fold_type",
            "auc", "mean_diff"])