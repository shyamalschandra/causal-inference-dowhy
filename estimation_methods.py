import numpy as np
import pandas as pd
import logging

import dowhy
from dowhy.do_why import CausalModel
import dowhy.datasets


def regression(model, identified_estimand):
    causal_estimate_reg = model.estimate_effect(identified_estimand,
    method_name="backdoor.linear_regression",
    test_significance=True)
    print(causal_estimate_reg)
    print("Causal Estimate is " + str(causal_estimate_reg.value))


def stratification(model, identified_estimand):
    causal_estimate_strat = model.estimate_effect(identified_estimand,
    method_name="backdoor.propensity_score_stratification")
    print(causal_estimate_strat)
    print("Causal Estimate is " + str(causal_estimate_strat.value))


def matching(model, identified_estimand):
    causal_estimate_match = model.estimate_effect(identified_estimand,
    method_name="backdoor.propensity_score_matching")
    print(causal_estimate_match)
    print("Causal Estimate is " + str(causal_estimate_match.value))

def weighting(model, identified_estimand):
    causal_estimate_ipw = model.estimate_effect(identified_estimand,
    method_name="backdoor.propensity_score_weighting")
    print(causal_estimate_ipw)
    print("Causal Estimate is " + str(causal_estimate_ipw.value))

def instrumental_variable(model, identified_estimand):
    causal_estimate_iv = model.estimate_effect(
        identified_estimand,
        method_name="iv.instrumental_variable",
        method_params={'iv_instrument_name':'Z1'})

    print(causal_estimate_iv)
    print("Causal Estimate is " + str(causal_estimate_iv.value))


def regression_discontinuity(model, identified_estimand):
    causal_estimate_regdist = model.estimate_effect(
        identified_estimand,
        method_name="iv.regression_discontinuity",
        method_params={
            'rd_variable_name':'Z1',
            'rd_threshold_value':0.5,
            'rd_bandwidth': 0.1})

    print(causal_estimate_regdist)
    print("Causal Estimate is " + str(causal_estimate_regdist.value))


if __name__ == "__main__":
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=5,
        num_instruments=2,
        num_samples=10000,
        treatment_is_binary=True)

    # With graph
    model = CausalModel(
        data=data['df'],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["dot_graph"],
        instruments=data["instrument_names"],
        logging_level=logging.INFO)

    model.view_model()
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    regression(model, identified_estimand)
