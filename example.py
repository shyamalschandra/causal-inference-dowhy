import dowhy
from dowhy.do_why import CausalModel
import dowhy.datasets

if __name__ == "__main__":
    data = dowhy.datasets.linear_dataset(beta=10,
                                         num_common_causes=5,
                                         num_instruments=2,
                                         num_samples=10000,
                                         treatment_is_binary=True)
    # Create a causal model from the data and given graph.
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["dot_graph"])

    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression")
    print("Causal Estimate is " + str(estimate.value))

    # Adding a random common cause variable
    res_random = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause")
    print(res_random)

    # Replacing treatment with a random (placebo) variable
    res_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
    print(res_placebo)

    # Removing a random subset of the data
    res_subset = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.9)
    print(res_subset)
