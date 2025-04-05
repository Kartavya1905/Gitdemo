# Project 1 Solution 
# Name: Kartavya Mandora

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ("travel", "disease"),
    ("immune", "disease"),
    ("immune", "lab_test"),
    ("disease", "fever"),
    ("disease", "lab_test")
])

# Define the Conditional Probability Distributions (CPDs)

cpd_travel = TabularCPD(variable="travel",
                      variable_card=2,
                      values=[[0.2], [0.8]],
                      state_names={"travel": ["yes", "no"]})

cpd_immune = TabularCPD(variable="immune",
                      variable_card=2,
                      values=[[0.7], [0.3]],
                      state_names={"immune": ["strong", "weak"]})

cpd_disease = TabularCPD(variable="disease",
                             variable_card=2,
                             values=[[0.1, 0.01, 0.3, 0.05],   # P(disease=present | immune, travel)
                                     [0.9, 0.99, 0.7, 0.95]],  # P(disease=absent  | immune, travel)
                             evidence=["immune", "travel"],
                             evidence_card=[2,2],
                             state_names={"disease": ["present", "absent"],
                                           "immune": ["strong", "weak"],
                                           "travel": ["yes", "no"]
                                           })

cpd_fever = TabularCPD(variable="fever",
                       variable_card=3,
                       values=[
                            [0.7, 0.05],  # P(fever=high | disease)
                            [0.3, 0.15],  # P(fever=low | disease)
                            [0.0, 0.8]    # P(fever=none | disease)
                       ],
                       evidence=["disease"],
                       evidence_card=[2],
                       state_names={"fever": ["high", "low", "none"],
                                     "disease": ["present", "absent"]
                                     })

cpd_lab_test = TabularCPD(variable="lab_test",
                             variable_card=2,
                             values=[[0.95, 0.7, 0.02, 0.1],   # P(lab_test=positive | disease, immune)
                                     [0.05, 0.3, 0.98, 0.9]],  # P(lab_test=negative | disease, immune)
                             evidence=["disease", "immune"], evidence_card=[2,2],
                             state_names={"lab_test": ["positive", "negative"],
                                           "disease": ["present", "absent"],
                                           "immune": ["strong", "weak"]
                                           })

# Add CPDs to the model
model.add_cpds(cpd_travel, cpd_immune, cpd_disease, cpd_fever, cpd_lab_test)

# Validate the model
assert model.check_model()

# Perform inference
inference = VariableElimination(model)
result = inference.query(variables=["disease"],
                         evidence={"fever": "low", "lab_test": "positive"})

# Print the results
print(result)


print(f"\nProbability of (D=P | F=L, L=+) is {result.values[0]:.4f}")