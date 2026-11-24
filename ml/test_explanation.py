
if __package__:
    from .explainability import CardiofyExplainer
else:
    from explainability import CardiofyExplainer

engine = CardiofyExplainer()

patient = {
    "age": 62,
    "sex": 1,
    "cp": 4,
    "trestbps": 150,
    "chol": 260,
    "fbs": 1,
    "restecg": 2,
    "thalch": 125,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 1,
    "thal": 7
}

result = engine.explain_patient(patient)

print("Model:", result["model"])

print(
    f"Predicted probability: "
    f"{result['probability']:.2%}"
)

print(
    f"Base probability: "
    f"{result['base_probability']:.2%}"
)

print("\nFEATURE CONTRIBUTIONS\n")

for item in result["contributions"]:

    print(
        f"{item['label']:25}"
        f"{item['impact_percentage_points']:+.3f} pp"
    )

print(
    "\nSHAP additivity error:",
    result["additivity_error"]
)

assert result["additivity_error"] < 1e-5

print("\nExplanation test passed.")
print(result["disclaimer"])
