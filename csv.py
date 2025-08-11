import pandas as pd
import numpy as np
import random

np.random.seed(42)

def generate_claim_data(n=5000):
    data = {
        "claim_id": [f"C{100000+i}" for i in range(n)],
        "claimant_age": np.random.randint(18, 80, size=n),
        "claim_amount": np.round(np.random.normal(5000, 2500, size=n), 2),
        "num_previous_claims": np.random.poisson(2, size=n),
        "claim_description": np.random.choice([
            "Car accident at intersection",
            "Stolen mobile phone claim",
            "Minor water damage in kitchen",
            "Lost luggage on flight",
            "Rear-ended on highway",
            "Unusual injury claim from treadmill"
        ], size=n),
        "policy_duration_years": np.random.randint(0, 15, size=n),
        "claim_urgency": np.random.choice(["low", "medium", "high"], size=n, p=[0.5, 0.3, 0.2]),
        "label_fraud": np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    
    # Introduce more fraud indicators
    fraud_indices = df[df["label_fraud"] == 1].index
    df.loc[fraud_indices, "claim_amount"] *= np.random.uniform(2.5, 4.5, size=len(fraud_indices))
    df.loc[fraud_indices, "claim_description"] = df.loc[fraud_indices, "claim_description"].apply(
        lambda x: x + " - suspicious details"
    )
    return df

df = generate_claim_data()
df.to_csv("synthetic_claims_data.csv", index=False)
print("✅ Synthetic data created")
