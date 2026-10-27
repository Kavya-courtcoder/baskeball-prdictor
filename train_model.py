import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Set seed for reproducible synthetic data generation
np.random.seed(42)

# Generate synthetic shot data for 160 subjects (5 shots each)
n_subjects = 160
shots_per_subject = 5

data = []
for subject_id in range(n_subjects):
    # Assign each subject a realistic baseline shooting percentage (30% - 80%)
    p_success = np.random.uniform(0.30, 0.80)
    shots = np.random.binomial(1, p_success, shots_per_subject)
    
    # 3-shot sliding window: Shots 1,2,3 predict Shot 4; Shots 2,3,4 predict Shot 5
    for i in range(len(shots) - 3):
        window = shots[i:i+3]
        target = shots[i+3]
        data.append(list(window) + [target])

columns = ['Shot_1', 'Shot_2', 'Shot_3', 'Next_Shot']
df = pd.DataFrame(data, columns=columns)

# Separate features (X) and target label (y)
X = df[['Shot_1', 'Shot_2', 'Shot_3']]
y = df['Next_Shot']

# Train Logistic Regression classifier
model = LogisticRegression()
model.fit(X, y)

# Save the trained model to a file for Streamlit to use
with open('basketball_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Data generated for 160 subjects successfully!")
print(f"Total training window samples: {len(X)}")
print("Model saved as 'basketball_model.pkl'")