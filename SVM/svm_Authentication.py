import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

DATASET = "KeyStrokeFeatures.csv"

df = pd.read_csv(DATASET)

feature_cols = df.columns.drop("subject")
users = df["subject"].unique()

models = {}

for user in users:

    genuine = df[df["subject"] == user]
    imposter = df[df["subject"] != user]

    X = pd.concat([genuine, imposter])[feature_cols]
    y = np.array([1]*len(genuine) + [0]*len(imposter))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf",
              probability=True,
              class_weight="balanced")

    clf.fit(X_scaled, y)

    models[user] = {
        "model": clf,
        "scaler": scaler
    }

print("Models trained.")

# ================================
# AUTHENTICATION
# ================================

def authenticate(user, sample_vector, threshold):

    if user not in models:
        print("User not found.")
        return

    clf = models[user]["model"]
    scaler = models[user]["scaler"]

    sample_scaled = scaler.transform(
        pd.DataFrame([sample_vector],
                     columns=feature_cols)
    )

    confidence = clf.predict_proba(sample_scaled)[0][1]

    print(f"Confidence: {confidence*100}%")

    if confidence >= threshold:
        print("Authentication: ACCEPTED")
    else:
        print("Authentication: REJECTED")


if __name__ == "__main__":

    claimed_user = input("Enter username: ")

    sample = []
    for col in feature_cols:
        val = float(input(f"Enter {col}: "))
        sample.append(val)

    threshold = float(input("Enter threshold (0-1): "))

    authenticate(claimed_user, sample, threshold)