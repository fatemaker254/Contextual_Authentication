import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)

from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ============================================================
# LOAD DATA
# ============================================================

DATASET = "KeyStrokeFeatures.csv"
df = pd.read_csv(DATASET)

feature_cols = df.columns.drop("subject")
users = df["subject"].unique()

print("\nUsers found:", users)


# ============================================================
# EER FUNCTION
# ============================================================

def compute_eer(fpr, tpr):
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


# ============================================================
# EVALUATION
# ============================================================

all_accuracies = []

plt.figure(figsize=(8,6))

for user in users:

    print("\n===============================")
    print("Evaluating User:", user)
    print("===============================")

    genuine = df[df["subject"] == user]
    imposter = df[df["subject"] != user]

    X = pd.concat([genuine, imposter])[feature_cols]
    y = np.array([1]*len(genuine) + [0]*len(imposter))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )

    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    all_accuracies.append(acc)

    print("Accuracy:", round(acc, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    TN, FP, FN, TP = cm.ravel()

    FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FRR = FN / (FN + TP) if (FN + TP) != 0 else 0

    print("FAR:", round(FAR, 4))
    print("FRR:", round(FRR, 4))

    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    eer = compute_eer(fpr, tpr)

    print("AUC:", round(roc_auc, 4))
    print("EER:", round(eer, 4))

    plt.plot(fpr, tpr,
             label=f"{user} (AUC={roc_auc:.2f}, EER={eer:.2f})")


# ============================================================
# FINAL RESULTS
# ============================================================

mean_accuracy = np.mean(all_accuracies)

print("\n=================================")
print("Average Accuracy Across Users:", round(mean_accuracy, 4))
print("=================================")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM Keystroke Authentication")
plt.legend()
plt.show()