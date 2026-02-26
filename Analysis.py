#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics import accuracy_score
import numpy as np


plt.style.use('ggplot')

DATASET_1 = 'KeyStrokeDistance.csv' #My dataset
DATASET_2 = 'keystroke.csv'


# ============================================================
# Utility Functions
# ============================================================

def compute_eer(fpr, tpr):
    if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
        return None
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


def get_feature_columns(df):
    return [col for col in df.columns
            if col.startswith("H")
            or col.startswith("DD")
            or col.startswith("UD")]


# ============================================================
# Distance-Based Authentication
# ============================================================

class DistanceDetector:

    def __init__(self, subjects, data, metric="manhattan"):
        self.subjects = subjects
        self.data = data
        self.metric = metric

    def _distance(self, x, mean):
        if self.metric == "manhattan":
            return cityblock(x, mean)
        else:
            return euclidean(x, mean)

    def evaluate(self):

        feature_columns = get_feature_columns(self.data)

        all_user_scores = []
        all_imposter_scores = []

        for subject in self.subjects:

            genuine_data = self.data.loc[
                self.data.subject == subject, feature_columns
            ]

            imposter_data = self.data.loc[
                self.data.subject != subject, feature_columns
            ]

            if len(genuine_data) < 8:
                continue

            split_index = int(len(genuine_data) * 0.7)

            if split_index < 1 or split_index >= len(genuine_data):
                continue

            train = genuine_data.iloc[:split_index]
            test_genuine = genuine_data.iloc[split_index:]

            if test_genuine.empty:
                continue

            mean_vector = train.mean().values

            # Genuine attempts
            for i in range(test_genuine.shape[0]):
                score = self._distance(
                    test_genuine.iloc[i].values,
                    mean_vector
                )
                all_user_scores.append(score)

            # Imposter attempts (5 samples per other user)
            imposter_samples = (
                self.data[self.data.subject != subject]
                .groupby("subject")
                .head(5)[feature_columns]
            )

            for i in range(imposter_samples.shape[0]):
                score = self._distance(
                    imposter_samples.iloc[i].values,
                    mean_vector
                )
                all_imposter_scores.append(score)

        if len(all_user_scores) == 0 or len(all_imposter_scores) == 0:
            raise ValueError("Insufficient data for ROC computation.")

        labels = [0]*len(all_user_scores) + [1]*len(all_imposter_scores)
        scores = all_user_scores + all_imposter_scores

        return roc_curve(labels, scores)


# ============================================================
# KNN Classification (Multi-class -> OvR ROC)
# ============================================================

def load_data(file_name, drop_cols):
    df = pd.read_csv(file_name)
    features = df.drop(columns=drop_cols)
    y = df["subject"].values
    return features, y


def calculate_knn_roc(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)

    # One-vs-rest ROC (micro-average)
    y_test_bin = pd.get_dummies(y_test).values

    fpr, tpr, thresholds = roc_curve(
        y_test_bin.ravel(),
        y_prob.ravel()
    )
    eer = compute_eer(fpr, tpr)
    threshold = thresholds[np.nanargmin(np.absolute(fpr - (1 - tpr)))]

# Convert scores to predictions
    predictions = [1 if s >= threshold else 0 for s in y_prob.ravel()]

    accuracy = accuracy_score(y_test_bin.ravel(), predictions)

    print("Accuracy for KNN:", accuracy)

    return fpr, tpr


# ============================================================
# Evaluate Dataset 1
# ============================================================

data1 = pd.read_csv(DATASET_1)
subjects1 = data1["subject"].unique()

# Manhattan
fpr1_1, tpr1_1, = DistanceDetector(subjects1, data1, "manhattan").evaluate()
eer1_1 = compute_eer(fpr1_1, tpr1_1)

# Euclidean
fpr3_1, tpr3_1, _ = DistanceDetector(subjects1, data1, "euclidean").evaluate()
eer3_1 = compute_eer(fpr3_1, tpr3_1)

# KNN
X1, y1 = load_data(DATASET_1, ['subject', 'key'])
fpr2_1, tpr2_1 = calculate_knn_roc(X1, y1)
eer2_1 = compute_eer(fpr2_1, tpr2_1)


# ============================================================
# Evaluate Dataset 2
# ============================================================

data2 = pd.read_csv(DATASET_2)
subjects2 = data2["subject"].unique()

# Manhattan
fpr1_2, tpr1_2, _ = DistanceDetector(subjects2, data2, "manhattan").evaluate()
eer1_2 = compute_eer(fpr1_2, tpr1_2)

# Euclidean
fpr3_2, tpr3_2, _ = DistanceDetector(subjects2, data2, "euclidean").evaluate()
eer3_2 = compute_eer(fpr3_2, tpr3_2)

# KNN
X2, y2 = load_data(DATASET_2, ['subject', 'sessionIndex', 'rep'])
fpr2_2, tpr2_2 = calculate_knn_roc(X2, y2)
eer2_2 = compute_eer(fpr2_2, tpr2_2)


# ============================================================
# Plot Results
# ============================================================

plt.figure(figsize=(10,6))
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr1_1, tpr1_1, label=f'Set1 Manhattan AUC={auc(fpr1_1,tpr1_1):.3f} EER={eer1_1:.3f}')
plt.plot(fpr1_2, tpr1_2, label=f'Set2 Manhattan AUC={auc(fpr1_2,tpr1_2):.3f} EER={eer1_2:.3f}')

plt.plot(fpr3_1, tpr3_1, label=f'Set1 Euclidean AUC={auc(fpr3_1,tpr3_1):.3f} EER={eer3_1:.3f}')
plt.plot(fpr3_2, tpr3_2, label=f'Set2 Euclidean AUC={auc(fpr3_2,tpr3_2):.3f} EER={eer3_2:.3f}')

plt.plot(fpr2_1, tpr2_1, label=f'Set1 KNN AUC={auc(fpr2_1,tpr2_1):.3f} EER={eer2_1:.3f}')
plt.plot(fpr2_2, tpr2_2, label=f'Set2 KNN AUC={auc(fpr2_2,tpr2_2):.3f} EER={eer2_2:.3f}')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


print("\nManhattan EER:", eer1_1, eer1_2)
print("Euclidean EER:", eer3_1, eer3_2)
print("KNN EER:", eer2_1, eer2_2)
