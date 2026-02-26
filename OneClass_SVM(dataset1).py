import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


print("===== STEP 1: Loading Dataset =====")

# Load dataset
df = pd.read_csv("KeyStrokeDistance.csv")

print("Dataset Loaded:\n", df)
print("\nShape of dataset:", df.shape)

# -----------------------------------------
# STEP 2: Extract Features
# -----------------------------------------

print("\n===== STEP 2: Extracting Features =====")
target_user = "soham"
genuine_df = df[df['subject'] == target_user]

X = genuine_df[['H','UD','DD']].values
# X = df[['H', 'UD', 'DD']].values

print("Raw Feature Matrix:\n", X)

# -----------------------------------------
# STEP 3: Normalize Features
# -----------------------------------------

print("\n===== STEP 3: Scaling Features =====")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled Features:\n", X_scaled)
print("Feature Mean After Scaling:", np.mean(X_scaled, axis=0))
print("Feature Std After Scaling:", np.std(X_scaled, axis=0))

# -----------------------------------------
# STEP 4: Train One-Class SVM
# -----------------------------------------

print("\n===== STEP 4: Training One-Class SVM =====")

model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
model.fit(X_scaled)

print("Model Training Completed")

# -----------------------------------------
# STEP 5: Authentication Function
# -----------------------------------------

def authenticate(H, UD, DD):
    print("\n===== AUTHENTICATION ATTEMPT =====")
    print("Input Values:")
    print("H =", H)
    print("UD =", UD)
    print("DD =", DD)

    sample = np.array([[H, UD, DD]])

    print("\nRaw Sample:", sample)

    # Scale input
    sample_scaled = scaler.transform(sample)

    print("Scaled Sample:", sample_scaled)

    # Prediction
    prediction = model.predict(sample_scaled)

    print("Raw Model Output:", prediction)

    # Decision
    if prediction[0] == 1:
        print("✅ AUTHENTICATED")
    else:
        print("❌ ACCESS DENIED")


# -----------------------------------------
# STEP 6: Test Authentication
# -----------------------------------------

# Try genuine-like sample
authenticate(0.076,0.019,0.095)

# Try abnormal sample
# authenticate(0.30, 0.20, 0.50)
# authenticate(0.137,0.003,0.14)
authenticate(0.061,0.055,0.116)

print("\n===== STEP 5: ROC & AUC Evaluation =====")

# Genuine samples (soham)
genuine_test = df[df['subject'] == target_user][['H','UD','DD']].values

# Impostor samples (all others)
impostor_test = df[df['subject'] != target_user][['H','UD','DD']].values

print("Number of Genuine Samples:", len(genuine_test))
print("Number of Impostor Samples:", len(impostor_test))

# Scale using same scaler
genuine_test_scaled = scaler.transform(genuine_test)
impostor_test_scaled = scaler.transform(impostor_test)

# Get decision scores
genuine_scores = model.decision_function(genuine_test_scaled)
impostor_scores = model.decision_function(impostor_test_scaled)

print("\nSample Genuine Scores:", genuine_scores[:5])
print("Sample Impostor Scores:", impostor_scores[:5])

# Combine
y_true = np.concatenate([np.ones(len(genuine_scores)),
                         np.zeros(len(impostor_scores))])

y_scores = np.concatenate([genuine_scores, impostor_scores])

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

auc_value = roc_auc_score(y_true, y_scores)

print("\nAUC Score:", auc_value)

# Compute EER
fnr = 1 - tpr
eer_threshold_index = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_threshold_index]

print("EER:", eer)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Keystroke Authentication")
plt.legend()
plt.show()