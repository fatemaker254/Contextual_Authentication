import pandas as pd
import numpy as np
import time
from pynput import keyboard
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ===============================
# CONFIG
# ===============================

DATASET = "KeyStrokeFeaturesNew.csv"
THRESHOLD = 0.7

# ===============================
# TRAIN SVM MODELS
# ===============================

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

    clf = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )

    clf.fit(X_scaled, y)

    models[user] = {
        "model": clf,
        "scaler": scaler
    }

print("Models trained successfully.\n")

# ===============================
# REAL TIME AUTHENTICATOR
# ===============================

class LiveAuthenticator:

    def __init__(self, claimed_user):
        self.claimed_user = claimed_user
        self.events = []
        self.typed_text = ""

    def on_press(self, key):

        try:
            char = key.char
            self.typed_text += char

        except AttributeError:

            if key == keyboard.Key.space:
                self.typed_text += " "

            elif key == keyboard.Key.backspace:
                self.typed_text = self.typed_text[:-1]
                return

            elif key == keyboard.Key.enter:
                return False

            else:
                return

        timestamp = int(time.time()*1000)

        self.events.append(("Down", timestamp))

    def on_release(self, key):

        try:
            key.char
        except AttributeError:

            if key != keyboard.Key.space:
                return

        timestamp = int(time.time()*1000)

        self.events.append(("Up", timestamp))

    # ===============================
    # FEATURE EXTRACTION
    # ===============================

    def compute_features(self):

        hold_times = []
        ud_times = []
        dd_times = []

        down_times = []
        up_times = []

        for event, ts in self.events:
            if event == "Down":
                down_times.append(ts)
            else:
                up_times.append(ts)

        for i in range(min(len(down_times), len(up_times))):
            hold_times.append(up_times[i] - down_times[i])

        for i in range(1, len(down_times)):
            dd_times.append(down_times[i] - down_times[i-1])

        for i in range(1, min(len(up_times), len(down_times))):
            ud_times.append(down_times[i] - up_times[i-1])

        total_duration = down_times[-1] - down_times[0]
        typing_speed = len(self.typed_text) / (total_duration / 1000)

        features = [
            np.mean(hold_times),
            np.std(hold_times),
            np.mean(ud_times),
            np.std(ud_times),
            np.mean(dd_times),
            np.std(dd_times),
            total_duration,
            typing_speed
        ]

        return features

    # ===============================
    # AUTHENTICATION
    # ===============================

    def authenticate(self):

        if self.claimed_user not in models:
            print("User not found.")
            return

        features = self.compute_features()

        clf = models[self.claimed_user]["model"]
        scaler = models[self.claimed_user]["scaler"]

        sample_scaled = scaler.transform(
            pd.DataFrame([features], columns=feature_cols)
        )

        confidence = clf.predict_proba(sample_scaled)[0][1]

        print(f"\nConfidence: {confidence*100:.2f}%")

        if confidence >= THRESHOLD:
            print("Authentication: ACCEPTED")
        else:
            print("Authentication: REJECTED")


# ===============================
# MAIN PROGRAM
# ===============================

if __name__ == "__main__":

    claimed_user = input("Enter username: ")

    print("\nChoose authentication input:")
    print("1 → Password only")
    print("2 → Password + additional text")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        print("\nType your password and press ENTER.\n")
    else:
        print("\nType your password followed by any text and press ENTER.\n")

    auth = LiveAuthenticator(claimed_user)

    with keyboard.Listener(
        on_press=auth.on_press,
        on_release=auth.on_release
    ) as listener:
        listener.join()

    auth.authenticate()