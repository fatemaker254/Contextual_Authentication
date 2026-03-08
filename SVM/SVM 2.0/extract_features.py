import pandas as pd
import numpy as np

data = pd.read_csv(
    "raw_keystrokes.csv",
    names=["user","session","key","event","Time"]
)

data = data.sort_values(by=["user","session","Time"])

rows = []

for (user, session), session_df in data.groupby(["user","session"]):

    down = session_df[session_df["event"] == "Down"]
    up = session_df[session_df["event"] == "Up"]

    if len(down) < 3 or len(up) < 3:
        continue

    H, UD, DD = [], [], []

    for i in range(min(len(down), len(up)) - 1):

        hold = (up.iloc[i]["Time"] - down.iloc[i]["Time"]) / 1000
        H.append(hold)

        ud = (down.iloc[i+1]["Time"] - up.iloc[i]["Time"]) / 1000
        UD.append(ud)

        dd = (down.iloc[i+1]["Time"] - down.iloc[i]["Time"]) / 1000
        DD.append(dd)

    total_duration = (
        down.iloc[-1]["Time"] - down.iloc[0]["Time"]
    ) / 1000

    typing_speed = len(H) / total_duration if total_duration > 0 else 0

    row = {
        "subject": user,

        "mean_H": np.mean(H),
        "std_H": np.std(H),

        "mean_UD": np.mean(UD),
        "std_UD": np.std(UD),

        "mean_DD": np.mean(DD),
        "std_DD": np.std(DD),

        "total_duration": total_duration,
        "typing_speed": typing_speed
    }

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("KeyStrokeFeatures.csv", index=False)

print("Feature extraction complete.")
print(df.head())