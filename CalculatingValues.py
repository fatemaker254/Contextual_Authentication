#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock
from sklearn.metrics import roc_curve

# Read data
data = pd.read_csv("Collecting_keyStorke.csv")

# Clean column names (safety fix)
data.columns = data.columns.str.strip()

userList = data['user'].unique()
keyList = data['key'].unique()

# Store rows in list (instead of append)
rows = []

for i in range(len(userList)):
    for j in range(len(keyList)):

        queryData = data.query(
            "user=='" + userList[i] + "' and key==" + str(keyList[j]) + " and key >=33 and key<=122"
        )

        queryLen = len(queryData)

        if queryLen > 0:
            finalData = {}

            if queryLen > 2:
                for k in range(0, queryLen - 1, 2):

                    finalData = {}
                    finalData['subject'] = userList[i]
                    finalData['key'] = chr(int(keyList[j]))

                    finalData['H'] = (
                        int(queryData.iloc[k+1].Time) - int(queryData.iloc[k].Time)
                    ) / 1000

                    keyUpIndex = queryData.iloc[k+1].name

                    if keyUpIndex + 1 < len(data) and data.iloc[keyUpIndex + 1].user == userList[i]:
                        finalData['UD'] = (
                            int(data.iloc[keyUpIndex+1].Time) - int(queryData.iloc[k+1].Time)
                        ) / 1000

                        finalData['DD'] = (
                            int(data.iloc[keyUpIndex+1].Time) - int(queryData.iloc[k].Time)
                        ) / 1000
                    else:
                        finalData['UD'] = finalData['H']
                        finalData['DD'] = finalData['H']

                    rows.append(finalData)

            else:
                finalData = {}
                finalData['subject'] = userList[i]
                finalData['key'] = chr(int(keyList[j]))
                print(queryData['keyEvent'].unique())

                up_rows = queryData.loc[queryData['keyEvent'] == 'Up', 'Time']
                if len(up_rows) == 0:
                    continue
                upTime = up_rows.iloc[0]
                print(queryData['keyEvent'].unique())

                # upTime = queryData.query("keyEvent=='Up'").Time.values[0]
                downTime = queryData.query("keyEvent=='Down'").Time.values[0]

                finalData['H'] = (int(upTime) - int(downTime)) / 1000

                keyUpIndex = queryData.query("keyEvent=='Up'").index[0]

                if keyUpIndex + 1 < len(data) and data.iloc[keyUpIndex + 1].user == userList[i]:
                    finalData['UD'] = (
                        int(data.iloc[keyUpIndex+1].Time) - int(upTime)
                    ) / 1000

                    finalData['DD'] = (
                        int(data.iloc[keyUpIndex+1].Time) - int(downTime)
                    ) / 1000
                else:
                    finalData['UD'] = finalData['H']
                    finalData['DD'] = finalData['H']

                rows.append(finalData)

# Create DataFrame once
df = pd.DataFrame(rows, columns=['subject','key','H','UD','DD'])

# Save CSV directly (no manual csv writer needed)
df.to_csv("KeyStrokeDistance.csv", index=False)

# Print specific user
print(df[df['subject'] == 'soham'])

# Grouped data
groupedDf = df[df['subject'] == 'soham'].groupby(
    ['subject','key','H','UD','DD']
).all()

print(groupedDf)
