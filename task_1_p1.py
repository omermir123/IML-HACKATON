import pandas as pd
X_df = pd.read_csv("train.feats.csv")
y_df = pd.read_csv("train.labels.0.csv")
df = pd.concat((X_df, y_df), axis=1)
x = df.loc[df['אבחנה-Location of distal metastases'] != '[]']
df.drop(columns=['אבחנה-Surgery date3', 'אבחנה-Surgery name3', 'אבחנה-Tumor depth', 'אבחנה-Tumor width'], inplace=True)