import pandas as pd
import numpy as np

def preprocessing(feats_file, labels_file):
    X_df = pd.read_csv(feats_file)
    y_df = pd.read_csv(labels_file)
    df = pd.concat((X_df, y_df), axis=1)
    x = df.loc[df['אבחנה-Location of distal metastases'] != '[]']
    df.drop(columns=['אבחנה-Surgery date3', 'אבחנה-Surgery name3', 'אבחנה-Surgery name1',
                     'אבחנה-Tumor depth', "אבחנה-Diagnosis date", 'אבחנה-Tumor width', "אבחנה-Her2"], inplace=True)

    df.drop_duplicates(
        subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'],
        inplace=True)
    node_exam_avg = pd.DataFrame(df["אבחנה-Nodes exam"]).mean(skipna=True, numeric_only=True)
    df["אבחנה-Nodes exam"].where(df["אבחנה-Nodes exam"].notnull(), node_exam_avg)

    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].str[:2]
    df["אבחנה-Histopatological degree"].replace(["G1", "G2", "G3", "G4", "GX", "Nu"], [1, 2, 3, 4, 0, 0], inplace=True)

    df["User Name"] = df["User Name"].str.split('_')
    df["User Name"] = df["User Name"].str[0]

    df["אבחנה-T -Tumor mark (TNM)"].fillna("XX", inplace=True)
    df["אבחנה-T -Tumor mark (TNM)"] = df["אבחנה-T -Tumor mark (TNM)"].str[:2]
    df["אבחנה-T -Tumor mark (TNM)"].replace(["T1", "T2", "T3", "T4"], [1, 2, 3, 4], inplace=True)
    tumor_num_avg = pd.DataFrame(df["אבחנה-T -Tumor mark (TNM)"]).mean(skipna=True, numeric_only=True)
    df["אבחנה-T -Tumor mark (TNM)"].where(df["אבחנה-T -Tumor mark (TNM)"].applymap(lambda x: str(x).isdigit()), tumor_num_avg)

    pd.get_dummies(df, columns=["אבחנה-Basic stage", "Hospital", "User Name"])

    print(df)
if __name__ == '__main__':
    preprocessing("train.feats.csv", "train.labels.0.csv")