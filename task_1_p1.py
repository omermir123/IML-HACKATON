import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def preprocessing(X, y=None):
    if y:
        # Organizing labels
        unique_set = {"PUL", "HEP", "PLE", "PER", "SKI", "OTH", "LYM", "BON",
                      "ADR", "MAR", "BRA"}
        for name in unique_set:
            y[name] = y["אבחנה-Location of distal metastases"].map(
                lambda x: 1 if name in x else 0)
    X.drop_duplicates(
        subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'],
        inplace=True) # TODO maybe add NAME FORM
    X.drop(columns=['אבחנה-Surgery date3', 'אבחנה-Surgery name3', 'אבחנה-Surgery name1',
                     'אבחנה-Tumor depth', "אבחנה-Diagnosis date", 'אבחנה-Tumor width', "אבחנה-Her2"], inplace=True)
    # making  "surgery before or after-Actual activity" to dummies
    X["surgery before or after-Actual activity"] = X[
        "surgery before or after-Actual activity"].fillna("")
    surgery_before_diagnosis_name = ["למפקטומי", "בלוטות", "מסטקטומי", "כריתה", "קרינה"]  # todo check the הוצ
    for name in surgery_before_diagnosis_name:
        X[name] = X["surgery before or after-Actual activity"].map(
            lambda x: 1 if name in x else 0)
    X.drop(columns=["surgery before or after-Actual activity"])

    # managing "אבחנה-Nodes exam"
    node_exam_avg = pd.DataFrame(X["אבחנה-Nodes exam"]).mean(skipna=True, numeric_only=True)
    X["אבחנה-Nodes exam"].where(X["אבחנה-Nodes exam"].notnull(), node_exam_avg)

    # managing "אבחנה-Histopatological degree"
    X["אבחנה-Histopatological degree"] = X["אבחנה-Histopatological degree"].str[:2]
    X["אבחנה-Histopatological degree"].replace(["G1", "G2", "G3", "G4", "GX", "Nu"], [1, 2, 3, 4, 0, 0], inplace=True)

    # managing "User Name"
    X["User Name"] = X["User Name"].str.split('_')
    X["User Name"] = X["User Name"].str[0]

    # managing "אבחנה-er"
    negaives = ["neg", "NEG", "-", "שלילי"]
    X["אבחנה-er"] = X["אבחנה-er"].apply(
        lambda x: -1 if any(s in str(x) for s in negaives) else 1)

    # managing "אבחנה-T -Tumor mark (TNM)"
    X["אבחנה-T -Tumor mark (TNM)"].fillna("XX", inplace=True)
    X["אבחנה-T -Tumor mark (TNM)"] = X["אבחנה-T -Tumor mark (TNM)"].str[:2]
    tumor_num_avg = pd.DataFrame(X["אבחנה-T -Tumor mark (TNM)"]).mean(skipna=True, numeric_only=True)
    X["אבחנה-T -Tumor mark (TNM)"].replace(["T1", "T2", "T3", "T4", "Ti", "Not yet Established", "Tx", "XX"], [1, 2, 3, 4, 0, 0, tumor_num_avg, tumor_num_avg], inplace=True)
    X["אבחנה-T -Tumor mark (TNM)"].where(not X["אבחנה-T -Tumor mark (TNM)"].str.isnumeric(), tumor_num_avg)
    pd.get_dummies(X, columns=["אבחנה-Basic stage", " Hospital", "User Name"])

    return X, y

def k_nn_imputation(X, y, feat_name):
    k_nn = KNeighborsClassifier(n_neighbors=10)
    X_fit = X[X[feat_name].notnull()]

def load_data(feats_file, labels_file):
    X_df = pd.read_csv(feats_file)
    y_df = pd.read_csv(labels_file)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2,
                                                        random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,
                                                      test_size=0.1,
                                                      random_state=21)
    X_train, y_train = preprocessing(X_train, y_train)
    return X_train, y_train, X_dev, y_dev, X_test ,y_test

if __name__ == '__main__':
    load_data("train.feats.csv", "train.labels.0.csv")