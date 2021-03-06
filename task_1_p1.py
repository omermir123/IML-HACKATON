import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import re


def clean_stage(stage):
    stage_dict = {'1': 1, '2a': 2, '2': 2, '2b': 3, 'Not yet Established': 1,
                  '0': 0, '3a': 4, '3b': 5, '4': 6}
    stage = stage.replace('Stage', '')
    if stage[0] in stage_dict.keys() or stage[:2] in stage_dict.keys():
        return stage_dict[stage[0]] if stage[0] in stage_dict.keys() else stage_dict[stage[:2]]
    return 0


def preprocessing(X):
    # making  "surgery before or after-Actual activity" to dummies
    X["surgery before or after-Actual activity"] = X["surgery before or after-Actual activity"].fillna("")
    surgery_before_diagnosis_name = ["למפקטומי", "בלוטות", "מסטקטומי", "כריתה", "קרינה"]  # todo check the הוצ
    for name in surgery_before_diagnosis_name:
        X[name] = X["surgery before or after-Actual activity"].map(lambda x: 1 if name in x else 0)
    X.drop(columns=["surgery before or after-Actual activity"], inplace=True)

    # managing "אבחנה-Nodes exam"
    node_exam_avg = pd.DataFrame(X["אבחנה-Nodes exam"]).mean(skipna=True, numeric_only=True)
    X["אבחנה-Nodes exam"].where(X["אבחנה-Nodes exam"].notnull(), int(node_exam_avg), inplace=True)

    # managing "אבחנה-Histopatological degree" #todo make more generic
    X["אבחנה-Histopatological degree"] = X["אבחנה-Histopatological degree"].str[:2]
    X["אבחנה-Histopatological degree"].replace(["G1", "G2", "G3", "G4", "GX", "Nu"], [1, 2, 3, 4, 0, 0], inplace=True)

    # managing stage
    X["אבחנה-Stage"] = X["אבחנה-Stage"].fillna("NULL")
    X["אבחנה-Stage"] = X["אבחנה-Stage"].apply(lambda x: clean_stage(x))

    # managing "User Name"
    # X["User Name"] = X["User Name"].str.split('_')
    # X["User Name"] = X["User Name"].str[0]

    # managing "אבחנה-er"
    negaives = ["neg", "NEG", "-", "שלילי"]
    X["אבחנה-er"] = X["אבחנה-er"].apply(
        lambda x: -1 if any(s in str(x) for s in negaives) else 1)

    # managing "אבחנה-pr"
    negaives = ["neg", "NEG", "-", "שלילי"]
    X["אבחנה-pr"] = X["אבחנה-pr"].apply(
        lambda x: -1 if any(s in str(x) for s in negaives) else 1)

    # managing "אבחנה-KI67 protein"
    X["אבחנה-KI67 protein"] = X["אבחנה-KI67 protein"].apply(
        lambda cell: re.search('\d+', str(cell)).group(
            0) if cell is not None and re.search('\d+',
                                                 str(cell)) is not None else '50')
    X["אבחנה-KI67 protein"] = X["אבחנה-KI67 protein"].astype(int)
    X["אבחנה-KI67 protein"] = X["אבחנה-KI67 protein"].apply(
        lambda x: 1 if x > 20 else 0)

    # managing "אבחנה-M -metastases mark (TNM)"
    X["אבחנה-M -metastases mark (TNM)"] = X["אבחנה-M -metastases mark (TNM)"].str[:2]
    X.loc[X['אבחנה-M -metastases mark (TNM)'] != 'M1', 'אבחנה-M -metastases mark (TNM)'] = 0
    X.loc[X['אבחנה-M -metastases mark (TNM)'] == 'M1', 'אבחנה-M -metastases mark (TNM)'] = 1

    # managing "אבחנה-T -Tumor mark (TNM)"
    X["אבחנה-T -Tumor mark (TNM)"].fillna("XX", inplace=True)
    X["אבחנה-T -Tumor mark (TNM)"] = X["אבחנה-T -Tumor mark (TNM)"].str[:2]
    X["אבחנה-T -Tumor mark (TNM)"].replace(["T1", "T2", "T3", "T4", "Ti", "No", "T0"], [1, 2, 3, 4, 0, 0, 0],
                                           inplace=True)

    # managing "אבחנה-Surgery name1"
    surgery_name = ["LUMPECTOMY", "SENTINEL NODE BIOPSY", "LESION", "AXILLARY DISSECTION",
                    "MASTECTOMY", "QUADRANTECTOMY", "SENTINEL NODE BIOPSY", "LYMPH NODE BIOPSY",
                    "AXILLARY"]
    X["אבחנה-Surgery name1"] = X["אבחנה-Surgery name1"].fillna('')
    for name in surgery_name:
        X[name] = X["אבחנה-Surgery name1"].apply(lambda x: 1 if name in x else 0)
    X.drop(columns=["אבחנה-Surgery name1"], inplace=True)
    #####
    # cols_to_remove = []
    #
    # for col in X.columns:
    #     try:
    #         _ = X[col].astype(float)
    #     except ValueError:
    #         # print('Couldn\'t covert %s to float' % col)
    #         if col != "אבחנה-T -Tumor mark (TNM)":
    #             cols_to_remove.append(col)
    #         pass

    # keep only the columns in df that do not contain string
    # X = X[[col for col in X.columns if col not in cols_to_remove]]
    # X.drop(columns=["אבחנה-Positive nodes", "אבחנה-Surgery sum"], inplace=True)
    #####
    markers = ["Tx", "MF", "XX"]
    X_fit = X.loc[X["אבחנה-T -Tumor mark (TNM)"].notnull()].copy()
    y_fit = X_fit["אבחנה-T -Tumor mark (TNM)"]
    for mark in markers:
        X_fit = X_fit[y_fit != mark]
        y_fit = y_fit[y_fit != mark]
    X_fit.drop(columns=["אבחנה-T -Tumor mark (TNM)"], inplace=True)

    for marker in markers:
        X_pred = X.loc[X["אבחנה-T -Tumor mark (TNM)"] == marker].copy()
        X_pred.drop(columns=["אבחנה-T -Tumor mark (TNM)"], inplace=True)
        pred = k_nn_imputation(X_fit, y_fit.astype(int), X_pred)
        X["אבחנה-T -Tumor mark (TNM)"].replace(marker, int(pred.mean()), inplace=True)

    return X


def k_nn_imputation(X_fit, y_fit, X_pred):
    k_nn = KNeighborsClassifier(n_neighbors=10).fit(X_fit.to_numpy(), y_fit.to_numpy())
    return k_nn.predict(X_pred.to_numpy())


def load_data(feats_file, labels_file):
    X_df = pd.read_csv(feats_file)
    y_df = pd.read_csv(labels_file)
    # Organizing labels
    unique_set = ["PUL", "HEP", "PLE", "PER", "SKI", "OTH", "LYM", "BON",
                  "ADR", "MAR", "BRA"]
    for name in unique_set:
        y_df[name] = y_df["אבחנה-Location of distal metastases"].map(lambda x: 1 if name in x else 0)
    y_df.drop(columns=["אבחנה-Location of distal metastases"], inplace=True)
    X_y_df = pd.concat((X_df, y_df), axis=1)
    X_y_df.drop_duplicates(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'], inplace=True)
    y_df = X_y_df[unique_set]
    X_df = X_y_df.drop(columns=unique_set)
    X_df.drop(columns=['אבחנה-Surgery date3', 'אבחנה-Surgery name3', 'אבחנה-Tumor depth', "אבחנה-Diagnosis date",
                    'אבחנה-Tumor width', "אבחנה-Her2", ' Form Name', "User Name", "אבחנה-Surgery date1",
                    "אבחנה-Surgery date2", "אבחנה-Surgery date3", "surgery before or after-Activity date",
                    'id-hushed_internalpatientid', 'אבחנה-Surgery name2', 'אבחנה-Histological diagnosis',
                    'אבחנה-Ivi -Lymphovascular invasion', 'אבחנה-Lymphatic penetration',
                    'אבחנה-N -lymph nodes mark (TNM)', "אבחנה-Positive nodes", "אבחנה-Surgery sum"], inplace=True)
    X_df = pd.get_dummies(X_df, columns=["אבחנה-Basic stage", " Hospital", "אבחנה-Side", "אבחנה-Margin Type"])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2,
                                                        random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,
                                                      test_size=0.1,
                                                      random_state=21)
    X_train = preprocessing(X_train)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


if __name__ == '__main__':
    load_data("train.feats.csv", "train.labels.0.csv")
