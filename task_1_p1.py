import pandas as pd
import plotly.express as px
X_df = pd.read_csv("train.feats.csv")
y_df = pd.read_csv("train.labels.0.csv")
df = pd.concat((X_df, y_df), axis=1)
# x = df.loc[df['אבחנה-Location of distal metastases'] != '[]']
df.drop(columns=['אבחנה-Surgery date3', 'אבחנה-Surgery name3', 'אבחנה-Tumor depth', 'אבחנה-Tumor width'], inplace=True)
col = df["אבחנה-KI67 protein"]
print(col.unique())
# fig = px.histogram(df,
#                  x="אבחנה-Age",
#                  color="אבחנה-Location of distal metastases",
#                  hover_data=df.columns,
#                  title="Distribution of  Diseases",
#                  )
# fig.show()

