import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df["Churn"].value_counts())
print(df.isnull().sum())
print(df.dtypes)
df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df["Churn"].value_counts())
print(df.dtypes)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

text_columns = df.select_dtypes(include=["object"]).columns

for col in text_columns:
    df[col] = le.fit_transform(df[col])


print(df.dtypes)
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop(columns=["Churn"])
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier

model = XGBClassifier(scale_pos_weight=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
import pandas as pd

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance.head(10))
print(classification_report(y_test, predictions))
import matplotlib.pyplot as plt

# Plot feature importance
feature_importance.head(10).plot(kind="barh", figsize=(10, 6))
plt.title("Top 10 Features Predicting Customer Churn")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Chart saved!")