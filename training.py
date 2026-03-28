import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("vision_log.csv")

print("Data loaded:")
print(df.head())

X = df[["total_objects", "person_count", "unique_objects"]]

df["crowded"] = df["person_count"].apply(lambda x: 1 if x >= 3 else 0)
y = df["crowded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
# Use .values to avoid feature name warnings during prediction
model.fit(X_train.values, y_train)

print("Model trained successfully!")
joblib.dump(model, "crowd_model.pkl")
print("Model saved as crowd_model.pkl")