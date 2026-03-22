import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# load dataset
data = pd.read_csv("dataset/pressure_data.csv")

# encode tissue type
le_tissue = LabelEncoder()
data["tissue_type"] = le_tissue.fit_transform(data["tissue_type"])

# encode risk
le_risk = LabelEncoder()
data["risk"] = le_risk.fit_transform(data["risk"])

X = data[["pressure","tissue_type"]]
y = data["risk"]

# train model
model = DecisionTreeClassifier()
model.fit(X,y)

# save model
pickle.dump(model,open("pressure_model.pkl","wb"))
pickle.dump(le_tissue,open("tissue_encoder.pkl","wb"))
pickle.dump(le_risk,open("risk_encoder.pkl","wb"))

print("Model trained and saved")