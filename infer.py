import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix ,accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

df = pd.read_csv('data/500_Person.csv')

df.dropna(inplace=True)

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

X = df.drop('Index',axis=1)
y = df['Index']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45)

model_tree = RandomForestClassifier()
model_tree.fit(X_train,y_train)
y_train_pred = model_tree.predict(X_train)
y_pred = model_tree.predict(X_test)
print(accuracy_score(y_train_pred,y_train))
print(accuracy_score(y_pred,y_test))


# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model_tree, initial_types=initial_type)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())