import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv("diabetes.csv")
X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data['Outcome']

scalar = StandardScaler()
scalar.fit(X)
standard_data = scalar.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

x_train_pre = classifier.predict(x_train)
train_data_accuracy = accuracy_score(x_train_pre,y_train)
print("Accuracy score (train): ",train_data_accuracy)

x_test_pre = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_pre,y_test)
print("Accuracy score (test): ",test_data_accuracy)

input_data = (10,168,74,0,0,38,0.537,34)
# 10,168,74,0,0,38,0.537,34
# 4,110,92,0,0,37.6,0.191,30
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scalar.transform(input_data_reshape)
prediction = classifier.predict(std_data)
if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")