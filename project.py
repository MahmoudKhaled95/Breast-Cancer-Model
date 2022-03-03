import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data:
df = pd.read_csv("breast_cancer.csv")

#veiw sample from the data:
print(df.head())
print(len(df))

#handle missing data methode (1):
print(df.info())
print(df.isna().sum())
##not found missing data##
            
### remove un-needed raws:
df.drop("Unnamed: 32",axis= 1,inplace= True)
df.drop("id" , axis = 1 ,inplace = True)
print(df.head())

#map labled data
df["diagnosis"] = df["diagnosis"].map({"M":1 , "B":0})
print(df["diagnosis"])

#seprate input and output from the df
x = df.drop("diagnosis" , axis = 1 , inplace = False)
y = df["diagnosis"]
print(x)

#split data to train and test data
from sklearn.model_selection import train_test_split
np.random.seed(42)
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2, shuffle = True)

#choose model/estimator based on the problem
from sklearn import svm
model = svm.SVC()

#train model
model.fit(x_train , y_train)

#model evaluation
print(model.score(x_train , y_train))
print(model.score(x_test , y_test))
y_predicted = model.predict(x_test)
print(y_predicted)
accuracy = np.mean(y_predicted == y_test)

#save best model
import pickle
pickle.dump(model, open("cancer_classifier_model.pkl" , "wb"))
