import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 15

 

# apply mlflow 

# for creating new experiments:
mlflow.set_experiment("iris-dt")


with mlflow.start_run(run_name="dhruvparikh"): 
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train, y_train)    

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) 

    mlflow.log_metric("accuracy", accuracy) 
    mlflow.log_param("max_depth", max_depth)

    print("accuracy", accuracy) 





