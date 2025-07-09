import mlflow.sklearn
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
import pandas as pd 
import mlflow 


df = pd.read_csv("https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Diabetes.csv") 

X = df.drop("diabetes", axis=1) 
y = df["diabetes"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2) 

rf = RandomForestClassifier(random_state=42) 

param_grid = {
    "n_estimators" : [10,50,100], 
    "max_depth" : [None, 10, 20, 30] 
}

grid_search = GridSearchCV(estimator = rf, param_grid=param_grid, cv = 5,  n_jobs=-1, verbose = 2) 

mlflow.set_experiment("Hyperparameter experiment") 

with mlflow.start_run():

    grid_search.fit(X_train, y_train) 

    best_params = grid_search.best_params_ 
    best_score = grid_search.best_score_ 

    # params 
    mlflow.log_params(best_params)

    # metrics 
    mlflow.log_metric("accuracy", best_score)

    # data 
    train_df = X_train.copy()
    train_df["diabetes"] = y_train
    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, context="training")


    test_df = X_test.copy()
    test_df["diabetes"] = y_test  
    test_df = mlflow.data.from_pandas(test_df) 
    mlflow.log_input(test_df, "testing_data")

    # source code 
    mlflow.log_artifact(__file__) 

    # model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random forest") 

    # tags 
    mlflow.set_tag("author", "dhruv")

    print(best_params) 
    print(best_score) 

