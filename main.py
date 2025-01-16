from src.DataPreprocessing import DataPreprocessing
from src.ModelTrainer import ModelTrainer,DataSplit
from src.ModelEvalution import ModelEvalution
from src.ModelPedict import ModelPredict
                                   
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import yaml



params=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['preprocess']
params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']
params_evalution=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["evalution"]
params_track=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["mlflow"]


if __name__=="__main__":
    
    DataPreprocessing(params['input'],params['output'])

    DataSplit(params_trainer,params_evalution,params['output'],
            params_trainer['X_train_reg'],params_trainer['y_train_reg'],params_trainer['X_train_clf'],params_trainer['y_train_clf'],
            params_evalution['y_test_reg'],params_evalution['y_test_clf'],params_evalution['X_test_reg'],params_evalution['X_test_clf'])
    
    reg=RandomForestRegressor()
    clf=RandomForestClassifier()

    ModelTrainer(reg,clf,params_trainer['model_reg'],params_trainer['model_clf'],
                params_trainer['X_train_reg'],params_trainer['y_train_reg'],params_trainer['X_train_clf'],params_trainer['y_train_clf'])
    ModelEvalution(params_track["uri"],params_trainer['model_reg'],params_trainer['model_clf'],
                params_evalution['X_test_reg'],params_evalution['y_test_reg'],params_evalution['X_test_clf'],params_evalution['y_test_clf'])
    
    X_test_reg=pd.read_csv(params_evalution['X_test_reg'])
    X_values=np.array(X_test_reg.iloc[1:2,:])
    print("X_Value shapeeeee",X_values.shape)
    Predictions=ModelPredict(params_trainer['model_reg'],params_trainer['model_clf'],X_values)
    print(Predictions)