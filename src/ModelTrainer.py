import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

params=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['preprocess']

params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']

params_evalution=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["evalution"]


def DataSplit(params_trainer,params_evalution,processed_path,
              X_train_reg_path,y_train_reg_path,X_train_clf_path,y_train_clf_path,
              y_test_reg_path,y_test_clf_path,X_test_reg_path,X_test_clf_path):

    df=pd.read_csv(processed_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    y_reg=df['quality']
    y_clf=df['is_red']

    X_clf=df.iloc[:,1:12]
    X_reg=df.iloc[:,1:11]

    X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
    X_train_clf,X_test_clf,y_train_clf,y_test_clf=train_test_split(X_clf,y_clf,test_size=0.2,random_state=42)

    os.makedirs(os.path.dirname(X_train_reg_path),exist_ok=True)
    os.makedirs(os.path.dirname(X_test_reg_path),exist_ok=True)
    os.makedirs(os.path.dirname(y_train_reg_path),exist_ok=True)
    os.makedirs(os.path.dirname(y_test_reg_path),exist_ok=True)
    os.makedirs(os.path.dirname(X_train_clf_path),exist_ok=True)
    os.makedirs(os.path.dirname(X_test_clf_path),exist_ok=True)
    os.makedirs(os.path.dirname(y_train_clf_path),exist_ok=True)
    os.makedirs(os.path.dirname(y_test_clf_path),exist_ok=True)
    
    X_train_reg.to_csv(X_train_reg_path,index=False,header=True)
    X_test_reg.to_csv(X_test_reg_path,index=False,header=True)
    y_train_reg.to_csv(y_train_reg_path,index=False,header=True)
    y_test_reg.to_csv(y_test_reg_path,index=False,header=True)
    X_train_clf.to_csv(X_train_clf_path,index=False,header=True)
    X_test_clf.to_csv(X_test_clf_path,index=False,header=True)
    y_train_clf.to_csv(y_train_clf_path,index=False,header=True)
    y_test_clf.to_csv(y_test_clf_path,index=False,header=True)
    
    print("Training Data Shape:", X_train_clf.shape, "Columns:", X_train_clf.columns)
    print("Testing Data Shape:", X_test_clf.shape, "Columns:", X_test_clf.columns)

    print(f"Data for Taining are saved at ->{params_trainer}")
    print(f"Data for evalution are saved at ->{params_evalution}")

def ModelTrainer(reg,clf,model_reg_path,model_clf_path,
                 X_train_reg,y_train_reg,X_train_clf,y_train_clf):
    X_train_clf=pd.read_csv(X_train_clf)
    y_train_clf=pd.read_csv(y_train_clf)
    X_train_reg=pd.read_csv(X_train_reg)
    y_train_reg=pd.read_csv(y_train_reg)
    
    reg.fit(X_train_reg,y_train_reg)
    os.makedirs(os.path.dirname(model_reg_path),exist_ok=True)
    pickle.dump(reg,open(model_reg_path,'wb'))

    clf.fit(X_train_clf,y_train_clf)
    os.makedirs(os.path.dirname(model_clf_path),exist_ok=True)
    pickle.dump(clf,open(model_clf_path,'wb'))

    print(f"Regression model is saved at -> {model_reg_path}",f"Classification model is saved at -> {model_clf_path}")



if __name__=="__main__":
    
    DataSplit(params_trainer,params_evalution,params['output'],
              params_trainer['X_train_reg'],params_trainer['y_train_reg'],params_trainer['X_train_clf'],params_trainer['y_train_clf'],
              params_evalution['y_test_reg'],params_evalution['y_test_clf'],params_evalution['X_test_reg'],params_evalution['X_test_clf'])
    reg=RandomForestRegressor()
    clf=RandomForestClassifier()
    ModelTrainer(reg,clf,params_trainer['model_reg'],params_trainer['model_clf'],
                params_trainer['X_train_reg'],params_trainer['y_train_reg'],params_trainer['X_train_clf'],params_trainer['y_train_clf'])