import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,precision_score,recall_score,f1_score
import pickle
import yaml

params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']

params_evalution=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["evalution"]

params_track=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["mlflow"]


def ModelEvalution(uri,model_reg,model_clf,
                   X_test_reg,y_test_reg,X_test_clf,y_test_clf):
    

    X_test_reg=pd.read_csv(X_test_reg)
    y_test_reg=pd.read_csv(y_test_reg)
    X_test_clf=pd.read_csv(X_test_clf)
    y_test_clf=pd.read_csv(y_test_clf)

    model_reg=pickle.load(open(model_reg,'rb'))
    model_clf=pickle.load(open(model_clf,'rb'))

    # Parameters to be track
    reg_params=model_reg.get_params()
    clf_params=model_clf.get_params()

    #Metrics to be track
    y_pred_reg=model_reg.predict(X_test_reg)
    y_pred_clf=model_clf.predict(X_test_clf)

    #for regression
    mse=mean_squared_error(y_pred_reg,y_test_reg)
    mae=mean_absolute_error(y_pred_reg,y_test_reg)
    r2_scr=r2_score(y_pred_reg,y_test_reg)

    #for classification
    acc=accuracy_score(y_pred_clf,y_test_clf)
    precision_scr=precision_score(y_pred_clf,y_test_clf)
    recall_scr=recall_score(y_pred_clf,y_test_clf)
    f1_scr=f1_score(y_pred_clf,y_test_clf)

    reg_metrics={
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score':r2_scr
    }

    clf_metrics={
        'accuracy_score': acc,
        'precision_score': precision_scr,
        'recall_score':recall_scr,
        'f1_score':f1_scr
    }



    # Mlflow Tracking
    mlflow.set_registry_uri(uri=uri)
    mlflow.set_experiment(experiment_name="Multioutput_Model")

    with mlflow.start_run():

        mlflow.log_params(reg_params)
        mlflow.log_metrics(reg_metrics)
        
        signature_reg=infer_signature(X_test_reg,model_reg.predict(X_test_reg))
        
        model_reg_info=mlflow.sklearn.log_model(
            sk_model=model_reg,
            artifact_path="artifacts/model_reg",
            signature=signature_reg,
            input_example=X_test_reg,
            registered_model_name="tracking-regressor"
        )

      
    with mlflow.start_run():

            mlflow.log_params(clf_params)

            mlflow.log_metrics(clf_metrics)

            signature_clf=infer_signature(X_test_clf,model_clf.predict(X_test_clf))

            model_clf_info=mlflow.sklearn.log_model(
                sk_model=model_clf,
                artifact_path="artifacts/model_clf",
                signature=signature_clf,
                input_example=X_test_reg,
                registered_model_name="tracking-classifier"
            )
    
    
    print("artifacts are save in-> artifacts/")


