import pickle
import numpy as np
import yaml
import pandas as pd

params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']

params_evalution=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))["evalution"]


def ModelPredict(model_reg,model_clf,X_values):

    #load models
    model_reg=pickle.load(open(model_reg,'rb'))
    model_clf=pickle.load(open(model_clf,'rb'))

    model_reg_output=model_reg.predict(X_values)

    model_clf_input=np.append(X_values,model_reg_output)

    model_clf_input=model_clf_input.reshape(1,-1)

    model_clf_output=model_clf.predict(model_clf_input)

    print(f"Wine Quality is -> {model_reg_output}",f"Wine is red if 1 else white if 0 {model_clf_output}")


