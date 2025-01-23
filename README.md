# Mlops-end-to-end (Wine Quality Multioutput Model)
### Project Summary
This project demonstrates an end-to-end MLOps pipeline for a wine quality prediction system using regression and classification models. The process includes data preprocessing, model training, evaluation, prediction, and API deployment.

1)Data Preprocessing: Raw data from data/raw was cleaned by removing unnecessary columns like "Unnamed : 0" and stored in data/processed.

2)Model Training: The data was split into train and test sets for both regression and classification tasks. Models were trained using two pipelines:

Pipeline 1: RandomForest Regressor and Classifier.
Pipeline 2: ElasticNet Regressor and DecisionTree Classifier.
Trained models were saved in the models directory.

3)Model Evaluation: Metrics such as MSE, MAE, R² (regression) and precision, recall, accuracy, F1-score (classification) were computed and logged to MLflow.

4)Model Prediction: Predictions were made using input features provided as a NumPy array. Regression predictions were used as input to the classification model for quality classification.

Pipeline Execution: The pipeline was orchestrated in main.py, enabling seamless execution of all stages.

5)API Deployment: A FastAPI application was created for serving predictions and tested using Postman.

This modular structure and detailed workflow provide a comprehensive demonstration of MLOps practices.
### Project Structure
```bash
.
├── Data
│   ├── raw                   # Raw data
│   ├── processed             # Optimized data
│   ├── for_training          # X_reg_train,y_reg_train,X_clf_train,y_clf_train
│   ├── for_evalution         # X_reg_test,y_reg_test,X_clf_test,y_clf_test
├── experiments
│   ├── experiment_1          # for unstructured experiment 
│   ├── experiment_main       # modular experiment
├── src
│   ├── DataPreprocessing.py  # Module for preprocessing the data
│   ├── ModelTrainer.py       # Module for training models and splitting data
│   ├── ModelEvalution.py     # Module for evaluating models and logging with MLflow
│   ├── ModelPredict.py       # Module for generating predictions
├── params.yaml               # Configuration file for pipeline parameters
├── main.py                   # Entry point for the pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── models                    # All models
    ├── model_1_reg.pkl       #Randomforest Regressor
    ├── model_1_clf.pkl       #Randomforest Classifier
    ├── model_2_reg.pkk       #ElasticNet Regressor
    ├── model_2_clf.pkl       #DecisionTree Classifier
```

# How to run?
### STEPS:


Clone the repository

```bash
https://github.com/PratikKujur/Mlops-end-to-end.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.10 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
uvicorn app:app --reload

```

Now,
```bash
open up you local host and port
```
### Representation of Model Prediction using Swagger ui 
![swagger_ui](https://github.com/user-attachments/assets/6ac236a6-4155-46a4-893f-b595a708e89a)

To Open Mlflow ui
```bash

# Run the following command
mlflow ui 
```
### STEP 03- Test API Endpoints with Postman
1)Open Postman and create a new request.

2)Set the request method to POST.

3)Use the endpoint for prediction (e.g., /predict).

4)Pass the input feature dictionary in JSON format in the request body(raw)
```bash

{
  "volatile_acidity": 0.32,
  "citric_acid": 0.12,
  "residual_sugar": 6.6,
  "chlorides": 0.423,
  "free_sulfur_dioxide": 22.0,
  "total_sulfur_dioxide": 141.0,
  "density": 0.9937,
  "pH": 3.36,
  "sulphates": 0.6,
  "alcohol": 10.4
}
```


### Testing API Endpoints using POSTMAN
![Testing_api](https://github.com/user-attachments/assets/bdbc0f0e-8b2d-4050-b7e7-9e6eef866ee9)


