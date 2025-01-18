# Mlops-end-to-end

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


### Testing API using POSTMAN
![Testing_api](https://github.com/user-attachments/assets/bdbc0f0e-8b2d-4050-b7e7-9e6eef866ee9)


