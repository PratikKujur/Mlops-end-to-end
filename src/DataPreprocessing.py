import os
import sys
import pandas as pd
import yaml


params=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['preprocess']

def DataPreprocessing(input_path,output_path):
    df=pd.read_csv(input_path)
    print("Raw Data \n",df.head())
    # add preprocessing step here

    unecessary_col=['Unnamed: 0']
    df.drop(labels=unecessary_col,axis='columns',inplace=True)
    
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False,header=True)
    print("Preprocessed Data \n",df.head())
    print(f"Preprocessed data is saved in this location -> {output_path}")

if __name__=="__main__":
    DataPreprocessing(params['input'],params['output'])


