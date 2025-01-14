import os
import sys
import pandas as pd
import yaml


params=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['preprocess']

def DataPreprocessing(input_path,output_path):
    df=pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    df.to_csv(output_path,header=False,index=False)
    print(f"Processed file is saved in this Location->{output_path}")

if __name__=="__main__":
    DataPreprocessing(params['input'],params['output'])


