import numpy as np 
import pandas as pd 

raw_data = pd.read_csv('./Backend/spy.csv')


date_processed_data = raw_data[raw_data['Year'] >= 2015]
date_processed_data = date_processed_data[date_processed_data['Day'] == 1]

def getProcessedData():
    return date_processed_data