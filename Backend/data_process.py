import numpy as np 
import pandas as pd 

raw_data = pd.read_csv('spy.csv')

date_processed_data = raw_data[raw_data['Year'] >= 2015]
date_processed_data = date_processed_data[date_processed_data['Day'] == 1]
date_processed_data = date_processed_data[['Date','Open','High','Low','Close','Volume']]

test_data = raw_data[raw_data['Year'] >= 2010]
test_data = raw_data[raw_data['Year'] < 2015]
test_data = test_data[test_data['Day'] == 1]
test_data = test_data[['Date','Open','High','Low','Close','Volume']]

def getProcessedTrainingData():
    return date_processed_data

def getProcessedTestingData():
    return test_data

