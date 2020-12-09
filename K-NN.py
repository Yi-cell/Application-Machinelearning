'''k-nearest neighbor'''
import pandas as pd 
import numpy as np 

test = pd.read_csv('Predict Health Insurance/test.csv')
train = pd.read_csv('Predict Health Insurance/train.csv')
print(test.head())
print(test.shape)
print(train.head())
print(train.shape)