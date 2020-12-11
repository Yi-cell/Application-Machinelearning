'''k-nearest neighbor'''
import pandas as pd 
import numpy as np 

test = pd.read_csv('Predict Health Insurance/test.csv')
train = pd.read_csv('Predict Health Insurance/train.csv')
'''
print(test.head())
print(test.shape)
print(train.head())
print('-------')
print(train.shape)
print(test.isnull().sum())
# try two variables (Annual_premium & Region_Code & Vintage)
numerical_columns = ['Annual_Premium','Region_Code','Vintage']
print(train[numerical_columns].describe())
print(len(train[numerical_columns]))
returnMat = np.zeros((len(train[numerical_columns]),3))
print(returnMat)
classLabelVactor = []
index = 0
 
returnMat[index,:] = train[numerical_columns].iloc[:,0:3]

print(train[numerical_columns].iloc[:,0:3]) 
print(returnMat)
'''
'''
def knn(train,test,labels,k):
    # number of rows
    row_size = train.shape[0]
    # calculate the difference 
    diff = np.tile(test,(row_size,1)) - train
    # sum of square 
    sqrdiff = diff ** 2
    sqrdiffsum = sqrdiff.sum(axis = 1)
    # calculate the distance
    distances = sqrdiffsum ** 0.5
    # order by distance
    sortdistances = distances.argsort()

    count = {}
    for i in range(k):
'''
def file_matrix(filename):
    rows_size = len(train)
    return_mat = np.zeros((rows_size,3))
    label_vector = []
    index = 0
    
    for line in train:
        print (line)

 from sklearn.preprocessing import StandardScaler