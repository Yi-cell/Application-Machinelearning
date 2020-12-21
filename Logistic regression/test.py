import matplotlib.pyplot as plt 
import numpy as np

def load_dataset():
    '''
    '''
    data_matrix =[]
    label_matrix = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0,float(line_array[0]),float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    fr.close()
    return data_matrix, label_matrix
