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
def sigmoid(inx):
    '''
    '''
    return 1.0/(1+np.exp(-inx))

def grad_ascent(data_matrixin,class_label):
    '''
    Parameter:
        data_matrix: dataset
        class_label: data label
    Return:
        weight.getA(): the optimal weights dataset

    '''
    data_matrix = np.mat(data_matrixin)
    label_matrix = np.mat(class_label).transpose()
    m,n = np.shape(data_matrix) # m represent rows 
    aplha = 0.001
    max_cylcles = 500 
    weights = np.ones((n,1))
    for k in range(max_cylcles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + aplha * data_matrix.transpose() * error
    return weights.getA()




def plot_dataset():
    '''
    '''
    data_matrix, label_matrix =load_dataset()
    data_array = np.array(data_matrix)
    n = np.shape(data_matrix)[0] # the number of data
    xcord1 =[]; ycord1 = []
    xcord2 =[]; ycord2 =[]
    for i in range(n):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i,1]);ycord1.append(data_array[i,2])
        else:
            xcord2.append(data_array[i,1]);ycord2.append(data_array[i,2])
    fig =plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s= 20,c='red',marker= 's', alpha=.5)
    ax.scatter(xcord2,ycord2, s=20, c='green',alpha=.5)
    plt.title('Dataset')
    plt.xlabel('x') ; plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    data_matrix, label_matrix =load_dataset()
    print(grad_ascent(data_matrix,label_matrix))
 