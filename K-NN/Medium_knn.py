import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines 
import matplotlib.pyplot as plt 
import operator

def classifier(inx,dataset,labels,k):
    '''
    Parameters:
    inx: test dataset
    dataset: train dataset
    labels: classify labels 
    k: choose k points which has smartest distance
    Return:
    sorted_class_count[0][0]: classification results
    print for the checking result
    '''
    #dataset rowsize
    datasetsize = dataset.shape[0]
    print('datasetsize :',datasetsize)
    #creat matrix 
    diffmat = np.tile(inx,(datasetsize,1))-dataset 
    '''print('diffmat:',diffmat)'''
    #sum of square
    sq_diffmat = diffmat**2
    '''print('sq_diffmat',sq_diffmat)'''
    sqdistance = sq_diffmat.sum(axis=1)
    '''print('sqdostance',sqdistance)'''
    #calculate distance
    distances = sqdistance**0.5
    '''print('distances',distances)'''
    #order by distance
    sorted_distance_indices = distances.argsort()
    '''print('sorted_distance_indices :',sorted_distance_indices)'''
    #
    class_count={}
    for i in range(k):
        vote_label = labels[sorted_distance_indices[i]]
        class_count[vote_label] = class_count.get(vote_label,0)+1
        
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    '''print(class_count)'''
    '''print(sorted_class_count)'''
    return sorted_class_count[0][0]   

def file_matrix(filename):
    '''
    Parameters:
    filename: filename
    Returns:
    return_mat: feature matrix
    class_label_vector: label vactor
    '''
    fr = open(filename)
    array_lines = fr.readlines()
    # file rows
    number_lines = len(array_lines)
    # return empty matrix, columns = 3
    return_mat = np.zeros((number_lines,3))
    # label vactor
    class_label_vector=[]
    # rows indices
    index = 0

    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        # retrieve top three columns
        return_mat[index,:] = list_from_line[0:3]
        #
        if list_from_line[-1] == 'didntLike':
            class_label_vector.append(1)
        elif list_from_line[-1] == 'smallDoses':
            class_label_vector.append(2)
        elif list_from_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        index += 1
    return return_mat,class_label_vector

def data_visualization(dating_datamat,dating_labels):
    '''
    Return: Graphs
    '''
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex = False, sharey=False,figsize=(13,8))

    number_labels = len(dating_labels)
    labels_colors =[]
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        elif i == 2:
            labels_colors.append('orange')
        elif i == 3:
            labels_colors.append('red')
    # Scatter plot (First column: flight miles, Second column: playing game)
    axs[0][0].scatter(x=dating_datamat[:,0],y=dating_datamat[:,1], color=labels_colors,s=15,alpha=.5)
    # set title, x_label, y_label
    plt.setp(axs[0][0].set_title(u'The proportion of the flight miles and time of playing game'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[0][0].set_xlabel(u'flight miles each year'),size = 9, weight ='bold', color='black')
    plt.setp(axs[0][0].set_ylabel(u'time of playing game'),size = 9, weight ='bold', color='black')
    # Scatter plot (First column: flight miles, Second column: the volume of icecream consumption )
    axs[0][1].scatter(x=dating_datamat[:,0],y=dating_datamat[:,2], color=labels_colors,s=15,alpha=.5)
    plt.setp(axs[0][1].set_title(u'The proportion of the flight miles and consumption of icecream'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[0][1].set_xlabel(u'flight miles each year'),size = 9, weight ='bold', color='black')
    plt.setp(axs[0][1].set_ylabel(u'consumption of icecream'),size = 9, weight ='bold', color='black')
    #
    axs[1][0].scatter(x=dating_datamat[:,0],y=dating_datamat[:,2], color=labels_colors,s=15,alpha=.5)
    plt.setp(axs[1][0].set_title(u'The proportion of the time of playing game and consumption of icecream'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[1][0].set_xlabel(u'time of playing game'),size = 9, weight ='bold', color='black')
    plt.setp(axs[1][0].set_ylabel(u'consumption of icecream'),size = 9, weight ='bold', color='black')
    # setting legends
    didntLike = mlines.Line2D([],[],color='black',marker='.', markersize = 6, label = 'didntLike')
    smallDoses = mlines.Line2D([],[],color ='orange',marker='.',markersize=6,label = 'smallDoses')
    largeDoses = mlines.Line2D([],[],color ='red',marker='.',markersize=6,label = 'largeDoses')
    # placing legends
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()
    
    '''Scaling data'''
    # newvalue = (oldvalue - min)/(max - min)
def data_norm(dataset):
    '''
    Parameter:
    dataset: data set
    Returns:
    norm_dataset: normalized dataset
    ranges: data range
    minvals: the minimum value
    '''
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    # range
    ranges = maxvals - minvals
    # the empty matrix
    norm_dataset = np.zeros(np.shape(dataset))
    # the rows of dataset
    rows = dataset.shape[0]
    # calculating
    norm_dataset = (dataset - np.tile(minvals,(rows,1))) / np.tile(ranges,(rows,1))
    return norm_dataset,ranges,minvals


def dating_classtest():
    '''
    test the model
    '''
    filename = 'datingTestSet.txt'
    dating_datamat,dating_labels = file_matrix(filename)
    # retrieve 10% data for test
    ho_ratio = 0.10
    norm_dataset,ranges,minvals = data_norm(dating_datamat)
    # rows
    rows = norm_dataset.shape[0]
    # 10% for test
    num_test_vectors = int(rows*ho_ratio)
    error_count = 0.0

    for i in range(num_test_vectors):
        # num_test_vectors as the test dataset, rows-num_test_vectors as the train dataset
        classifier_result = classifier(norm_dataset[i,:],norm_dataset[num_test_vectors:rows,:],
        dating_labels[num_test_vectors:rows],4)
        print('classifier results:%d\ttrue results:%d'%(classifier_result,dating_labels[i]))
        if classifier_result != dating_labels[i]:

            error_count+=1.0
    print('error rate: %f%%'%(error_count/float(num_test_vectors)*100))






if __name__ == '__main__':
    dating_classtest()
    '''
    filename = 'datingTestSet.txt'
    dating_datamat, dating_labels = file_matrix(filename)
    print(dating_datamat)
    print(dating_labels)
    data_visualization(dating_datamat,dating_labels)
    norm_dataset,ranges,minvals = data_norm(dating_datamat)
    print(norm_dataset)
    print(ranges)
    print(minvals)
    '''

