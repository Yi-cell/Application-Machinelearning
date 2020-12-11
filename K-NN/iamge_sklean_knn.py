import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img_to_vector(filename):
    # create 1x1024 vector
    return_vector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_string = fr.readline()
        # for each 32 strings in each row, adding it to the return vector
        for j in range(32):
            return_vector[0,32*i+j] = int(line_string[j])
    return return_vector

def classifier():
    '''
    '''
    test_labels =[]
    # return train file list 
    training_file_list = listdir('trainingDigits')
    # number of files
    file_number = len(training_file_list)
    # initialize the train matrix
    training_mat = np.zeros((file_number,1024))
    # train
    for i in range(file_number):
        #get the name
        file_name = training_file_list[i]
        class_number = int(file_name.split('_')[0])
        # adding the classnumber to the test labels
        test_labels.append(class_number)
        # save each file into the matrix
        training_mat[i,:] = img_to_vector('trainingDigits/%s' % (file_name))
    # build the classifier
    k_neigh = kNN(n_neighbors=3,algorithm='auto')
    # fit the model, training_mat as the training matrix, test_labels as the labels
    k_neigh.fit(training_mat,test_labels)
    #return the test list
    test_file_list =listdir('testDigits')
    #error 
    error_count = 0.0
    # number of test file
    test_file_number = len(test_file_list)
    # test
    for i in range(test_file_number):
        # get the name
        file_name = test_file_list[i]
        class_number = int(file_name.split('_')[0])
        vector_test = img_to_vector('testDigits/%s' % (file_name))
        # get the results
        classifier_results = k_neigh.predict(vector_test)
        print('classifier results: %d\t real results:%d' % (classifier_results,class_number))
        if (classifier_results != class_number):
            error_count += 1.0
    print("the number of error:%d/n error rate:%f%%" %(error_count,error_count/test_file_number*100))



if __name__ =='__main__':
    classifier()