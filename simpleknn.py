import numpy as np
import operator
def create_dataset():
    # four group features
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    # four group labels
    labels = ['Romantic','Romantic','Action','Action']
    return group,labels

def classifier(inx,dataset,labels,k):
    '''
    Parameters:
    inx: test dataset
    dataset: train dataset
    labels: classify labels 
    k: choose k points which has smartest distance
    Return:
    sorted_class_count[0][0]: classification results
    '''
    #dataset rowsize
    datasetsize = dataset.shape[0]
    print('datasetsize :',datasetsize)
    #creat matrix 
    diffmat = np.tile(inx,(datasetsize,1))-dataset 
    print('diffmat:',diffmat)
    #sum of square
    sq_diffmat = diffmat**2
    print('sq_diffmat',sq_diffmat)
    sqdistance = sq_diffmat.sum(axis=1)
    print('sqdostance',sqdistance)
    #calculate distance
    distances = sqdistance**0.5
    print('distances',distances)
    #order by distance
    sorted_distance_indices = distances.argsort()
    print('sorted_distance_indices :',sorted_distance_indices)
    #
    class_count={}
    for i in range(k):
        vote_label = labels[sorted_distance_indices[i]]
        class_count[vote_label] = class_count.get(vote_label,0)+1
        
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]
    print('class_count:',class_count)
    
if __name__ == '__main__':
    group,labels = create_dataset()

    test = [101,20]
    test_class = classifier(test,group,labels,3)
    print(test_class)





