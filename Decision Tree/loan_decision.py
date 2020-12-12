from math import log
import operator 
'''
'''
def empirical_entropy(dataset):
    '''
    Parameter:
    dataset: dataset
    Retrun:
    Ent: empirical entropy
    '''
    rows = len(dataset) #return the rows of dataset
    label_counts = {} #save the counts of labels
    for features in dataset:
        current_label = features[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    Ent = 0.0 #Empirical entropy
    for key in label_counts:
        prob = float(label_counts[key]) / rows
        Ent -= prob * log(prob,2)
    return Ent

def create_dataset():
    '''
    creat a dataset
    '''
    dataset = [ [0, 0, 0, 0, 'no'],                        
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['Age','Job','Property','Credit']
    return dataset, labels

def split_dataset(dataset,axis,value):
    '''
    Parameters:
    dataset: dataset
    axis: features
    value: the value of features
    '''
    new_dataset = []
    for features in dataset:
        if features[axis] == value:
            remove_feature_vector = features[:axis]
            remove_feature_vector.extend(features[axis+1:])
            new_dataset.append(remove_feature_vector)
    return new_dataset

def optimal_feature(dataset):
    '''
    Parameter:
    dataset: dataset
    Returns: 
    opt_feature: the optimal feature index
    '''
    num_features = len(dataset[0])-1 # number of features
    entropy = empirical_entropy(dataset) # get the entropy
    optimal_information_gain = 0.0
    opt_feature = -1
    for i in range(num_features):
        #get all the features in i
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0

        for value in unique_vals:
            #calculate the empirical entropy
            new_dataset = split_dataset(dataset,i,value)
            prob = len(new_dataset) / float(len(dataset))
            new_entropy += prob * empirical_entropy(new_dataset) 
        information_gain = entropy - new_entropy

        if (information_gain > optimal_information_gain):
            optimal_information_gain = information_gain
            opt_feature = i
        return opt_feature