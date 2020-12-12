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
