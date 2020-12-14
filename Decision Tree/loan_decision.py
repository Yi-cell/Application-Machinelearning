from math import log
import matplotlib.pyplot as plt 
import operator 

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
        if current_label not in label_counts.keys(): # count the labels 
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    Ent = 0.0 #Empirical entropy
    for key in label_counts:
        prob = float(label_counts[key]) / rows # the probability of labels
        Ent -= prob * log(prob,2) # calculate the the entropy 
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
    Return:
        return a dataset
    '''
    new_dataset = []
    for features in dataset:
        if features[axis] == value:
            remove_axis_vector = features[:axis] # remove the axis feature
            remove_axis_vector.extend(features[axis+1:])
            new_dataset.append(remove_axis_vector)
    return new_dataset

def optimal_feature(dataset):
    '''
    Parameter:
        dataset: dataset
    Returns: 
        opt_feature: the optimal feature index
    '''
    num_features = len(dataset[0])-1 # number of features
    entropy = empirical_entropy(dataset) # get the figure of entropy
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

def majority_count(classlist):
    '''
    Parameter:
        classlist: class lsit
    Return:
        sorted_class_count[0][0]: the most count of class
    '''
    class_count={}
    for vote in classlist:
        if vote not in class_count.keys():class_count[vote]=0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset,labels,feature_labels):
    '''
    Parameter:
        dataset: train dataset
        labels: class labels
        feature_labels: save the optimal class labels
    '''
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):# if the class completely same then stop split
         return classlist[0]
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majority_count(classlist)
    best_features = optimal_feature(dataset)
    best_features_labels = labels[best_features]
    feature_labels.append(best_features_labels)
    my_tree = {best_features_labels:{}}
    del(labels[best_features])
    feature_values = [example[best_features] for example in dataset]
    unique_vals = set(feature_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_features_labels][value] = create_tree(split_dataset(dataset,best_features,value),sub_labels,feature_labels)
    return my_tree




''' Decision tree visualization'''
def get_leafs(my_tree):
    '''
    Parameter:
        my_tree: decision tree
    Return:
        num_leafs: number of leaf nodes
    '''
    num_leafs = 0
    first_string = next(iter(my_tree))
    second_dict = my_tree[first_string]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    '''
    Parameter:
        my_tree: decision tree
    Return:
        max_depth: the depth of decision tree
    '''
    max_depth = 0
    first_string = next(iter(my_tree))
    second_dict = my_tree[first_string]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ =='dict':
            depth = 1 + get_tree_depth(second_dict[key])
        else:
            depth = 1
        if depth > max_depth: 
            max_depth = depth
    return max_depth

def plot_node(node_name,position,arrow,node_type):
    '''
    Parameter:
        node_name: node name 
        position: the text position 
        arrow: the arrow position
        node_type: type of node
    Return:
        None
    '''
    arrow_args = dict(arrowstyle = '<-')
    create_plot.ax1.annotate(node_name,xy=arrow,xycoords='axes fraction',xytext = position,
    textcoords='axes fraction',va='center',ha='center',bbox = node_type,arrowprops=arrow_args)

def plot_text(center,parent,text):
    '''
    Parameter:
        center,parent: locating the position
        text: the content
    Return:
        None
    '''
    x = (parent[0]-center[0])/2.0 + center[0]
    y = (parent[1]-center[1])/2.0 + center[1]
    create_plot.ax1.text(x,y,text,va ='center',ha='center',rotation =30)

def plot_tree(my_tree,parent,text):
    decision_node = dict(boxstyle = 'sawtooth',fc = '0.8')
    leaf_node = dict(boxstyle='round4',fc='0.8')
    num_leafs =get_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_string = next(iter(my_tree))
    center = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW,plot_tree.yOff)
    plot_text(center,parent,text)
    plot_node(first_string,center,parent,decision_node)

    second_dict = my_tree[first_string]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD 
   
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ =='dict':
            plot_tree(second_dict[key],center,str(key))

        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(second_dict[key],(plot_tree.xOff,plot_tree.yOff),center,leaf_node)
            plot_text((plot_tree.xOff,plot_tree.yOff), center,str(key))

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD

def create_plot(intree):
    '''
    Parameter:
        intree: my tree dictionary
    Return:
        None
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    create_plot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plot_tree.totalW = float(get_leafs(intree))
    plot_tree.totalD =  float(get_tree_depth(intree))
    plot_tree.xOff = -0.5 / plot_tree.totalW; plot_tree.yOff = 1.0
    plot_tree(intree,(0.5,1.0),'')
    plt.show()




if __name__ == '__main__':
    dataset, labels = create_dataset()
    feature_labels=[]
    my_tree = create_tree(dataset,labels,feature_labels)
    print(my_tree)
    create_plot(my_tree)






