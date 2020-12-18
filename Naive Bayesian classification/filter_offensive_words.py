import numpy as np
def load_dataset():
    '''
    Parameter:
        None
    Return:
        posting_list: sample words
        class_vector: class label vector
    '''
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],               
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    class_vector =[0,1,0,1,0,1] # '1' represent offensive, '0' represent non-offensive

    return posting_list, class_vector

def set_words_vec(vocabulary_list,input_set):
    '''
    Parameter:
        vocabulary_list: create vocabulary list 
        input_set: slice the words list
    Return:
        return_vec: word vector
    '''
    return_vec =[0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vec[vocabulary_list.index(word)] = 1
        else:
            print('The word: %s is not in my vocabulary list!' % word)
    return return_vec
def create_vocabulary_list(dataset):
    '''
    Parameter:
        dataset: dataset
    Returns:
        vocabulary_set: vocabulary dataset list
    '''
    vocabulary_set =set([]) # create an empty and non-repeat list
    for document in dataset:
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)
def classifier(word_vector, p0vec,p1vec,pclass):
    '''
    Parameter:
        word_vector: vector of words
        p0vec: vector of probability of offensive words
        p1vec: vector of probablity of non-offensive words
        pclass: probablity of the file belongs to offensive
    Return:
        0: non-offensive
        1: offensive
    '''
    p1 = reduce(lambda x,y: x*y, word_vector*p1vec) * pclass
    p0 = reduce(lambda x,y: x*y, word_vector*p0vec) * (1.0-pclass)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0
        
def train_NB(train_matrix,train_category):
    '''
    Parameter:
        train_matrix: return_vector
        train_category: class_vector
    Return:
        p0_vector: the vector of probability of non-offensive words
        p1_vector: the vector of probability of offensive words
        p_abusive: the probability of the abusive words
    '''
    num_train_docs =len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.zeros(num_words); p1_num = np.zeros(num_words)
    p0_denom = 0.0; p1_Denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_Denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vector = p1_num / p1_Denom
    p0_vector = p0_num / p0_denom

    return p0_vector, p1_vector, p_abusive

if __name__ == '__main__':
    posting_list, class_vector = load_dataset()
    ''' for each in posting_list:
        print(each)
        print(class_vector)
    
    print('postinglist:\n', posting_list)
    my_vocab_list = create_vocabulary_list(posting_list)
    print('my_vocab_list: \n', my_vocab_list)
    train_matrix = []
    for posting_doc in posting_list:
        train_matrix.append(set_words_vec(my_vocab_list,posting_doc))
    print('train matrix:\n',train_matrix)
    '''
    my_vocab_list = create_vocabulary_list(posting_list)
    print('my_vocab_list: \n', my_vocab_list)
    train_matrix = []
    for postingdoc in posting_list:
        train_matrix.append(set_words_vec(my_vocab_list,postingdoc))
    p0v, p1v, pab = train_NB(train_matrix, class_vector)
    print('p0v:\n',p0v)
    print('p1v:\n',p1v)
    print('classVec:\n',class_vector)
    print('pab:\n',pab)