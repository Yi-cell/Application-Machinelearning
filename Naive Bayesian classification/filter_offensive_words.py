
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



if __name__ == '__main__':
    posting_list, class_vector = load_dataset()
    ''' for each in posting_list:
        print(each)
        print(class_vector)
    '''
    print('postinglist:\n', posting_list)
    my_vocab_list = create_vocabulary_list(posting_list)
    print('my_vocab_list: \n', my_vocab_list)
    train_matrix = []
    for posting_doc in posting_list:
        train_matrix.append(set_words_vec(my_vocab_list,posting_doc))
    print('train matrix:\n',train_matrix)