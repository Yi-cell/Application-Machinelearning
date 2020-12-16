
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
            print('The word: %s is not in my vocabulary list!') % word)
    return return_vec

    
if __name__ == '__main__':
    posting_list, class_vector = load_dataset()
    for each in posting_list:
        print(each)
    print(class_vector)
    