import re 
import numpy as np
import random


def text_parse(string):
    '''
    Parameter:
        None
    Return:
        None
    '''
    list_tokens = re.split(r'\W+',string)
    return [tok.lower() for tok in list_tokens if len(tok)>2]

def create_vocabulary_list(dataset):
    '''
    Parameter:
        dataset: dataset
    Return:
        vocabulary_set: non-repeat words list
    '''
    vocabulary_set = set([])
    for document in dataset:
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)
def set_words_vector(vocab_list,inputset):
    '''
    Parameter:
        vocab_list: return of create_vocabulary_list
        inputset: slice words list
    Return:
        return_vector: words vector
    '''
    return_vector = [0] * len(vocab_list)
    for word in inputset:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] = 1
        else:
            print('The word: %s is not in my vocabulary!' % word)
    return return_vector

def bag_word_vector(vocab_list,inputset):
    '''
    Parameter:
        vocab_list: return of create_vocabulary_list
        inputset: slice words list
    Return:
        return_vector: words vector
    '''
        return_vector = [0] * len(vocab_list)
    for word in inputset:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] += 1
    return return_vector

def train(train_matrix,train_category):
    '''
    Parameter:
        train_matrix: train matrix, return of set_word_vector
        train_category: train label, the class vector
    Return:
        p0: non-offensive probability 
        p1: pffensive probability
        pa: the probability of offensive file
    '''
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    pa = sum(train_category)/float(num_train_docs) # the probablity of offensive file
    p0_num = np.ones(num_words); p1_num = np.ones(num_words)
    p0_denom =2.0; p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1 = np.log(p1_num/p1_denom)
    p0 = np.log(p0_num/p0_denom)
    return p0,p1,pa

def classifier(classify_vector,p0_vector,p1_vector,p_class):
    '''
    Parameter:
        classify_vector: as the input word vector
        p0_vector: non-offensice words probability vector
        p1_vector: offensive words probability vector
        p_class: the probability of offensive file
    Return:
        0: non-offensive
        1: offensive
    '''
    p1 = sum(classify_vector * p1_vector) + np.log(p_class)
    p0 = sum(classify_vector * p0_vector) + np.log(1.0-p_class)
    if p1 > p0:
        return 1
    else:
        return 0


def spam_test():
    '''
    Parameter:
        None
    Return:
        None
    '''
    doc_list=[]; class_list = []; full_text = []
    for i in range (1,26): #range 25 txt documents.
        word_list = text_parse(open('email/spam/%d.txt' % i,'r',encoding='cp1252').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)
        word_list=text_parse(open('email/ham/%d.txt' % i,'r',encoding='cp1252').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)
    vocabulary_list = create_vocabulary_list(doc_list)
    train_set = list(range(50)); test_set = []
    for i in range(10):
        


if __name__ == '__main__':
    spam_test()