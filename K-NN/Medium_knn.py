import numpy as np

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
        elif list_from_line[-1] == 'largedoses':
            class_label_vector.append(3)
        index += 1
    return return_mat,class_label_vector

if __name__ == '__main__':
    filename = 'datingTestSet.txt'
    dating_datamat, dating_labels = file_matrix(filename)
    print(dating_datamat)
    print(dating_labels)


