import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines 
import matplotlib.pyplot as plt 

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
        elif list_from_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        index += 1
    return return_mat,class_label_vector

def data_visualization(dating_datamat,dating_labels):
    '''
    Return: Graphs
    '''

    number_labels = len(dating_labels)
    labels_colors =[]
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        elif i == 2:
            labels_colors.append('orange')
        elif i == 3:
            labels_colors.append('red')
    # Scatter plot (First column: flight miles, Second column: playing game)
    axs[0][0].scatter(x=dating_datamat[:,0],y=dating_datamat[:,1], color=labels_colors,s=15,alpha=.5)
    # set title, x_label, y_label
    plt.setp(axs[0][0].set_title(u'The proportion of the flight miles and time of playing game'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[0][0].set_xlabel(u'flight miles each year'),size = 9, weight ='bold', color='black')
    plt.setp(axs[0][0].set_ylabel(u'time of playing game'),size = 9, weight ='bold', color='black')
    # Scatter plot (First column: flight miles, Second column: the volume of icecream consumption )
    axs[0][1].scatter(x=dating_datamat[:,0],y=dating_datamat[:,2], color=labels_colors,s=15,alpha=.5)
    plt.setp(axs[0][1].set_title(u'The proportion of the flight miles and consumption of icecream'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[0][1].set_xlabel(u'flight miles each year'),size = 9, weight ='bold', color='black')
    plt.setp(axs[0][1].set_ylabel(u'consumption of icecream'),size = 9, weight ='bold', color='black')
    #
    axs[1][0].scatter(x=dating_datamat[:,0],y=dating_datamat[:,2], color=labels_colors,s=15,alpha=.5)
    plt.setp(axs[1][0].set_title(u'The proportion of the time of playing game and consumption of icecream'),size = 9, weight = 'bold',color='red')
    plt.setp(axs[1][0].set_xlabel(u'time of playing game'),size = 9, weight ='bold', color='black')
    plt.setp(axs[1][0].set_ylabel(u'consumption of icecream'),size = 9, weight ='bold', color='black')
    # setting legends
    didntLike = mlines.Line2D([],[],color='black',marker='.', markersize = 6, label = 'didntLike')
    smallDoses = mlines.Line2D([],[],color ='orange',marker='.',markersize=6,label = 'smallDoses')
    largeDoses = mlines.Line2D([],[],color ='red',marker='.',markersize=6,label = 'largeDoses')
    # placing legends
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()




if __name__ == '__main__':
    filename = 'datingTestSet.txt'
    dating_datamat, dating_labels = file_matrix(filename)
    print(dating_datamat)
    print(dating_labels)
    data_visualization(dating_datamat,dating_labels)


