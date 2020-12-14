from sklearn import tree
import pandas as pd 

if __name__ =='__main__':
    fr = open('lenses.txt')
    lenses = [instance.strip().split('\t') for instance in fr.readlines()]
    print(lenses)
    lenses_labels = ['age','prescript','astigmatic','tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lenses,lenses_labels)