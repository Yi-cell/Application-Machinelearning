from sklearn import tree
import pandas as pd 

if __name__ =='__main__':
    fr = open('lenses.txt')
    lenses = [instance.strip().split('\t') for instance in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lenses_labels = ['age','prescript','astigmatic','tearRate']
    lenses_list=[] #save the lenses data temporarily
    lenses_dict={} #save the lenses data dictionary

    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list=[]
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    