# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:30:54 2019

@author: TSHIBA-PC
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

dataset = pd.read_csv('prediction_de_fraud.csv')

#récupération les données predictives
X = dataset.drop('isFraud', axis = 1).values
target = dataset['isFraud'].values

from sklearn.preprocessing import LabelEncoder
labEncr_X = LabelEncoder()
X[:,1] = labEncr_X.fit_transform(X[:,1])
X[:,3] = labEncr_X.fit_transform(X[:,3])
X[:,6] = labEncr_X.fit_transform(X[:,6])


#Initialisation du classifieur DT
decTree_2 = DecisionTreeClassifier(criterion = 'gini', random_state = 50, max_depth= 2, min_samples_leaf=0.02)


# Adapter le classificateur aux données
decTree_2.fit(X, target)

#Extrait du nom des données predictives

X_names = dataset.drop('isFraud', axis = 1)

data = tree.export_graphviz(decTree_2, out_file=None, feature_names= X_names.columns.values, proportion= True)



graph = pydotplus.graph_from_dot_data(data) 

Image(graph.create_png())












