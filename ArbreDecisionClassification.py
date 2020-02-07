# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:30:54 2019

@author: TSHIBA-PC
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('prediction_de_fraud.csv')

X = dataset.drop('isFraud', axis = 1).values

target = dataset['isFraud'].values

labEncr_X = LabelEncoder()
X[:,1] = labEncr_X.fit_transform(X[:,1])
X[:,3] = labEncr_X.fit_transform(X[:,3])
X[:,6] = labEncr_X.fit_transform(X[:,6])


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.3, random_state = 42, stratify = target)


#Nous initialisons ensuite un DecisionTreeClassifierobjet avec deux arguments.
decTree = DecisionTreeClassifier(criterion = 'gini', random_state = 50)


#Enfin, nous ajustons le modèle sur les données d’entraînement
decTree.fit(X_train, y_train)

# évaluons sa précision sur les données de test.
decTree.score(X_test, y_test)
y_pred = decTree.predict(X_test)

# Evalution avec Matrice de Confusion 
cm = confusion_matrix(y_test, y_pred)



#Creation de grille des differents hyperparameters
grid_params = {
    'max_depth': [1,2,3,4,5,6],
    'min_samples_leaf': [0.02,0.04, 0.06, 0.08]
}



#ous créons un GridSearchCVobjet avec le classifieur de l’arbre de décision comme estimateur
grid_object = GridSearchCV(estimator = decTree, param_grid = grid_params, scoring = 'accuracy', cv = 10)


#Nous ajustons ensuite cet objet de grille aux données d'apprentissage
grid_object.fit(X_train, y_train)

#Extraction des meilleures parametres
grid_object.best_params_


