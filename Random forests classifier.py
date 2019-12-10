# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:55:09 2019

@author: Formateur IT
"""
import pandas as pd

df = pd.read_csv('prediction_de_fraud_2.csv')

#Création des données predictives et de la données à prédire

caracteristiques = df.drop('isFraud', axis = 1).values
cible = df['isFraud'].values

from sklearn.preprocessing import LabelEncoder

labEncr_X = LabelEncoder()
caracteristiques[:,1] = labEncr_X.fit_transform(caracteristiques[:,1])
caracteristiques[:,3] = labEncr_X.fit_transform(caracteristiques[:,3])
caracteristiques[:,6] = labEncr_X.fit_transform(caracteristiques[:,6])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(caracteristiques, cible, test_size = 0.3, random_state = 42, stratify = cible)

############L'étape suivante consiste à créer le classifieur de forêt aléatoire.


#Nous importons d'abord RandomForestClassifierde scikit-learn.
from sklearn.ensemble import RandomForestClassifier


# Initialisation d'un classifieur de forêt aléatoire avec des paramètres par défaut
Random_FrsCls = RandomForestClassifier(random_state = 50)

#Nous adaptons ensuite ce modèle à nos données d’entraînement  
Random_FrsCls.fit(X_train, y_train)



######évaluons son exactitude à partir des données de test.

#évaluons son exactitude à partir des données de test.
test_score = Random_FrsCls.score(X_test, y_test)

print("Test score: %.2f%%" % (test_score * 100.0))



y_pred = Random_FrsCls.predict(X_test)

# Evalution avec Matrice de Confusion 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)




from sklearn.model_selection import RandomizedSearchCV

#Nous initialisons un dictionnaire de valeurs d'hyperparamètre.

grid_params = {
 'n_estimators': [100,200, 300,400,5000],
 'max_depth': [1,2,4,6,8],
 'min_samples_leaf': [0.05, 0.1, 0.2]
}


import parallelTestModule

import multiprocessing as mp


if __name__ == '__main__':
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)
    rf_random = RandomizedSearchCV(estimator = Random_FrsCls, param_distributions = grid_params, n_iter = 1, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    
    
#extrayons les paramètres optimaux.  
rf_random.best_params_









































