# Author: scott j ward

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge, RidgeClassifier, SGDClassifier

from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.neural_network import MLPClassifier


class SkModel():
	
	@classmethod
	def estimator_grid_search(cls, X_train, y_train, model_names):
		
		models = {'ridge_classifier': RidgeClassifier(),
				'ridge': Ridge(),
				'extra_trees_classifier': ExtraTreesClassifier(),
				'extra_trees_regressor': ExtraTreesRegressor(),
				'mlp': MLPClassifier(),
				'bayes': GaussianNB(),
				'multi_bayes': MultinomialNB(),
				'svc': SVC(),
				'sgd': SGDClassifier()}	
			
		params = {'ridge_classifier': {'alpha': [0.1,1.0]},
				'ridge': {'alpha': [0.1,1.0]},
				'extra_trees_classifier': {'n_estimators': [100]},
				'extra_trees_regressor': {'n_estimators': [100]},
				'mlp': {'hidden_layer_sizes': [(1024,512,256)], 'max_iter': [10]},
				'bayes': {},
				'multi_bayes': {'alpha': [.01, 0.1, 1]},
				'svc': {'C': [1], 'kernel': ['linear']},
				'sgd': {'loss': ['hinge','log','perceptron'], 
						'max_iter': [50], 'tol': [.01],'n_jobs': [-1]}}
		
		estimators = {}
		for model in model_names:
			print "running sk model: {}...".format(model)
			estimator = models[model]
			param = params[model]
			try:
				gs = GridSearchCV(estimator, param, cv=3, n_jobs=-1, 
						verbose=1, scoring=None, refit=True)
				gs.fit(X_train, y_train)
				estimators[model] = gs
			except:
				print '{} failed..'.format(model)
		
		return estimators
		
	@classmethod			
	def voting_classifier(self):
		# TODO
		pass
	
	@classmethod
	def online_learning(self):
		# TODO
		# iterate over data sending batches to the partial_fit method
		pass
