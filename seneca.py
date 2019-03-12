# Author: scott j ward

import json
import os
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.datasets import make_regression, make_classification
from sklearn.datasets import make_multilabel_classification, fetch_20newsgroups

from sk_pipe import SkPipe
from sp_pipe import SpPipe
from tf_pipe import TfPipe


np.random.seed(88)

class Seneca():
	
	def __init__(self):
		
		params = {
				'model': 'ffnet',
				'batch_size': 100, 
				'epochs': 10, 
				'nlp_steps': 1,
				'n_classes': 4, 
				'column_names': [], 
				'loss_func': 'sparse',
				'hidden_units': [1024,512,256],
				'target': '',
				'feature_columns': [], #populated by _input_func()
				'model_dir': None
				}
				
		X, y = self.load_synth_data('classification')
		X_train, y_train, X_test, y_test = SkPipe.create_sets(X, y, 0.2)
		
		
		tfpipe = TfPipe(params)
		
		#~ tfpipe.train_test(X_train, y_train, X_test, y_test)
		
		for i in range(X.shape[1]):
			params['column_names'].append('feature_{}'.format(i))
		X_train = pd.DataFrame(X_train, columns=params['column_names'])
		X_test = pd.DataFrame(X_test, columns=params['column_names'])
		
		tfpipe.train_estimator(X_train, y_train)
		tfpipe.evaluate_estimator(X_test, y_test)
		
	@staticmethod
	def load_csv(filename):

		return pd.read_csv(filename)
		
	@staticmethod
	def save_csv(data, filepath):

		if not isinstance(data, pd.DataFrame):
			data = pd.DataFrame(data)
		data.to_csv(filepath)
	
	@staticmethod		
	def load_txt(filename):
		
		with open(filename, 'r') as f:
			text = f.read().decode('utf8')
		
		return text
		
	@staticmethod		
	def load_json(filename):
		
		with open(filename, 'r') as f:
			text = json.load(f)
		
		return text
		
	@staticmethod
	def save_json(text, filename):
		
		with open(filename, 'w') as f:
			json.dump(text, f)
			
	@staticmethod	
	def load_synth_data(dataset):
		
		datasets = {'blobs': make_blobs,
					'moons': make_moons,
					'circles': make_circles,
					'regression': make_regression,
					'classification': make_classification,
					'multilabel_classification': make_multilabel_classification}
				
		params = {'blobs':{'n_samples':1000, 'n_features':4, 'centers': 5}, # multiclass
				'moons':{'n_samples':100}, # binary 
				'circles':{'n_samples': 100}, # binary
				'regression':{'n_samples': 100, 'n_features': 20,
							'n_informative': 18, 'n_redundant': 0,
							'n_targets':1}, # continuous-multioutput
				'classification':{'n_samples': 10000, 'n_features': 10,
								'n_informative': 8, 'n_redundant': 1, 
								'n_classes': 4,'n_clusters_per_class': 1, 
								'flip_y': 0.05}, # multiclass
				'multilabel_classification':{'n_samples': 100, 'n_features': 20,
											'n_classes': 5, 'n_labels': 2}} # multiclass onehot
					
		X, y = datasets[dataset](**params[dataset])
	
		print 'X.shape: {}, y.shape: {}'.format(X.shape, y.shape)
		return X, y
	
	@staticmethod
	def explore_df(data):
		
		if not isinstance(data, pd.DataFrame):
			data = pd.DataFrame(data)
	
		print 'data.head() \n', data.head()
		print 'data.info() \n', data.info()
		print 'data.describe() \n', data.describe()
		print 'data.corr() \n', data.corr()
			
							
if __name__ == '__main__':
	
	seneca = Seneca()
