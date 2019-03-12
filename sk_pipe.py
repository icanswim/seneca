# Author: scott j ward

import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.externals import joblib

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import ShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN

from sklearn.metrics import confusion_matrix, classification_report

from sk_model import SkModel
from sk_utils import type_of_target


class SkPipe():
	
	def __init__(self):
		
		self.estimators = None

	@classmethod																
	def create_sets(cls, X, y, test_split):
		#TODO also pd dataframes
		ss = ShuffleSplit(n_splits=1, test_size=test_split, random_state=88)
		for train_index, test_index in ss.split(X, y):
			X_train = X[train_index]
			y_train = y[train_index]
			X_test = X[test_index]
			y_test = y[test_index]
			
		print 'X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(
			X_train.shape, X_test.shape, y_train.shape, y_test.shape)
		return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test
	
	@classmethod
	def transform_labels(cls, y):
		# sklearn estimators do y transformation internally
		target_type = type_of_target(y)
		
		if target_type == 'continuous':
			y_transformed = y
			print 'continuous target type...'
			
		elif target_type == 'continuous-multioutput':
			y_transformed = y
			print 'continuous-multioutput target type...'
			
		elif target_type == 'binary':
			y_transformed = y
			print 'binary target type...'
			
		elif target_type == 'multiclass':
			enc = LabelBinarizer(sparse_output=False)
			#~ y = np.reshape(y, [1, -1])
			y_transformed = enc.fit_transform(y)
			print 'multiclass target type.  transforming y to onehots...'
			
		elif target_type == 'multiclass-multioutput':
			y_transformed = y
			print 'multiclass-multioutput target type...'
			
		elif target_type == 'multilabel-indicator':
			y_transformed = y
			print 'multilabel-indicator target type...'
			
		elif target_type == 'unknown':
			y_transformed = y
			print 'unknown target type...'
		
		return y_transformed
		
	@classmethod	
	def transform_data(cls, X, y=None, transform_labels=False, 
						continuous_features=None, discrete_features=None):
		
		pipelines = []
			
		if continuous_features != None:
			continuous_pipeline = Pipeline([
				('selector', DataFrameSelector(continuous_features)),
				('imputer', SimpleImputer(strategy="median")),
				('std_scaler', StandardScaler()),
				('normalize', Normalizer()),
				('f_regression_filter', SelectPercentile(
						score_func=f_regression, percentile=90)),
				('mutual_info_filter', SelectPercentile(
						score_func=mutual_info_regression, percentile=90)), 
				#~ ('pca', PCA(n_components=0.9)),
				('clusterer', Clusterer('kmeans', 
						n_clusters=len(np.unique(y)), plot=False)),
				])		
			
			pipelines.append(("continuous_pipeline", continuous_pipeline))
		
		if discrete_features != None:
			discrete_pipeline = Pipeline([
				('selector', DataFrameSelector(discrete_features)),
				('f_classif_filter', SelectPercentile(
						score_func=f_classif, percentile=90)),
				('mutual_info_filter', SelectPercentile(
						score_func=mutual_info_classif, percentile=90)), 
				('encode_discrete', EncodeDiscrete()),
				])
				
			pipelines.append(("discrete_pipeline", discrete_pipeline))	
					
		features_pipeline = FeatureUnion(pipelines)		
		X = features_pipeline.fit_transform(X, y)
		X = np.float32(X)
		
		if transform_labels:
			y = SkPipe.transform_labels(y)

		print 'X transformed: {}...'.format(X.shape)
		try: print 'y transformed: {}...'.format(y.shape)
		except: print 'no y to transform...'
		
		return X, y
	
	@classmethod
	def transform_text(cls, texts, labels=None, transform_labels=False):
		
		text_pipeline = Pipeline([
					#~ ('hash', HashingVectorizer(
							#~ stop_words='english', alternate_sign=False,
							#~ ngram_range=(1,1), n_features=2**18)),
					('tfidf', TfidfVectorizer(
							ngram_range=(1,1), max_df=.5, min_df=2,
							stop_words='english')),
					('chi2_filter', SelectPercentile(
							score_func=chi2, percentile=90)),
					#~ ('svd', TruncatedSVD(n_components=10000)),
					#~ ('lda', LatentDirichletAllocation(n_components=20, 
							#~ verbose=3, n_jobs=-1)),
					])
				
		texts = text_pipeline.fit_transform(texts, labels)
		
		if transform_labels:
			labels = SkPipe.transform_labels(labels)

		print 'text transformed: {}...'.format(texts.shape)
		try: print 'labels transformed: {}...'.format(labels.shape)
		except: print 'no labels to transform...'
		
		return texts, labels
	
	def train(self, X_train, y_train, model_names):
		 
		self.estimators = SkModel.estimator_grid_search(X_train, y_train, model_names)
		
	def predict(self, X_test):
		
		self.predictions = {}
		
		for est in self.estimators:
			self.predictions[est] = self.estimators[est].predict(X_test)
			print '{}: prediction complete...'.format(est)
		
	def score(self, X_test, y_test):
		
		self.scores = {}
		
		for est in self.estimators:
			self.scores[est] = self.estimators[est].score(X_test, y_test)
			print '{}: score: {}'.format(est, self.scores[est])
		
	def estimator_report(self, X_test, y_test):
		
		for est in self.estimators:
			
			mean_cv_scores = self.estimators[est].cv_results_['mean_test_score']
			test_score = self.estimators[est].score(X_test, y_test)
			print '{}: mean cv score: {}, test score: {}'.format(
								est, mean_cv_scores, test_score)
				
			best_params = self.estimators[est].best_estimator_
			print 'best_params: \n {}'.format(best_params)
			
			y_pred = self.estimators[est].predict(X_test)
			class_report = classification_report(y_test, y_pred, labels=np.unique(y_test))
			print 'classification report: \n{}'.format(class_report)	
		
	def save_model(self, model_name='sk_model.pkl'):
		
		joblib.dump(self.estimators, model_name)
		print 'model: {} saved...'.format(model_name)
	
	def load_model(self, model_name='sk_model.pkl'):
		
		self.estimators = joblib.load(model_name)
		print 'model: {} loaded...'.format(model_name)
			
	@classmethod	
	def transform_image(cls, image, label):
		# TODO
		pass
		

class DataFrameSelector(BaseEstimator, TransformerMixin):
	# feature_list should either be a list of colnames if X is a df
	# or a list of indexes if X is an array or 'all'
	def __init__(self, feature_list=None):
		self.feature_list = feature_list
	
	def fit(self, X, y=None):
		return self
		
	def transform(self, X):
		
		if isinstance(X, pd.DataFrame):
			if self.feature_list == 'all':
				return X.as_matrix()
			else:
				return X[self.feature_list]

		else:
			if self.feature_list == 'all':
				return X
			else:
				return X[:,self.feature_list]
	
	
class EncodeDiscrete(BaseEstimator, TransformerMixin):
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		# if X are int they will not be converted to onehots
		v = DictVectorizer(sparse=False)
		D = pd.DataFrame(X).to_dict('records')
		X_transformed = v.fit_transform(D)
		
		print 'X_trans', type(X_transformed)
		return X_transformed
	

class Clusterer(BaseEstimator, TransformerMixin):

	def __init__(self, model, n_clusters=None, plot=False):
	
		self.model = model
		self.n_clusters = n_clusters
		self.plot = plot
		
	def fit(self, X, y=None):
		return self
		
	def transform(self, X):
		
		models = {'kmeans': MiniBatchKMeans,
				'dbscan': DBSCAN}
		
		params = {'kmeans': {'n_clusters': self.n_clusters},
				'dbscan': {'eps': 0.5, 'min_samples': 10}}
				
		clusterer = models[self.model](**params[self.model])
		
		try:
			clusterer.fit_transform(X)
		except:
			clusterer.fit(X)
			
		if self.plot == True:
			plt.scatter(X[:,0], X[:,1], c=clusterer.labels_)
			plt.show()
			
		_labels = np.reshape(clusterer.labels_, (-1,1))
		unique, counts = np.unique(_labels, return_counts=True)
		print 'cluster: counts', dict(zip(unique, counts))
		
		enc = OneHotEncoder(sparse=False, categories='auto')
		_labels_encoded = enc.fit_transform(_labels)
		
		X_transformed = np.c_[_labels_encoded, X]
		
		print '{} cluster features added...'.format(_labels_encoded.shape[1])
						
		return X_transformed
 
           
	
