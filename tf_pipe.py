# Author: scott j ward

import pandas as pd
import numpy as np
import tensorflow as tf

from tf_model import load_model

		
class TfPipe:
	
	def __init__(self, params, embedding=None):
		
		#~ tf.set_random_seed(88)
		self.params = params
		self.params['embedding'] = embedding
		self.estimator = None
		self.model = None
		self.sess = None
			
	def _input_func(self, X, y=None):
		#dataset inputs must be roundly divisible by the batch_size
		if  len(self.params['feature_columns']) == 0:
			if len(self.params['column_names']) > 0:
				keys = self.params['column_names']
			else:	
				if isinstance(X, (pd.DataFrame, dict)):
					keys = X.keys()
				else:
					print 'X input must be a dict or df or you must provide the column_names..'
					
			for key in keys:
				self.params['feature_columns'].append(
								tf.feature_column.numeric_column(key=str(key)))
		
		if y is not None:
			inputs = (dict(X), y)
		else:
			inputs = (dict(X))
			
		dataset = tf.data.Dataset.from_tensor_slices(inputs)	
		dataset = dataset.shuffle(X.shape[0])
		dataset = dataset.repeat(self.params['epochs'])             
		dataset = dataset.batch(self.params['batch_size'])
		dataset = dataset.prefetch(self.params['batch_size'])
		
		return dataset
											
	def _model_func(self, features, labels, mode):
		
		X = features
		y = labels
							
		model_input = tf.feature_column.input_layer(X, self.params['feature_columns'])
		
		model = load_model(self.params, model_input, mode)
		
		predictions = tf.argmax(model.output, axis=1)
		accuracy = tf.metrics.accuracy(labels=y,
									   predictions=predictions,
									   name='acc_op')				   
		metrics = {'accuracy': accuracy}
		tf.summary.scalar('accuracy', accuracy[1])
															
		train_op, loss = self._loss_func(y, model.output)
		
		return tf.estimator.EstimatorSpec(mode=mode,
									  predictions=predictions,
									  loss=loss,
									  train_op=train_op,
									  eval_metric_ops=metrics)
	
	def _loss_func(self, y, model_output):
		
		loss_funcs = {'softmax': tf.nn.softmax_cross_entropy_with_logits_v2,
					'sparse': tf.nn.sparse_softmax_cross_entropy_with_logits, 
					'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
	
		with tf.name_scope('loss'):
			losses = loss_funcs[self.params['loss_func']](labels=y, logits=model_output)
			loss = tf.reduce_mean(losses)
		
		optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
		#~ optimizer = tf.train.AdamOptimizer()
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())	
		return train_op, loss
	
	def _build_estimator(self):
			
		if self.params['model'] in ['dnn_class','trees_class']:
			return load_model(self.params).estimator
		else:
			return tf.estimator.Estimator(self._model_func,
										model_dir=self.params['model_dir'],
										config=None,
										params=None, # not used
										warm_start_from=None)
	
	def train_estimator(self, X_train, y_train):
		
		self.estimator = self._build_estimator()
		self.estimator.train(input_fn=lambda:self._input_func(X_train, y_train))
		
	def evaluate_estimator(self, X_test, y_test):
		
		evaluation = self.estimator.evaluate(input_fn=lambda:self._input_func(X_test, y_test))
		print evaluation
		
	def predict_estimator(self, X_test):
		
		prediction = self.estimator.predict(X_test)
		print prediction
		
	def _build_dataset(self, X_train, y_train, X_test, y_test):
		#dataset inputs must be roundly divisible by the batch_size
		dataset = {}
		
		dataset['X_train'] = tf.placeholder(X_train.dtype, X_train.shape)
		dataset['y_train'] = tf.placeholder(y_train.dtype, y_train.shape)
		dataset['X_test'] = tf.placeholder(X_test.dtype, X_test.shape)
		dataset['y_test'] = tf.placeholder(y_test.dtype, y_test.shape)
		
		train_dataset = tf.data.Dataset.from_tensor_slices(
								(dataset['X_train'], dataset['y_train']))
		train_dataset = train_dataset.shuffle(X_train.shape[0])              
		train_dataset = train_dataset.batch(self.params['batch_size'])
			
		test_dataset = tf.data.Dataset.from_tensor_slices(
								(dataset['X_test'], dataset['y_test']))                
		test_dataset = test_dataset.batch(self.params['batch_size'])
		
		dataset['iterator'] = tf.data.Iterator.from_structure(
					train_dataset.output_types, train_dataset.output_shapes)
														
		dataset['train_init'] = dataset['iterator'].make_initializer(train_dataset)
		dataset['test_init'] = dataset['iterator'].make_initializer(test_dataset)
		
		dataset['label'] = tf.placeholder(dtype=tf.int32, shape=[None])
		dataset['prediction'] = tf.placeholder(dtype=tf.int32, shape=[None])
		
		return dataset
											
	def train_test(self, X_train, y_train, X_test, y_test):
	
		dataset = self._build_dataset(X_train, y_train, X_test, y_test)
		
		self.sess = tf.Session()
		with self.sess.as_default():
		
			X, y = dataset['iterator'].get_next()
			
			self.model = load_model(self.params, X)
											
			predictions = tf.argmax(self.model.output, axis=1)
			
			accuracy, accuracy_op = tf.metrics.accuracy(dataset['label'], 
												dataset['prediction'], 
												name='accuracy')
																		
			train_op, loss = self._loss_func(y, self.model.output)
				
			self.sess.run(tf.global_variables_initializer())
			
			for e in range(self.params['epochs']):
				self.sess.run(tf.local_variables_initializer())
				self.sess.run(dataset['train_init'], feed_dict={
					dataset['X_train']: X_train, dataset['y_train']: y_train})
				self.model.training = True
				try:
					while True:
						_, loss_val = self.sess.run([train_op, loss])
						label, prediction = self.sess.run([y, predictions])
						self.sess.run(accuracy_op, 
										feed_dict={dataset['label']: label, 
										dataset['prediction']: prediction})
						
				except tf.errors.OutOfRangeError:
					score = self.sess.run(accuracy)
					print 'epoch: {}, train accuracy: {}'.format(e+1, score)
				
				self.sess.run(tf.local_variables_initializer())
				self.sess.run(dataset['test_init'], feed_dict={
					dataset['X_test']: X_test, dataset['y_test']: y_test})
				self.model.training = False				
				try:
					while True:
						label, prediction = self.sess.run([y, predictions])
						self.sess.run(accuracy_op, 
										feed_dict={dataset['label']: label, 
										dataset['prediction']: prediction})
			
				except tf.errors.OutOfRangeError:
					score = self.sess.run(accuracy)
					print 'epoch: {}, test accuracy: {}'.format(e+1, score)
		
	def load_model(self, model_name):
		#TODO
		pass
		
	def save_model(self, model_name):
		#TODO
		pass	
