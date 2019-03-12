# Author: scott j ward

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

import tf_utils


def load_model(params, model_input=None, mode=None):
	
	models = {'ffnet': FfNet,
			'convnet': ConvNet,
			'lstm': Lstm,
			'dnn_class': DnnClass,
			'trees_class': TreesClass}
			
	return models[params['model']](params, model_input, mode)
	
class TfModel(object):
	
	__metaclass__ = ABCMeta
	
	def __init__(self, params, model_input, mode):
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			self.training = True
		else:
			self.training = False
			
		self.output = self._build_model(model_input, params)
		
	@abstractmethod	
	def _build_model(self, model_input, mode, params):
		
		return output
		
class TfEstimator(object):
	
	__metaclass__ = ABCMeta
	
	def __init__(self, params, model_input, mode):

		self.estimator = self._build_estimator(params)
	
	@abstractmethod	
	def _build_estimator(self, params):
		
		return estimator
		
class DnnClass(TfEstimator):
	
	def _build_estimator(self, params):
		
		return tf.estimator.DNNClassifier(hidden_units=params['hidden_units'],
										feature_columns=params['feature_columns'],
										model_dir=params['model_dir'],
										n_classes=params['n_classes'],
										weight_column=None,
										label_vocabulary=None,
										optimizer='Adagrad',
										activation_fn=tf.nn.relu,
										dropout=0.2,
										input_layer_partitioner=None,
										config=None,
										warm_start_from=None,
										loss_reduction=tf.losses.Reduction.SUM,
										batch_norm=False)
		
class TreesClass(TfEstimator):
	
	def _build_estimator(self, params):
		
		return tf.estimator.BoostedTreesClassifier(feature_columns,
												n_batches_per_layer,
												model_dir=None,
												n_classes=_HOLD_FOR_MULTI_CLASS_SUPPORT,
												weight_column=None,
												label_vocabulary=None,
												n_trees=100,
												max_depth=6,
												learning_rate=0.1,
												l1_regularization=0.0,
												l2_regularization=0.0,
												tree_complexity=0.0,
												min_node_weight=0.0,
												config=None,
												center_bias=False,
												pruning_mode='none',
												quantile_sketch_epsilon=0.01)

		 
class FfNet(TfModel):
		
	def _build_model(self, model_input, params):
		
		with tf.variable_scope('ffnet-model'):
			
			for layer, units in enumerate(params['hidden_units']):
				
				fc = self._fc_layer(model_input, units, 'fc_{}'.format(layer))
				model_input = tf.layers.dropout(inputs=fc, rate=0.2, 
										training=self.training, name='dropout_{}'.format(layer))
		
			output = self._fc_layer(model_input, params['n_classes'], 
									'model-output', act=tf.identity)
									
		return output
		
	def _fc_layer(self, model_input, output_dim, layer_name, act=tf.nn.leaky_relu):
		
		with tf.variable_scope(layer_name) as scope:
			
			W = tf.get_variable('weights', 
							[model_input.shape[1], output_dim], 
							initializer=tf.truncated_normal_initializer)
						
			b = tf.get_variable('biases', 
							[output_dim],
							initializer=tf.truncated_normal_initializer)
							
			preactivate = tf.matmul(model_input, W) + b
			
			fc = act(preactivate, name=scope.name)
			
		return fc	
		
class ConvNet(TfModel):
	
	def _build_model(self, model_input, params):
		
		init = tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32)
		
		with tf.variable_scope('convnet-model', initializer=init):	
			
			conv1 = tf.layers.conv2d(inputs=model_input, filters=32, 
						kernel_size=[3, 3], padding="same", 
						activation=tf.nn.leaky_relu, name='conv1')
					
			conv2 = tf.layers.conv2d(inputs=conv1, filters=32, 
						kernel_size=[3, 3], padding="same", 
						activation=tf.nn.leaky_relu, name='conv2')

			pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], 
						strides=2, name='pool1')
						
			conv3 = tf.layers.conv2d(inputs=pool1, filters=16, 
						kernel_size=[5, 5], padding="same", 
						activation=tf.nn.leaky_relu, name='conv3')
						
			conv4 = tf.layers.conv2d(inputs=conv3, filters=16, 
						kernel_size=[5, 5], padding="same", 
						activation=tf.nn.leaky_relu, name='conv4')
			  
			pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], 
						strides=2, name='pool2')
						
			conv5 = tf.layers.conv2d(inputs=pool2, filters=8, 
						kernel_size=[5, 5], padding="same", 
						activation=tf.nn.leaky_relu, name='conv5')
						
			conv6 = tf.layers.conv2d(inputs=conv5, filters=8, 
						kernel_size=[5, 5], padding="same", 
						activation=tf.nn.leaky_relu, name='conv6')
			  
			pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], 
						strides=2, name='pool3')
					
			flatten = tf.layers.flatten(inputs=pool3, name='flatten')

			dense1 = tf.layers.dense(inputs=flatten, units=1024, 
						activation=tf.nn.leaky_relu, name='dense1')
			
			dropout1 = tf.layers.dropout(inputs=dense1, rate=0.3, 
						training=self.training, name='dropout1')
			
			dense2 = tf.layers.dense(inputs=dropout1, units=512, 
						activation=tf.nn.leaky_relu, name='dense2')
			
			dropout2 = tf.layers.dropout(inputs=dense2, rate=0.2, 
						training=self.training, name='dropout2')
			
			output = tf.layers.dense(inputs=dropout2, 
						units=output_shape, name='model-output')
		return output
		
class Lstm(TfModel):
	
	def _build_model(self, model_input, params):
		
		if params['embedding'] is not None:
			model_input = tf.nn.embedding_lookup(params['embedding'], model_input)
			model_input = tf.transpose(model_input, perm=[1,0,2])
			# spacy embedding = [vocab_size, 300]
			model_input = tf.reshape(model_input, params['nlp_steps'], params['batch_size'], params['embedding'].shape[1])
		
		init = tf.contrib.layers.xavier_initializer()
		
		with tf.variable_scope('lstm-model'):
			self.model = tf.contrib.cudnn_rnn.CudnnLSTM(
												num_layers=4,
												num_units=64,
												input_mode='linear_input',
												direction='bidirectional',
												dropout=training,
												seed=None,
												dtype=tf.float32,
												kernel_initializer=init,
												bias_initializer=init,
												name='lstm')
												
			output, state = self.model(model_input, initial_state=None, 
												training=self.training)
																	
			output = tf.layers.dense(inputs=output, 
										units=output_shape, 
										activation=tf.nn.leaky_relu)
			output = tf.transpose(output)	
			output = tf.layers.dense(inputs=output, 
										units=1, 
										activation=None, 
										name='model-output')
																				
			output = tf.transpose(tf.squeeze(output, axis=2))
	
		return (output, state)							
													
																		
																	
