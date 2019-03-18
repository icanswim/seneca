# seneca
A data science framework in tensorflow, sklearn and spacy.

	#Ex: sklearn pipeline and model grid search
	X, y = self.load_synth_data('classification')

	skpipe = SkPipe()
	X, y = skpipe.transform_data(X, y, continuous_features='all')
	X_train, y_train, X_test, y_test = skpipe.create_sets(X.astype(np.float32), y, 0.2)
	skpipe.train(X_train, y_train, ['sgd','mlp','extra_trees_classifier'])
	skpipe.estimator_report(X_test, y_test)

	#Ex: tensorflow session api pipeline and custom model
	params = {
		'model': 'ffnet',
		'batch_size': 100, 
		'epochs': 10, 
		'classes': 4, 
		'column_names': [], 
		'loss_func': 'sparse',
		'units': [1024,512,256],
		'model_dir': None
		}

	X, y = self.load_synth_data('classification')

	X_train, y_train, X_test, y_test = SkPipe.create_sets(X.astype(np.float32), y, 0.2)

	tfpipe = TfPipe(params)
	tfpipe.train_test(X_train, y_train, X_test, y_test)
	tfpipe.sess.close()

	#Ex: sklearn pipeline with tensorflow estimator api and canned or custom model
	params = {
		'model': 'dnn_class', #'ffnet'
		'batch_size': 100, 
		'epochs': 10, 
		'classes': 4, 
		'column_names': [], 
		'loss_func': 'sparse',
		'units': [1024,512,256],
		'model_dir': None,
		'feature_columns': [] # populated by _input_func() as part of the tf.estimator api
		}

	X, y = self.load_synth_data('classification')

	skpipe = SkPipe()
	X, y = skpipe.transform_data(X, y, continuous_features='all')
	X_train, y_train, X_test, y_test = skpipe.create_sets(X.astype(np.float32), y, 0.2)	

	for i in range(X.shape[1]):
		params['column_names'].append('feature_{}'.format(i))
	X_train = pd.DataFrame(X_train, columns=params['column_names'])
	X_test = pd.DataFrame(X_test, columns=params['column_names'])

	#Ex: nlp with spacy and tensorflow session api pipeline and custom model
	params = {
		'model': 'lstm',
		'batch_size': 5, 
		'epochs': 10, 
		'steps': 100,
		'classes': 20, 
		'column_names': [], 
		'loss_func': 'sparse',
		'units': 64,
		'layers': 5,
		'model_dir': None
		}

	data = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))

	sp_pipe = SpPipe()
	X, y = sp_pipe(data.data[:10000], data.target[:10000], steps=params['steps'])

	X_train, y_train, X_test, y_test = SkPipe.create_sets(X, y, 0.2)

	tf_pipe = TfPipe(params, embedding=sp_pipe.get_embedding())
	tf_pipe.train_test(X_train, y_train, X_test, y_test)
	tf_pipe.sess.close()

	#Ex: nlp with sklearn pipeline and model grid search
	data = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))

	skpipe = SkPipe()
	text, labels = skpipe.transform_text(data.data[:10000], data.target[:10000])
	X_train, y_train, X_test, y_test = skpipe.create_sets(text, labels, 0.2)
	skpipe.train(X_train, y_train, ['multi_bayes', 'ridge_classifier', 'extra_trees_classifier'])
	skpipe.estimator_report(X_test, y_test)
