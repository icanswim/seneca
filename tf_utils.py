# Author: scott j ward

import tensorflow as tf

import zipfile


def read_zip(filename, encoding='unicode'):
	# encoding = format to be returned
	with zipfile.ZipFile(filename) as f:
		
		if encoding == 'utf8':
			data = tf.compat.as_str(f.read(f.namelist()[0])).split()
			
		if encoding == 'unicode':
			data = tf.compat.as_text(f.read(f.namelist()[0])).split()
			
	return data

