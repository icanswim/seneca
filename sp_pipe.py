# Author: scott j ward

from __future__ import unicode_literals

import spacy
import numpy as np


class SpPipe():
	
	def __init__(self):
		
		self.nlp = spacy.load('en_core_web_md', disable=['ner','parser','tagger','textcat'])

	def __call__(self, texts, labels, steps=10):
		
		print 'tokenizing with spacy...'
		docs = [self.nlp(unicode(text)) for text in texts]
	
		X, y = self._get_features(docs, labels, steps)
		
		return X, y
		
	def _get_features(self, docs, labels, steps):
		
		X = np.zeros((len(labels), steps), dtype='int32')
		
		for n, doc in enumerate(docs):
			m = 0
			for token in doc:
				vector_id = token.vocab.vectors.find(key=token.orth)
				if vector_id >= 0:
					X[n, m] = vector_id
				else:
					X[n, m] = 0
				m += 1
				if m >= steps:
					break
					
		return X, labels

	def get_embedding(self):
		
		return self.nlp.vocab.vectors.data
