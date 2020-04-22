import numpy as np
import tensorflow as tf
import pickle


class TextGenerator:
	def __init__(self):
		self.text = 'initiate class with random text'
		with open('application/vocab.data', 'rb') as filehandle:
		# read the data as binary data stream
			self.vocab = pickle.load(filehandle)
		self.embedded = np.load('application/embedded.npy')
		self.model = tf.keras.models.load_model('application/model.h5')
		
		
	def generateInputArray(self, inputs):
		embDim = self.embedded.shape[1]
		x_sample =[]
		for x in inputs:
			if x in self.vocab:
				x_sample.append(self.embedded[self.vocab.index(x)])
			else:
				x_sample.append(np.zeros(embDim))

		x_sample = np.array(x_sample)
		x_sample = np.expand_dims(x_sample, axis=0)
		return(x_sample)
		
	def generatecandidates(self):
		textToken = tf.keras.preprocessing.text.text_to_word_sequence(self.text,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789'+"'")   
		
		x_sample = self.generateInputArray(textToken[-5:])
		
		y_sample = self.model.predict(x_sample)
		
		# Get top 50 candidates
		ind = np.argpartition(y_sample[0,:], -50)[-50:]

		candidates=dict()
		for i in ind:
			if self.vocab[i] != "<Unkown>":
				candidates[self.vocab[i]] = y_sample[0,i]

		return(candidates)
   