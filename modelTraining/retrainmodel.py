# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
import pickle
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random as rand
from random import randint
from random import seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import date

rand.seed(4)
import warnings
warnings.filterwarnings('ignore')

print('Reading new weekly data')
text = open('weekdata.txt', 'rb').read().decode(encoding='utf-8')

#===================================================================================================#
print('Generating Vocab of most frequent words')

tokens = text_to_word_sequence(text,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789'+"'")

# Read current Vocabulary of data
current_vocab = []

with open('vocab.data', 'rb') as filehandle:
    # read the data as binary data stream
    current_vocab = pickle.load(filehandle)

# Start creating new vocab to later be appended to master vocab

new_vocab_dict= {}

for word in tokens:
    if word in current_vocab:
      pass
    else:
      if word in new_vocab_dict:
          new_vocab_dict[word] += 1
      else:
          new_vocab_dict[word] = 1

new_vocab_words = []
new_vocab_freq=[]

for k,v in new_vocab_dict.items():
    new_vocab_freq.append(v)
    new_vocab_words.append(k)

wordFreqDf = pd.DataFrame(list(zip(new_vocab_words, new_vocab_freq)), columns =['Words', 'Freq'])
wordFreqDf = wordFreqDf.sort_values(by='Freq', ascending=False)

# Get words the top 20% frequently used words
TopwordFreqDf = wordFreqDf[wordFreqDf['Freq']>=np.percentile(wordFreqDf['Freq'].values,80)]
new_vocab = TopwordFreqDf['Words'].values

#====================================================================================================#
print('Generating emmbeding matrix based on Glove')

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    model = {}
    with open(gloveFile,encoding="utf8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model
	
gloveEmb = loadGloveModel('glove.6B.100d.txt')

"""Because of the data and the techniques applied during cleaning, it is possible to have some 
Out-Of-Vocabulary words (OOV), that means words that are not in the downloaded Glove dictionary"""

new_embedded = np.zeros(shape=(len(new_vocab),100))

for i in range(0,len(new_vocab)):
    try:
        new_embedded[i] = gloveEmb[new_vocab[i]]
    except:
        continue

vocab = np.append(current_vocab, new_vocab)

# Read current embedded spaces of current vocab data

current_embedded = np.load('embedded.npy')
current_embedded.shape

embedded = np.zeros(shape=(len(vocab),100))

embedded[0:len(current_vocab)] = current_embedded
embedded[len(current_vocab):] = new_embedded

#====================================================================================================#
print('Define function to retrain model with new data')

# Separate text file by lines
# open new data lines
new_TextLines = open('weekdata.txt',encoding="utf8").readlines()
# open current data lines
TextLines = open('masterdata.txt',encoding="utf8").readlines()

TextLines.extend(new_TextLines) 

# Validation list
TrainLines, TestLines = train_test_split(TextLines, test_size=0.10, random_state=42)

def generate_text_sequences(Lines, pastWords, vocab):
    X_line = list()
    Y_line = list()
    pastWords = pastWords
    for line in Lines:
        # Tokenize line
        lineTokenized = text_to_word_sequence(line,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789'+"'")
        #Get line length
        lengthLine = len(lineTokenized)
        lineBatch = lengthLine-pastWords
        
        # Substitute words outside vocab with <Unkown>
        for idx in range(0,len(lineTokenized)):
            if lineTokenized[idx] in vocab:
                continue
            else:
                lineTokenized[idx] = '<Unkown>'
        
        #Crate sequences of text 
        for i in range(0,lineBatch):
            X_sequence = lineTokenized[i:i+pastWords]
            X_line.append(X_sequence)
            Y_sequence = lineTokenized[i+pastWords]
            Y_line.append(Y_sequence)
    
    return(X_line, Y_line)
	
pastWords = 5  # number of words to look back for prediction
X_lineTrain, Y_lineTrain = generate_text_sequences(TrainLines, pastWords, vocab)

#Creating a batch generator for training data. Converting the whole dataset will take too much memory

def batch_generator_data(batchsize, X_line, Y_line, embDim, pastWords, embedded, vocab):
    embDim=embDim
    pastWords = pastWords
    x_batch = np.zeros(shape=(batchsize,pastWords,embDim))
    y_batch = np.zeros(shape=(batchsize))

    while True:
        # Fill the batch with random continuous sequences of data.

        # Get a random start-index.
        # This points somewhere into the data.
        idx = np.random.randint(len(X_line) - batchsize)

        for i in range(0,batchsize):
            x_batch[i] = [embedded[list(vocab).index(x)] for x in X_line[idx+i]]
            y_batch[i] = list(vocab).index(Y_line[idx+i])

        #y_batch = to_categorical(y_batch, num_classes=len(vocab))
        
        yield (x_batch, y_batch)

embDim = 100 #shape of the embbeded latent space
batchsize = 300 #batch size for each training step
generator = batch_generator_data(batchsize,X_lineTrain, Y_lineTrain, embDim, pastWords, embedded, vocab)

# Generate Validation data
X_lineTest, Y_lineTest = generate_text_sequences(TestLines, pastWords, vocab)
valgenerator = batch_generator_data(batchsize,X_lineTest, Y_lineTest, embDim, pastWords, embedded, vocab)

#====================================================================================================#
print('Create model to retrain with new data')

model = Sequential()
model.add(Bidirectional(LSTM(units=500, return_sequences=True),input_shape=(pastWords,embDim)))
model.add(LSTM(units=200))
model.add(Dense(len(vocab), activation='softmax'))

optimizer = RMSprop(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

model_file = "newmodel.h5"

mc = ModelCheckpoint(model_file, monitor="loss", mode="min", save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=4, min_lr=1e-4)

es = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, mode='min')

history = model.fit_generator(generator=generator,
                    epochs=30,
                    steps_per_epoch= len(X_lineTrain)//batchsize,
                    validation_data=valgenerator,
                    validation_steps= len(X_lineTest)//batchsize,
                    callbacks=[mc, reduce_lr])


#====================================================================================================#
print('Evaluate model')

'''For this project we'll like to get the top 50 predictions for the next word. The model will be evaluated by checking 
the if the next word relies in the candidates, as well as the probability given to the actual word.'''

def generateInputArray(inputs):
  embDim = embedded.shape[1]
  x_sample =[]
  for x in inputs:
    if x in vocab:
      x_sample.append(embedded[list(vocab).index(x)])
    else:
      x_sample.append(np.zeros(embDim))

  x_sample = np.array(x_sample)
  x_sample = np.expand_dims(x_sample, axis=0)
  return(x_sample)
  
def generatecandidates(text):
  textToken = tf.keras.preprocessing.text.text_to_word_sequence(str(text),  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789'+"'")   
  
  x_sample = generateInputArray(textToken[-5:])
  
  y_sample = model.predict(x_sample)
  
  # Get top 50 candidates
  ind = np.argpartition(y_sample[0,:], -50)[-50:]

  candidates=dict()
  for i in ind:
    if vocab[i] != "<Unkown>":
      candidates[vocab[i]] = y_sample[0,i]

  return(candidates)


X_line, Y_line = generate_text_sequences(TestLines, pastWords, vocab)


points=0
TotalPredictions = 0
for i in range(0, len(TestLines)):
    # Verify if the correct word is in the top-50 words predicted
    candidates = generatecandidates(X_line[i])
    if Y_line[i] == '<Unkown>':
        continue
    else:
        if Y_line[i] in candidates.keys():
            word_prob = list(candidates.keys()).index(Y_line[i])
            p = np.array(list(candidates.values()))
            points += 1 - p[word_prob] 
            TotalPredictions +=1
        else:
            points+=1
            TotalPredictions +=1

with open("model_metric.txt","a") as f:
	f.write("Metric of model of "+str(date.today())+" is: "+str(points)+"\n") 

#====================================================================================================#
print('Save new vocab, embbeding and data')

# Append new data lines to current data
data = open("masterdata.txt","a",encoding="utf8") 
for lines in new_TextLines:
  data.writelines(lines) 
data.close() #to change file access modes

# Save new vocab
with open('vocab.data', 'wb') as filehandle:
   # store the data as binary data stream
   pickle.dump(vocab, filehandle)

# Save new embedded
np.save('embedded.npy', embedded)
