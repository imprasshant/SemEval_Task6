from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding,Dense, Input, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers import Activation,Bidirectional,MaxPooling1D,Permute,RepeatVector,multiply,Lambda
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from keras.layers.merge import concatenate
from numpy import argmax
import pandas as pd


Embedding_Dim = 300
max_length = 40
seed = 7
Epoch = [3,5]
Batch_size = [16,32]
np.random.seed(seed)
NUM_WORDS = 50000
#The input file for reading.The csv file is containing the class in numbers staring from 0. 
with open("File_name.txt","r")as f:
	texts = f.readlines()

tokenizer = Tokenizer(NUM_WORDS)
tokenizer.fit_on_texts(texts) #Fit the tokenizer on text that will be processed
sequences = tokenizer.texts_to_sequences(texts) # here the conversion to tokens to their unique index
word_index = tokenizer.word_index
#Invert of dictionary with keys and values reversed.
invert = dict(map(reversed, word_index.items()))
#Padding of all the sentences to max length
data = pad_sequences(sequences, maxlen=max_length,padding = 'post')
data_1 = pd.read_csv("File_class.csv")   ##This is a file containing the classes of each instanec in the same order as in text file.
labels = data_1['class'].values
labels_one_hot_encoded = np_utils.to_categorical(labels,2)
print labels_one_hot_encoded
#Maximum number of words to get processed

vocab_size=min(len(word_index)+1,(NUM_WORDS))
#Loading Glove embedding into memory
embeddings_index = dict()
f = open('glove.twitter.27B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
embeddings_matrix = zeros((vocab_size, 100))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embeddings_matrix[i] = embedding_vector
print "The embedding matrix is ", embeddings_matrix
print "The length of embedding matrix is ", len(embeddings_matrix)

filenum = 1
#StratifiedKfold the training data to have equal proportion of all the classes in (k-1)folds as training data and last as texting data.
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
z = np.zeros([2,2],dtype= int)
score_array = []
for epoch in Epoch:
	for bs in Batch_size:
		for train,test in kfold.split(data,labels):
			input_1 = Input(shape=(max_length,), dtype='int32')
			embedding = Embedding(input_dim=vocab_size, output_dim=100, weights=[embeddings_matrix], input_length=max_length, mask_zero=False, 				trainable=True)(input_1)
			bigram_branch = Conv1D(filters=100, kernel_size=2, padding='same', activation='relu', strides=1)(embedding)
			pooling_1 = MaxPooling1D(pool_size = 4)(bigram_branch)
			trigram_branch = Conv1D(filters=100, kernel_size=3, padding='same', activation='relu', strides=1)(embedding)
			pooling_2 = MaxPooling1D(pool_size = 4)(trigram_branch)
			fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu', strides=1)(embedding)
			pooling_3 = MaxPooling1D(pool_size = 4)(fourgram_branch)
			merged = concatenate([pooling_1, pooling_2,pooling_3], axis=1)
			activations = (Bidirectional(LSTM(100, return_sequences = True, input_shape=(max_length, 100))))(merged)
			attention = Dense(1, activation='tanh')(activations) 
			attention = Flatten()(attention)
			attention = Activation('softmax')(attention)
			attention = RepeatVector(200)(attention)
			attention = Permute([2, 1])(attention)
			attn_Multiply = multiply([activations,attention]) 
			sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(attn_Multiply)
			probabilities = Dense(2, activation='softmax')(sent_representation)
			model = Model(inputs=input_1, outputs=probabilities)
			model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
			print "The model summary is ",model.summary()
			filepath='CNN_BiSTM_Attn'+'_'+str(epoch)+'_'+str(bs)+".hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			model.fit(data[train],labels_one_hot_encoded[train],validation_data = (data[test], labels_one_hot_encoded[test]),callbacks=callbacks_list,epochs = epoch,batch_size = bs,verbose = 2)
			model = load_model('CNN_BiSTM_Attn'+'_'+str(epoch)+'_'+str(bs)+".hdf5")
			scores = model.evaluate(data[test], labels_one_hot_encoded[test],
				            batch_size=bs)
			cvscores.append(scores[1] * 100)
			#The prediction of testing file
			new_prediction = model.predict(data[test])
			#Conversion of softmax probability values into class with max probability.
			new_prediction = new_prediction.argmax(axis= -1)
			original_prediction =  np.argmax(labels_one_hot_encoded[test],axis = 1)
			#Mapping the index of each word in sentence back to in their english characters
			for line in data[test]:
				#print line
				list = []
				for a in line:
					if a in invert:
						a = invert[a]
						list.append(a)
				a =  " ".join(list)
			test_data=np.asarray(data[test])
			#Save each fold into csv fold with their original and predicted labels.
			sentence = []
			df=pd.DataFrame(columns=['tweets','real_label','new_label'])
			df['tweets']=sentence
			df['real_label']=original_prediction
			df['new_label']=new_prediction
			df.to_csv('CNN_BiLSTM'+str(filenum)+".csv",sep=',')
			filenum += 1
			cm1 = confusion_matrix(np.argmax(labels_one_hot_encoded[test],axis = 1),new_prediction)
			precision_recall = precision_recall_fscore_support(original_prediction,new_prediction)
			score_array.append(precision_recall_fscore_support(original_prediction,new_prediction, average=None))
			print "The confusion matrix",cm1
			z = cm1+z
			print precision_recall
		print "The final confusion matrix",z
		z = np.zeros([2,2],dtype= int)
		avg_score = np.mean(score_array,axis=0)
		print avg_score
		avg_precision =  (avg_score[0,0]+avg_score[0,1])/2
		print ("precision:",avg_precision)
		avg_recall =  (avg_score[1,0]+avg_score[1,1])/2
		print ("recall:",avg_recall)
		avg_fscore =  (avg_score[2,0]+avg_score[2,1])/2
		print ("fscore:",avg_fscore)
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
