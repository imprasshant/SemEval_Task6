from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Dense,Bidirectional
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from keras.models import load_model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.layers.merge import concatenate
from numpy import argmax
import pandas as pd

# first we need to tokenize our text, turning sequence of words into
# sequence of numbers
count = 1
my_dict ={}
Embedding_Dim = 100
NUM_WORDS = 50000
Epoch = [3,5]
Batch_size = [32]
max_length = 40  #length of longest sentence
seed = 7
np.random.seed(seed)

with open("File_name.txt","r")as f:
	texts = f.readlines()

tokenizer = Tokenizer(vocabulary_size)
#print tokenizer
tokenizer.fit_on_texts(texts) # Tokenizer is being fitted on the text to be processed
sequences = tokenizer.texts_to_sequences(texts) # here the conversion to tokens happens
#print sequences
word_index = tokenizer.word_index
invert = dict(map(reversed, word_index.items()))
#Data to be padded to have all sequence as having equal length
data = pad_sequences(sequences, maxlen=max_length,padding = 'post')
print('Loaded %s word vectors.' % len(embeddings_index))
#Reading the class value to convert it into one hot encoded.
classes = pd.read_csv("File_class.csv")
labels = classes['class'].values
labels_one_hot_encoded = np_utils.to_categorical(labels,3)
vocab_size=min(len(word_index)+1,(NUM_WORDS))
# create a weight matrix for words in training docs
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.twitter.27B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
embeddings_matrix = zeros((vocab_size, 100))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embeddings_matrix[i] = embedding_vector
print"The embedding matrix is", embeddings_matrix
print "The length of embedding matrix is", len(embeddings_matrix)
filenum = 1
#Stratifiedkfold the Data
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
z = np.zeros([3,3],dtype= int)
#z = np.zeros([2,2],dtype= int)
score_array = []
for bs in Batch_size:
	for epoch in Epoch:
		for train,test in kfold.split(data,labels):
			input_1 = Input(shape=(max_length,), dtype='int32')
			embedding_1 = Embedding(input_dim=8123, output_dim=100, weights=[embedding_matrix], input_length=40, mask_zero=False, 			
			lstm_1 = Bidirectional(LSTM(100, return_sequences = True,batch_input_shape=(None, 40, 100), input_shape=(40, 100)))(embedding_1)
			lstm_2 = Bidirectional(LSTM(100, return_sequences = False,batch_input_shape=(None, 40, 100), input_shape=(40, 100)))(lstm_1)
			dense_out = Dense(3, activation='softmax')(lstm_2)
			model = Model(inputs=input_1, outputs=dense_out)
			model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
			print"The model summary is", model.summary()
			filepath='BiLSTM'+'_'+str(epoch)+'_'+str(bs)+".hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			model.fit(data[train],labels_one_hot_encoded[train],validation_data = (data[test], labels_one_hot_encoded[test]),callbacks=callbacks_list,epochs = 		epoch,batch_size = bs,verbose = 2)
			model = load_model('BiLSTM'+'_'+str(epoch)+'_'+str(bs)+".hdf5")
			scores = model.evaluate(data[test], labels_one_hot_encoded[test],
				            batch_size=bs)

			cvscores.append(scores[1] * 100)
			new_prediction = model.predict(data[test])
			new_prediction = new_prediction.argmax(axis= -1)
			original_prediction =  np.argmax(labels_one_hot_encoded[test],axis = 1)
		    # restoring the sentence in word index form to its original sentence by using invert Dictionary
			#print data[test]
			sentence = []
			for line in data[test]:
				list = []
				for a in line:
					if a in invert:
						a = invert[a]
						list.append(a)
				a =  " ".join(list)
				sentence.append(a)
		
			test_data=np.asarray(data[test])
			#print test_data
			df=pd.DataFrame(columns=['tweets','real_label','new_label'])
			df['tweets']=sentence
			#df['tweets'] =
			#df = LabelEncoder.inverse_transform([argmax(test_data[:])])
			#print df
			df['real_label']=original_prediction
			df['new_label']=new_prediction
			df.to_csv('Bilstm'+str(filenum)+".csv",sep=',')
			filenum += 1
			cm1 = confusion_matrix(np.argmax(labels_one_hot_encoded[test],axis = 1),new_prediction)
			precision_recall = precision_recall_fscore_support(original_prediction,new_prediction)
			score_array.append(precision_recall_fscore_support(original_prediction,new_prediction, average=None))
			print cm1
			z = cm1+z
			print precision_recall
		print z
		z = np.zeros([3,3],dtype= int)
		avg_score = np.mean(score_array,axis=0)
		print avg_score
		print avg_score[0,0],avg_score[1,0],avg_score[2,0]
		print avg_score[0,1],avg_score[1,1],avg_score[2,1]
		print avg_score[0,2],avg_score[1,2],avg_score[2,2]
		avg_precision =  (avg_score[0,0]+avg_score[0,1]+avg_score[0,2])/3
		print ("precision:",avg_precision)
		avg_recall =  (avg_score[1,0]+avg_score[1,1]+avg_score[1,2])/3
		print ("recall:",avg_recall)
		avg_fscore =  (avg_score[2,0]+avg_score[2,1]+avg_score[2,2])/3
		print ("fscore:",avg_fscore)
		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

		 
	
