from numpy import array
from numpy import asarray
from numpy import zeros
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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
word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary = True)

# first we need to tokenize our text, turning sequence of words into
# sequence of numbers
EMB_DIM = 20
text_list = []
count = 1
my_dict ={}
POS_TAG_SIZE = 14
Epoch = [3,5]
Batch_size = [16,32] 
max_length = 40  #length of longest sentence
seed = 7
Embedding_Dim = 100
NUM_WORDS = 50000

# the file "g" is for reading the labels where as "f" have texts(tweets)
with open("File_name.txt","r")as f:
	texts = f.readlines()
tokenizer = Tokenizer(NUM_WORDS)
tokenizer.fit_on_texts(texts) # we fit tokenizer on texts we will process
sequences = tokenizer.texts_to_sequences(texts) # here the conversion to tokens happens
word_index = tokenizer.word_index
invert = dict(map(reversed, word_index.items()))
#Text data for training with word embeddings features
data = pad_sequences(sequences, maxlen=max_length,padding = 'post')
with open("POS_Tag.txt","r") as k, open("POS_Tag.txt","r") as h:
	for line in k:
		line = line.lower()
		line = line.split()
		text_list.append(line)
		count +=1
	texts_1 = h.readlines()

tokenizer_POS = Tokenizer(POS_TAG_SIZE)
tokenizer_POS.fit_on_texts(texts_1) # we fit tokenizer on texts we will process
word_index_POS = tokenizer_POS.word_index
sequences_pos = tokenizer_POS.texts_to_sequences(texts_1) # here the conversion to tokens happens
data_pos = pad_sequences(sequences_pos,maxlen = max_length,padding = 'post')
#Generate the embedding of POS Tags.	
w2v = Word2Vec(text_list,size = EMB_DIM,window=5,iter=15,negative=15)
for idx,key in enumerate(w2v.wv.vocab):
	my_dict[key] = w2v.wv[key]
POS_TAG_Embedding_matrix = zeros((POS_TAG_SIZE,EMB_DIM))
for word,i in word_index_POS.items():
	POS_embedding = my_dict.get(word)
	if POS_embedding is not None:
		POS_TAG_Embedding_matrix[i] = POS_embedding

#Reading the class value to convert it into one hot encoded.
classes = pd.read_csv("File_class.csv")  ##This contains the class of each instance in the same order of input text sentences.
labels = classes['class'].values
labels_one_hot_encoded = np_utils.to_categorical(labels,2)
vocab_size=min(len(word_index)+1,(NUM_WORDS))
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
z = np.zeros([2,2],dtype= int)
score_array = []
for epoch in Epoch:
	for bs in Batch_size:
		for train,test in kfold.split(data,labels):
			input_1 = Input(shape=(max_length,), dtype='int32')
			POS_input = Input(shape = (max_length,),dtype = 'int32')
			tweet_encoder = Embedding(input_dim=vocab_size, output_dim=100, weights=[embeddings_matrix], input_length=max_length, mask_zero=False, 				trainable=True)(input_1)
			POS_embedding = Embedding(input_dim = POS_TAG_SIZE,output_dim = EMB_DIM,weights = [POS_TAG_Embedding_matrix],input_length = max_length,mask_zero = False,trainable = True)(POS_input)
			bilstm_1 =Bidirectional (LSTM(100, return_sequences = True,batch_input_shape=(None, 40, 100), input_shape=(40, 100)))(tweet_encoder)
			bilstm_2 = Bidirectional(LSTM(100, return_sequences = False,batch_input_shape=(None, 40, 100), input_shape=(40, 100)))(bilstm_1)
			POS_embedding = Embedding(input_dim = POS_TAG_SIZE,output_dim = EMB_DIM,weights = [POS_TAG_Embedding_matrix],input_length = max_length,trainable =True)(POS_input)
			POS_BILSTM = Bidirectional(LSTM(100, return_sequences = False,batch_input_shape=(None, 40, 100), input_shape=(40, 100)))(POS_embedding)
			merged_1 = concatenate([bilstm_2,POS_BILSTM])
			dense_out = Dense(2,activation = 'softmax')(merged_1)
			model = Model(inputs = [input_1,POS_input],outputs = dense_out)
			#plot_model(model,to_file= 'BiLSTM_word+POS.png',show_shapes = True,show_layer_names = True)
			model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
			print "The model summary is",model.summary()	
			filepath="Bilstm_word_POS.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			model.fit([data[train],data_pos[train]],labels_one_hot_encoded[train],validation_data = ([data[test],data_pos[test]], labels_one_hot_encoded[test]),callbacks=callbacks_list,epochs = 		epoch,batch_size = bs,verbose = 2)
			model = load_model("Bilstm_word_POS.hdf5")
			scores = model.evaluate([data[test],data_pos[test]], labels_one_hot_encoded[test],
                            batch_size=bs)
			#The prediction of testing file
			new_prediction = model.predict([data[test],data_pos[test]])
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
			#Save each fold into csv fold with their original and predicted labels.
			sentence = []
			df=pd.DataFrame(columns=['tweets','real_label','new_label'])
			df['tweets']=sentence
			df['real_label']=original_prediction
			df['new_label']=new_prediction
			df.to_csv('BiLSTM_POS'+str(filenum)+".csv",sep=',')
			filenum += 1
			cm1 = confusion_matrix(np.argmax(labels_one_hot_encoded[test],axis = 1),new_prediction)
			precision_recall = precision_recall_fscore_support(original_prediction,new_prediction)
			score_array.append(precision_recall_fscore_support(original_prediction,new_prediction, average=None))
			print "The confusion matrix",cm1
			z = cm1+z
			print precision_recall
		print"The Final confusion matrix",z
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
