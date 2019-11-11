import nltk
from nltk import pos_tag
from nltk import word_tokenize
import re
count = 1
with open("File_name.txt","r") as f,open("POS_Tag.txt","w") as g:
	for line in f:
		POS = []
		line = line.lower()
		#line = line.replace("."," ")
		text = nltk.word_tokenize(line)
		sentence = nltk.pos_tag(text,tagset='universal')
		for i in sentence:
		    POS.append(i[1])
		sentence =  " ".join(POS)
		print sentence
		g.write(sentence+"\n")
