"""
utils_preprocessing.py -> Consists of the preprocessing functions used in Part 2 of chatbot.py

Description: Implementation of a chatbot using Seq2seq model by Tensorflow. 
Dataset: Cornell movie dialog corpus with 200k+ conversations
"""

import re
import time

def create_id_to_line(lines):
	id2line = {}
	for line in lines:
		_line = line.split(' +++$+++ ')
		if len(_line)==5:
			id2line[_line[0]] = _line[4]
	return id2line

def get_convo_id(conversations):
	conversations_ids = []
	for conversation in conversations[:-1]:
		_conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
		conversations_ids.append(_conversation.split(","))
	return conversations_ids

def get_qa(id2line,conversations_ids):
	q = []
	a = []
	for conversation in conversations_ids:
		for i in range(len(conversation) - 1):
			q.append(id2line[conversation[i]])
			a.append(id2line[conversation[i+1]])
	return q,a

def clean_text(text):
	text = text.lower()
	text = re.sub(r"i'm", "i am", text)
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"what's", "what is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "can not", text)
	text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
	return text

def cleaner(text):
	cleaned_text = []
	for doc in text:
		cleaned_text.append(clean_text(doc))
	return cleaned_text

def create_word_to_count(clean_text,dict):
	for doc in clean_text:
		for word in doc.split():
			if word not in dict:
				dict[word] = 1
			else:
				dict[word] += 1
	return dict


def create_words_to_int(w2c):
	threshold = 20 #keeping only words whih occured 20 or more times
	words2int = {} 
	word_number = 0
	for word, count in w2c.items():
		if count >= threshold:
			words2int[word] = word_number
			word_number += 1
	return words2int

def add_tokens(w2i,tokens):
	for token in tokens:
		w2i[token] = len(w2i)+1
	return w2i

def create_text_to_int(text,w2i):
	text_to_int = []
	for doc in text:
		ints = []
		for word in doc.split():
			if word not in w2i:
				ints.append(w2i['<OUT>'])
			else:
				ints.append(w2i[word])
		text_to_int.append(ints)
	return text_to_int

def sort_text(q2i,a2i):
	q = []
	a = []
	for length in range(1, 25 + 1):
		for i in enumerate(q2i):
			if(len(i[1])) == length:
				q.append(q2i[i[0]])
				a.append(a2i[i[0]])
	return q,a