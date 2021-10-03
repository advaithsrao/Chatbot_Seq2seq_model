"""
test.py -> testing the chatbot 

Description: Implementation of a chatbot using Seq2seq model by Tensorflow. 
Dataset: Cornell movie dialog corpus with 200k+ conversations
"""

######################################################### Part 1 : Import packages & dataset ################################################
print("Part 1 : Importing")

# from absl import logging
# logging.set_verbosity(logging.ERROR)
import numpy as np
import tensorflow as tf
import re
import time

#user-defined functions from utils/
from utils.utils_preprocess import * #Custom utilities for preprocessing steps
from utils.utils_model import * #Custom utilities for the Seq2seq model
from chatbot import * #So that we could pull variables defined prior to training like -> questionswords2int and answerswords2int

########################################################### Part 2 : Setup Q&A #############################################################

#Loading weights and running session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting questions from string to list of encoded int
def convert_string2int(question, word2int):
	question = clean_text(question)
	return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

while(True):
	question = input("You:")
	if question == 'Goodbye':
		break
	question = convert_string2int(question, questionswords2int)
	question = question + [questionswords2int['<PAD>']] * (20 - len(question))
	fake_batch = np.zeros((batch_size, 20))
	fake_batch[0] = question
	predicted_answer = session.run(test_predictions, {input: fake_batch, keep_prob: 0.5})[0]
	answer = ''
	for i in np.argmax(predicted_answer, 1):
		if answerswords2int[i] == 'i':
			token = 'I'
		elif answerswords2int[i] == '<EOS>':
			token = '.'
		elif answerswords2int[i] == '<OUT>':
			token = 'out'
		else:
			token = ' ' + answerswords2int[i]
		answer += token
		if token == '.':
			break
	print('Chatbot: '+ answer)