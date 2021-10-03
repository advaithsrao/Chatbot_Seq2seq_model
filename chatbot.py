"""
chatbot.py -> Setup preprocessing, training and saving checkpoints for the chatbot

Description: Implementation of a chatbot using Seq2seq model by Tensorflow. 
Dataset: Cornell movie dialog corpus with 200k+ conversations
"""
######################################################### Part 1 : Import packages & dataset ################################################
print("Part 1 : Importing")

# import warnings
# warnings.filterwarnings("ignore")
# from absl import logging
# logging.set_verbosity(logging.ERROR)
import numpy as np
import tensorflow as tf
import re
import time

#user-defined functions from utils/
from utils.utils_preprocess import * #Custom utilities for preprocessing steps
from utils.utils_model import * #Custom utilities for the Seq2seq model

#import dataset
lines = open('data/lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('data/conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

########################################################### Part 2 : Preprocessing ###########################################################
print("Part 2 : Preprocessing")
# # Creating dictionary mapping line <=> id
# id2line = create_id_to_line(lines)

# # Creating list of all conversations
# conversations_ids = get_convo_id(conversations)

# # Getting separately the questions and the answers
# questions,answers = get_qa(id2line,conversations)

#Define a dictionary to map id to line
id2line={}
for line in lines:
    _line = line.split(' +++$+++ ') # Splitting the line to take the elements we want
    if len(_line) == 5:             # To maintain all the lines of the same no. of elements
        id2line[_line[0]] = _line[4] # Creating a dictionary mapping between the id and the line

# List of conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "") # Splitting the conversations, and formatting them to remove [];'';spaces
    conversations_ids.append(_conversation.split(',')) # Formating the large list to consist only the id's in a list

# Making the data into q&a or into two speakers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# Cleaing the questions & answers
clean_questions = cleaner(questions)
clean_answers = cleaner(answers)

# Creating a dictionary mapping each words <=> no. of occurences
word2count = {}
word2count = create_word_to_count(clean_questions,word2count)
word2count = create_word_to_count(clean_answers,word2count)

# Creating two dict mapping questions words n answer words to unique int
questionswords2int = create_words_to_int(word2count)
answerswords2int = create_words_to_int(word2count)

# Adding the last toknes to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
questionswords2int = add_tokens(questionswords2int,tokens)
answerswords2int = add_tokens(answerswords2int,tokens)

# Creating inv dictionary of answerswords2int dict
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
#Adding the EOS to all answers
for i in range(len(clean_answers)):
	clean_answers[i] += ' <EOS>'

#Translating all questions and answers into integers
# and replacing all words by <OUT>
questions_to_int = create_text_to_int(clean_questions,questionswords2int)
answers_to_int = create_text_to_int(clean_answers,answerswords2int)

# Sorting q and a by the length of questions
sorted_clean_answers,sorted_clean_questions = sort_text(questions_to_int,answers_to_int)

########################################################### Part 3 : Training the seq2se1 model #############################################
print("Part 3 : Training")

#Setting the hyperparam
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading the model inputs
inputs, targets, lr, keep_prob =  model_inputs()

#Setting the sequence_length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

#Getting the shape of inputs tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions 
training_predictions, test_predictions = seq2seq_model(
	tf.reverse(inputs, [-1]),
	targets,
	keep_prob,
	batch_size,
	sequence_length,
	len(answerswords2int),
	len(questionswords2int),
	encoding_embedding_size,
	decoding_embedding_size,
	rnn_size,
	num_layers,
	questionswords2int)

# Setting up the loss error, the optimizer, gradient clipping
with tf.name_scope("optimization"):
	loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
		targets,
		tf.ones([input_shape[0], sequence_length]))
	optimizer = tf.train.AdamOptimizer(learning_rate)
	gradients = optimizer.compute_gradients(loss_error)
	clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
	optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

#Splitting the q and a into training/validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

#Train
batch_index_check_training_loss  = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) -1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000 #100 maybe
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs+1):
	for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size, questionswords2int, answerswords2int)):
		starting_time = time.time()
		_, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
			targets: padded_answers_in_batch,
			lr: learning_rate,
			sequence_length: padded_answers_in_batch.shape[1],
			keep_prob: keep_probability})
		total_training_loss_error += batch_training_loss_error
		ending_time = time.time()
		batch_time = ending_time - starting_time
		if batch_index % batch_index_check_training_loss == 0:
			print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss error: {:>6.3f}, Training Time on 100 batches: {:d} seconds'. format(epoch,
				epochs,
				batch_index,
				len(training_questions) // batch_size,
				total_training_loss_error / batch_index_check_training_loss,
				int(batch_time * batch_index_check_training_loss)))
			total_training_loss_error = 0
		if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
			total_validation_loss_error = 0
			starting_time = time.time()
			for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
				batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
					targets: padded_answers_in_batch,
					lr: learning_rate,
					sequence_length: padded_answers_in_batch.shape[1],
					keep_prob: 1})
				total_validation_loss_error += batch_validation_loss_error
			ending_time = time.time()
			batch_time = ending_time - starting_time
			average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
			print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds' .format(average_validation_loss_error, int(batch_time)))
			learning_rate  *= learning_rate_decay
			if learning_rate < min_learning_rate:
				learning_rate = min_learning_rate
			list_validation_loss_error.append(average_validation_loss_error)
			if average_validation_loss_error <= min(list_validation_loss_error):
				print('Improvement over epoch!')
				early_stopping_check = 0
				saver = tf.train.Saver()
				saver.save(session, checkpoint)
			else:
				print("Needs work!")
				early_stopping_check += 1
				if early_stopping_check == early_stopping_stop:
					break
	if early_stopping_check == early_stopping_stop:
		print("Failed to learn")
		break
print("Training over")

