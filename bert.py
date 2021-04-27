import re
import nltk
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig
import tensorflow as tf



def max_length(data): #using this, we find out the maximum length of tokenizer is 171 for train, 167 for dev and 94 for test
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	max_len = 0
	for sent in data:
		input_ids = tokenizer.encode(sent, add_special_tokens=True)
		max_len = max(max_len, len(input_ids))
	#print('max length',max_len)
	return max_len

def tokenizer(data,labels):
	# Load the BERT tokenizer.
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []

	# For every sentence: Add '[CLS]' and '[SEP]', Pad & truncate all sentences, Construct attn. masks. and Return pytorch tensors.)
	for sent in data:
		encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = 171, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')  
		input_ids.append(encoded_dict['input_ids']) # Add the encoded sentence to the list.    
		attention_masks.append(encoded_dict['attention_mask']) # And its attention mask (simply differentiates padding from non-padding).
	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	# Print sentence 0, now as a list of IDs.
	# print('Original: ', data[0])
	# print('Token IDs:', input_ids[0])
	return input_ids, attention_masks, labels

def data_loader(data, labels, flag):
	# flag == 1 means the input dataset is training dataset
	batch_size = 32
	input_ids, attention_masks, labels = tokenizer(data, labels)
	dataset = TensorDataset(input_ids, attention_masks, labels)
	if flag == 1:# Select batches randomly 
		dataloader = DataLoader(dataset,  sampler = RandomSampler(dataset), batch_size = batch_size)
	else: # Pull out batches sequentially.
		dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = batch_size)
	print(dataloader)
	return dataloader

def train(epochs, train_dataloader):
	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')
	        # Reset the total loss for this epoch.
		total_train_loss = 0
		model.train()
	    # For each batch of training data...
		for step, batch in enumerate(train_dataloader):
	        # Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
	            # Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader)))
			# `batch` contains three pytorch tensors:
	        #   [0]: input ids 
	        #   [1]: attention masks
	        #   [2]: labels 
			# b_input_ids = batch[0]
			# b_input_mask = batch[1]
			# b_labels = batch[2]
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			model.zero_grad() 
			result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,return_dict=True)
			loss = result.loss
			logits = result.logits
			total_train_loss += loss.item()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()
		avg_train_loss = total_train_loss / len(train_dataloader)            
		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
	return model

def eval(model, val_dataloader):
	model.eval()
	total_eval_accuracy = 0
	total_eval_loss = 0
	nb_eval_steps = 0
	 # Evaluate data for one epoch
	for batch in validation_dataloader:
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)
		with torch.no_grad():        
			result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels,return_dict=True)
		loss = result.loss
		logits = result.logits
		total_eval_loss += loss.item()
	avg_val_loss = total_eval_loss / len(validation_dataloader)
	print('logits', logits)
	print('b_labels',b_labels)
	print('avg_val_loss', avg_val_loss)
	return 1
	




if __name__ == "__main__":
	# Get the GPU device name.
	device_name = tf.test.gpu_device_name()
	train_df = pd.read_csv('data/train.csv')
	train_data, train_labels = train_df["tweet"], train_df["class"]
	# load dev data
	dev_df = pd.read_csv('data/dev.csv')
	dev_data, dev_labels = dev_df["tweet"], dev_df["class"]
    # load test data
	test_df = pd.read_csv('data/test.csv')
	test_data, test_labels = test_df["tweet"], test_df["class"]
	#create dataloader
	train_dataloader = data_loader(train_data, train_labels,1)

	#model
	# Load BertForSequenceClassification, the pretrained BERT model with a single 
	# linear classification layer on top. 
	# Use the 12-layer BERT model, with an uncased vocab.
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False)
	optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)
	epochs = 1
	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * epochs
	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,  num_training_steps = total_steps)
	train(epochs, train_dataloader)