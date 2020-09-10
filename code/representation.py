# bert-serving-start -model_dir /Users/qianchen/Documents/3科研/1研究工作/10【SIGIR-2020】【在投】单文本分类/3实验阶段/LogoNet/BERT/multi_cased_L-12_H-768_A-12
#
# Adress already exists:
# 	-port 5678 -port_out 5679 (port=5679)
# bert-serving-start -model_dir ./ -port 5678 -port_out 5679

import sys
sys.path.append("/usr/local/bin")


import _public as pb
from scipy import optimize
import math
import numpy as np
# import plot as fig
from gensim.models.word2vec import Word2Vec
import dataset
# import torch
import os
from bert_serving.client import BertClient
from scipy.stats import chisquare
# from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model
# from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel
# from allennlp.commands.elmo import ElmoEmbedder
import random
from scipy import stats
# import matplotlib.pyplot as plt
# from scipy import stats
# import dataset as ds
# import random as random
# import threading
# import time
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from pytorch_pretrained_bert import BertModel, BertTokenizer
import tensorflow as tf
from BERT import modeling



class Seed:
	def __init__(self):
		self.ngrams = ""
		self.chi = 0.0
		self.mu, self. lv = [], []

class Representator:
	def __init__(self, examples):
		self.default_word = "boy"

		self.w2v = None
		self.w2s = None

		self.elmo = None
		self.gpt  = None
		self.gpt2 = None
		self.bert = None

		self.bert_tokenizer = None

		labels = [example.label for example in examples]

		self.En_seeds = self.Get_Seeds(labels, [example.En for example in examples], "./data/"+pb.source+"_En_seeds", "English")
		self.Ge_seeds = self.Get_Seeds(labels, [example.Ge for example in examples], "./data/"+pb.source+"_Ge_seeds", "Germany")
		self.Th_seeds = self.Get_Seeds(labels, [example.Th for example in examples], "./data/"+pb.source+"_Th_seeds", "Thai")
		self.Ar_seeds = self.Get_Seeds(labels, [example.Ar for example in examples], "./data/"+pb.source+"_Ar_seeds", "Arabic")
		self.Ja_seeds = self.Get_Seeds(labels, [example.Ja for example in examples], "./data/"+pb.source+"_Ja_seeds", "Japanese")
		self.Ch_seeds = self.Get_Seeds(labels, [example.Ch for example in examples], "./data/"+pb.source+"_Ch_seeds", "Chinese")

		print("En Seeds: {}".format(len(self.En_seeds)))
		print("Ge Seeds: {}".format(len(self.Ge_seeds)))
		print("Th Seeds: {}".format(len(self.Th_seeds)))
		print("Ar Seeds: {}".format(len(self.Ar_seeds)))
		print("Ja Seeds: {}".format(len(self.Ja_seeds)))
		print("Ch Seeds: {}".format(len(self.Ch_seeds)))
		# pass

	def Chi_Square_Test(self, labels, texts, seed_candidate):

		u = np.zeros( len(pb.LABELS) )
		v = np.zeros( len(pb.LABELS) )

		for i in range(len(texts)):
			if (seed_candidate in texts[i]):
				u[pb.LABELS.index(labels[i])] += 1
			else:
				v[pb.LABELS.index(labels[i])] += 1

		sum_u = np.sum(u)
		sum_v = np.sum(v)
		ratio_u = sum_u * 1.0 / (sum_u + sum_v)
		ratio_v = sum_v * 1.0 / (sum_u + sum_v)

		chi = 0.0
		for i in range(len(pb.LABELS)):
			e_u = (u[i] + v[i]) * ratio_u
			e_v = (u[i] + v[i]) * ratio_v
			chi += (u[i] - e_u) ** 2 / (e_u + 0.00000001)
			chi += (v[i] - e_v) ** 2 / (e_v + 0.00000001)

		return chi

	def Raw_Seed_Generation(self, labels, texts):
		if(len(texts)==0):
			return []

		seed_candidates = set()
		for sentence in texts:
			for begin_index in range(len(sentence)):
				for l in range(pb.seed_minlength, pb.seed_maxlength + 1):
					if ( l>0 and begin_index+l-1<=len(sentence)-1):
						gram = sentence[begin_index:begin_index + l]
						seed_candidates.add(gram)
		print("{}\tseed_candidates:{}".format(pb.NAME, len(seed_candidates)))

		print("Seeding")
		seeds = []
		chi_map = {}
		for i,seed_candidate in enumerate(seed_candidates):
			chi = self.Chi_Square_Test(labels, texts, seed_candidate)
			chi_map[seed_candidate] = chi
			print("{}\tSeeding {:.2%}".format(pb.NAME, (i+1)*1.0/(len(seed_candidates)+1) ))

		chi_sorted_x, chi_sorted_y = pb.Map_To_Sorted_List(chi_map)

		print("Filtering")
		flag = [True for _ in chi_sorted_x]
		for i in range(len(chi_sorted_x)):
			if(flag[i]==False):
				continue
			for j in range(i + 1, len(chi_sorted_x)):
				if (flag[j]==True and chi_sorted_x[i] in chi_sorted_x[j]):
					flag[j] = False
			print("{}\tSeed Filtering {:.2%}".format(pb.NAME, i*1.0/len(chi_sorted_x)))

		for i in range(len(chi_sorted_x)):
			if (flag[i] == True and chi_sorted_y[i]>0.00):
				seed = Seed()
				seed.word = chi_sorted_x[i]
				seed.chi = chi_map[seed.word]
				seeds.append(seed)

		return seeds

	def Get_Seeds(self, labels, texts, filepath, name):

		if (os.path.exists(filepath) == False):
			pb.NAME = name
			seeds = self.Raw_Seed_Generation(labels, texts)
			pb.Pickle_Save(seeds, filepath)

		seeds = pb.Pickle_Read(filepath)

		if pb.seed_upperbound > 0:
			seeds = [seed for seed in seeds if len(seed.word)<=pb.seed_upperbound]

		sum, subsum, seperator = np.sum([seed.chi for seed in seeds]), 0.0, 0
		for i,seed in enumerate(seeds):
			subsum += seed.chi

			if(subsum/sum>=pb.hard_filtering):
				seperator = i
				break

			sv1 = seed.chi / sum
			sv2 = 1.0 / math.exp(sv1)
			# print(sv1, sv2)
			for char in seed.word:
				try:
					if(name=="English"):
						seed.mu.append(pb.mus_En[pb.c2i_En[char]])
						seed.lv.append(pb.lvs_En[pb.c2i_En[char]] * sv2)
					elif(name=="Germany"):
						seed.mu.append(pb.mus_Ge[pb.c2i_Ge[char]])
						seed.lv.append(pb.lvs_Ge[pb.c2i_Ge[char]] * sv2)
					elif (name == "Thai"):
						seed.mu.append(pb.mus_Th[pb.c2i_Th[char]])
						seed.lv.append(pb.lvs_Th[pb.c2i_Th[char]] * sv2)
					elif (name == "Arabic"):
						seed.mu.append(pb.mus_Ar[pb.c2i_Ar[char]])
						seed.lv.append(pb.lvs_Ar[pb.c2i_Ar[char]] * sv2)
					elif (name == "Japanese"):
						seed.mu.append(pb.mus_Ja[pb.c2i_Ja[char]])
						seed.lv.append(pb.lvs_Ja[pb.c2i_Ja[char]] * sv2)
					elif (pb.target == "Chinese"):
						seed.mu.append(pb.mus_Ch[pb.c2i_Ch[char]])
						seed.lv.append(pb.lvs_Ch[pb.c2i_Ch[char]] * sv2)
				except:
					seed.mu.append( np.zeros(20) )
					seed.lv.append( np.full(20, math.log(0.05)) )
					print("[0.0]*20")

		seeds = seeds[:seperator]

		return seeds

	def Gaussian_Convolution(self, mu1, logvar1, mu2, logvar2):

		var1 = np.exp(logvar1)
		var2 = np.exp(logvar2)
		var_add = np.add(var1, var2)

		diff = mu1 - mu2
		ss_inv = 1.0 / (var_add)
		exp_term = np.sum(diff * ss_inv * diff)

		return -0.5*exp_term

	def Get_nGram_Representations(self, examples):
		for I, example in enumerate(examples):
			# example.En_gram = [1.0 if seed.word in example.En else 0.0 for seed in self.En_seeds]
			# example.Ge_gram = [1.0 if seed.word in example.Ge else 0.0 for seed in self.Ge_seeds]
			# example.Th_gram = [1.0 if seed.word in example.Th else 0.0 for seed in self.Th_seeds]
			# example.Ar_gram = [1.0 if seed.word in example.Ar else 0.0 for seed in self.Ar_seeds]
			# example.Ja_gram = [1.0 if seed.word in example.Ja else 0.0 for seed in self.Ja_seeds]
			# example.Ch_gram = [1.0 if seed.word in example.Ch else 0.0 for seed in self.Ch_seeds]

			# print(len(example.En_gram))

			if os.path.exists('./'+pb.source)==False:
				if (len(example.En_gram) == 0): example.En_gram = [1.0 if seed.word in example.En else 0.0 for seed in self.En_seeds]
				if (len(example.Ge_gram) == 0): example.Ge_gram = [1.0 if seed.word in example.Ge else 0.0 for seed in self.Ge_seeds]
				if (len(example.Th_gram) == 0): example.Th_gram = [1.0 if seed.word in example.Th else 0.0 for seed in self.Th_seeds]
				if (len(example.Ar_gram) == 0): example.Ar_gram = [1.0 if seed.word in example.Ar else 0.0 for seed in self.Ar_seeds]
				if (len(example.Ja_gram) == 0): example.Ja_gram = [1.0 if seed.word in example.Ja else 0.0 for seed in self.Ja_seeds]
				if (len(example.Ch_gram) == 0): example.Ch_gram = [1.0 if seed.word in example.Ch else 0.0 for seed in self.Ch_seeds]

			else:
				text, rep = example.Ja, []
				best_piece = ""
				for seed in self.Ja_seeds:
					if(seed.word in text):
						rep.append( 1.0 )
					else:
						rep.append(0.0)
						continue

						att = seed.word
						while True:
							if(len(att)==0):
								break
							index = text.find(att)
							if(index!=-1):
								end = index + len(seed.word) - 1
								if(end < len(text)):
									best_piece = text[index:end+1]
								else:
									best_piece = ""
								break
							else:
								att = att[:-1]

						o = 0.0
						if (best_piece != ""):
							for char1 in best_piece:
								for k in range(len(seed.word)):
									o += self.Gaussian_Convolution(pb.mus_Ja[pb.c2i_Ja[char1]], pb.lvs_Ja[pb.c2i_Ja[char1]], seed.mu[k], seed.lv[k])
							# print(o)
						rep.append(o)

				example.Ja_gram = rep

				print("gram_Representating {:.2%}".format((I + 1) * 1.0 / (len(examples) + 1)))

	def Get_Representations(self, examples):
		self.Get_Bert_Representation(examples)
		# self.Get_Bert_Original_Representation(examples)
		# self.Get_Elmo_Representation(examples)
		# self.Get_GPT_Representation(examples)
		# self.Get_GPT2_Representation(examples)
		# self.Get_Word2Vec_Representation(examples)
		# self.Get_Word2Sense_Representation(examples)
		return

	# def Get_Bert_Original_Representation(self, examples):


	def Get_Bert_Representation(self, examples):
		if(self.bert==None):
			print("Bert Initializing")
			self.bert = BertClient()
			print("Done Bert Initializing")

		# for i, example in enumerate(examples):
		# 	if (len(example.En_bert)!=768):
		# 		example.En_bert = self.bert.encode([example.En])[0]
		# 	if (len(example.Ge_bert)!=768):
		# 		example.Ge_bert = self.bert.encode([example.Ge])[0]
		# 	if (len(example.Th_bert)!=768):
		# 		example.Th_bert = self.bert.encode([example.Th])[0]
		# 	if (len(example.Ar_bert)!=768):
		# 		example.Ar_bert = self.bert.encode([example.Ar])[0]
		# 	if (len(example.Ja_bert)!=768):
		# 		example.Ja_bert = self.bert.encode([example.Ja])[0]
		# 	if (len(example.Ch_bert)!=768):
		# 		example.Ch_bert = self.bert.encode([example.Ch])[0]

		for i, example in enumerate(examples):
			if (len(example.En_bert) != 768 or len(example.Ge_bert) != 768 or len(example.Th_bert) != 768 or len(example.Ar_bert) != 768 or len(example.Ja_bert) != 768 or len(example.Ch_bert) != 768):
				[example.En_bert, example.Ge_bert, example.Th_bert, example.Ar_bert, example.Ja_bert, example.Ch_bert] = self.bert.encode([example.En, example.Ge, example.Th, example.Ar, example.Ja, example.Ch])

			# print("Bert_Representating {:.2%}".format((i + 1) * 1.0 / (len(examples) + 1)))

	# Baselines
	# def Get_Word2Vec_Representation(self, examples):
	# 	if (self.w2v == None):
	# 		self.w2v_embdding_size = 100
	# 		self.w2v = Word2Vec.load("./w2v/w2v_model")
	# 		self.vocabulary = set(open("./w2v/text8_vocabulary.txt").read().split("\n"))
	#
	# 	for i, example in enumerate(examples):
	# 		representation = np.zeros(self.w2v_embdding_size)
	# 		counter = 0
	# 		for word in example.En.split(" "):
	# 			if(word in self.vocabulary):
	# 				representation += self.w2v[word]
	# 				counter += 1
	# 			else:
	# 				representation += self.w2v[self.default_word]
	# 				counter += 1
	# 		example.word2vec_mat = representation / counter
	#
	# def Get_Word2Sense_Representation(self, examples):
	# 	if (self.w2s == None):
	# 		self.w2s = pb.Pickle_Read("./Word2Sense_2250")
	#
	# 	for i, example in enumerate(examples):
	# 		representation = np.zeros(2250)
	# 		counter = 0
	# 		for word in example.En.split(" "):
	# 			if(word in self.w2s.keys()):
	# 				representation += self.w2s[word]
	# 				counter += 1
	# 			else:
	# 				representation += self.w2s[self.default_word]
	# 				counter += 1
	# 		example.En_baseline = representation / counter
	#
	# def Get_Elmo_Representation(self, examples):
	# 	if(self.elmo==None):
	# 		options_file = "./sources/elmo_2x1024_128_2048cnn_1xhighway_options.json"
	# 		weight_file = "./sources/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
	# 		self.elmo = ElmoEmbedder(options_file, weight_file)
	#
	# 	for i,example in enumerate(examples):
	# 		text = example.En
	#
	# 		context_tokens = [text.split(" ")]
	# 		elmo_embedding, _ = self.elmo.batch_to_embeddings(context_tokens)
	# 		# print(np.array(elmo_embedding).shape)
	#
	# 		example.En_baseline = np.average(elmo_embedding[0][-1], axis=0)
	#
	# 		print("{:.2%}".format(i*1.0/len(examples)))
	#
	# def Get_GPT_Representation(self, examples):
	# 	for i, example in enumerate(examples):
	#
	# 		if (len(example.En_baseline) == 768):
	# 			continue
	#
	# 		if (self.gpt == None):
	# 			self.gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
	# 			self.gpt = OpenAIGPTModel.from_pretrained('openai-gpt')
	# 			self.gpt.eval()
	#
	# 		try:
	# 			indexed_tokens = self.gpt_tokenizer.encode(example.En)
	# 			tokens_tensor = torch.tensor([indexed_tokens])
	#
	# 			with torch.no_grad():
	# 				gpt_embedding, _ = self.gpt(tokens_tensor)
	#
	# 			example.En_baseline = np.average(gpt_embedding[0], axis=0)
	#
	# 		except:
	# 			example.En_baseline = np.zeros(768)
	#
	# 		print(i, "{:.2%}".format(i * 1.0 / len(examples)))
	#
	# def Get_GPT2_Representation(self, examples):
	# 	for i, example in enumerate(examples):
	#
	# 		if (len(example.En_baseline) == 768):
	# 			continue
	#
	# 		if (self.gpt2 == None):
	# 			self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	# 			self.gpt2 = GPT2Model.from_pretrained('gpt2')
	# 			self.gpt2.eval()
	#
	# 		try:
	# 			indexed_tokens = self.gpt2_tokenizer.encode(example.En)
	# 			tokens_tensor = torch.tensor([indexed_tokens])
	#
	# 			with torch.no_grad():
	# 				gpt_embedding, _ = self.gpt2(tokens_tensor)
	#
	# 			example.En_baseline = np.average(gpt_embedding[0], axis=0)
	#
	# 		except:
	# 			example.En_baseline = np.zeros(768)
	#
	# 		print(i, "{:.2%}".format(i * 1.0 / len(examples)))
