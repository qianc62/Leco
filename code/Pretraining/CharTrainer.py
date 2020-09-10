# bert-serving-start -model_dir /Users/qianchen/Workspace/PythonWorkspace/Bert-Server-Client/uncased_L-12_H-768_A-12
import Public as pb
import tensorflow as tf
import math
import os
import random
import numpy as np
import time
# from bert_serving.client import BertClient
import pickle



class Char2GaussianTrainer:
	def __init__(self):

		mus, lvs = [], []

		mu_scale = math.sqrt(3.0 / (1.0 * pb.embedding_size))
		for i in range(len(pb.i2c)):
			mus.append( np.random.uniform(-mu_scale, mu_scale, pb.embedding_size) )
			lvs.append( np.full(pb.embedding_size, 0.05) )

		print("mus size: {}[{}]".format(len(mus), len(mus[0])))
		print("logvars size: {}[{}]".format(len(lvs), len(lvs[0])))

		self.mus = tf.Variable( tf.cast(np.array(mus), tf.float32) )
		self.lvs = tf.Variable( tf.cast(np.array(lvs), tf.float32) )

		self.placeholder_cen_word_idx = tf.placeholder(dtype=tf.int32)
		self.placeholder_pos_word_idx = tf.placeholder(dtype=tf.int32)
		self.placeholder_neg_word_idx = tf.placeholder(dtype=tf.int32)

		self.session = tf.Session()

		self.zeros_vec = tf.zeros([1])

		self.loss = self.caculate_loss(self.placeholder_cen_word_idx, self.placeholder_pos_word_idx, self.placeholder_neg_word_idx)

		print("learning_rate: {}".format(pb.learning_rate))
		self.trainer = tf.train.AdagradOptimizer(pb.learning_rate).minimize(self.loss)

		self.session.run(tf.global_variables_initializer())

	def save(self):
		mus = self.session.run(self.mus)
		lvs = self.session.run(self.lvs)
		pb.Pickle_Save([pb.c2i, pb.i2c, mus, lvs], pb.save_path)
		# print(pb.i2c[0], mus[0], lvs[0])
		# print(pb.i2c[1], mus[1], lvs[1])
		# print(pb.i2c[2], mus[2], lvs[2])
		# print(pb.i2c[3], mus[3], lvs[3])

		writer = open(pb.save_path+"_", "w")
		for key in pb.c2i:
			writer.write("{} ".format(key))
			for v in mus[pb.c2i[key]]:
				writer.write("{} ".format(v))
			for v in lvs[pb.c2i[key]]:
				writer.write("{} ".format(v))
			writer.write("\n")
		writer.close()

	def caculate_loss(self, cen_char_idx, pos_char_idx, neg_char_idx):
		def similarity_energy(mu1, logvar1, mu2, logvar2):
			var1 = tf.exp(logvar1)
			var2 = tf.exp(logvar2)
			var_add = tf.add(var1, var2)
			logdet = tf.reduce_sum(tf.log(var_add))
			constant = tf.log(2 * pb.PI)
			diff = tf.subtract(mu1, mu2)
			ss_inv = 1.0 / (var_add)
			exp_term = tf.reduce_sum(diff * ss_inv * diff)
			enegy = -0.5 * logdet - pb.embedding_size * 1.0 / 2 * constant - 0.5 * exp_term
			return enegy

		mu_cen_char_emb = tf.nn.embedding_lookup(self.mus, cen_char_idx)
		mu_pos_char_emb = tf.nn.embedding_lookup(self.mus, pos_char_idx)
		mu_neg_char_emb = tf.nn.embedding_lookup(self.mus, neg_char_idx)
		lv_cen_char_emb = tf.nn.embedding_lookup(self.lvs, cen_char_idx)
		lv_pos_char_emb = tf.nn.embedding_lookup(self.lvs, pos_char_idx)
		lv_neg_char_emb = tf.nn.embedding_lookup(self.lvs, neg_char_idx)

		se_pos = similarity_energy(mu_cen_char_emb, lv_cen_char_emb, mu_pos_char_emb, lv_pos_char_emb)
		se_neg = similarity_energy(mu_cen_char_emb, lv_cen_char_emb, mu_neg_char_emb, lv_neg_char_emb)
		loss_indiv = tf.maximum(self.zeros_vec, pb.objective_threshold - se_pos + se_neg)
		self.sloss = tf.reduce_mean(loss_indiv)

		return self.sloss

	def word_negative_sampling(self):
		word = random.choice(pb.i2c)
		return word

	def train(self):
		counter = 0

		file = open(pb.corpus_path, "r")

		while True:
			line = file.readline()
			if not line:
				break

			line = line[:len(line) - 1]

			for i in range(len(line)):
				cen_char = line[i]

				for w in range(-pb.window_size, pb.window_size + 1):
					c = i + w
					if (c < 0 or c >= len(line) or c == i):
						continue

					pos_char = line[c]
					neg_char = self.word_negative_sampling()

					cen_char_idx = pb.c2i[cen_char]
					pos_char_idx = pb.c2i[pos_char]
					neg_char_idx = pb.c2i[neg_char]

					self.session.run([self.trainer], feed_dict={ self.placeholder_cen_word_idx: cen_char_idx,
																 self.placeholder_pos_word_idx: pos_char_idx,
																 self.placeholder_neg_word_idx: neg_char_idx})

			counter += 1
			if(counter%100==0):
				print(counter, line)
