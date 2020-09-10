import Public as pb
# import math
# import numpy as np
import os
from CharTrainer import Char2GaussianTrainer
import time
# import random
# import pandas as pd
# from gensim.models.word2vec import Word2Vec
# import random



def Init():
	file = open(pb.corpus_path, "r")

	while True:
		string = file.readline()
		if not string:
			break

		string = string[:-1]

		# if(len(string)>0):
		# 	print(string)

		for ch in string:
			if(ch not in pb.c2i.keys()):
				pb.c2i[ch] = len(pb.i2c)
				pb.i2c.append(ch)

	print(pb.c2i)
	print(pb.i2c)

	file.close()

# [pb.c2i, pb.i2c, mus, lvs] = pb.Pickle_Read(pb.save_path)
# print(pb.c2i)
# print(len(pb.i2c))
# print(len(mus), len(mus[0]))
# print(mus[pb.c2i["忍"]])
# print(lvs[pb.c2i["忍"]])

for pb.language in ["Japanese"]:
	pb.corpus_path = "./" + pb.language + "_Corpora.txt"
	pb.save_path = "./" + pb.language

	if (os.path.exists(pb.save_path)==False):
		pb.c2i, pb.i2c = {}, []
		Init()
		model = Char2GaussianTrainer()
		print(pb.language)
		model.train()
		model.save()
		print()
