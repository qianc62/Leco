# from OpenHowNet import HowNet
# import math
import pickle
# import random
import pandas as pd
import numpy as np
import threading
from scipy import stats
import time
from sklearn import metrics
import os



INF = 9999999999
PI = 3.1415926
Epsilon = 1e-5

language = "English"
corpus_path = "./"+language+"_Corpora.txt"
save_path = "./"+language
c2i, i2c = {}, []

embedding_size = 20
learning_rate = 0.01
window_size = 5
objective_threshold = 1.0


def Pickle_Save(variable, path):
	with open(path, 'wb') as file:
		pickle.dump(variable, file)
	print("Pickle Saved {}".format(path))

def Pickle_Read(filepath):
	with open(filepath, 'rb') as file:
		obj = pickle.load(file)
	print("Pickle Read")
	return obj

def Weighted_Sampling(weight, num):
	mat = []
	for i in range(len(weight)):
		mat.append([i, weight[i]])
	df = pd.DataFrame(np.array(mat))
	ans = df.sample(n=num, replace=True, weights=weight)

	idxs = []
	for idx in ans[0]:
		idxs.append(int(idx))
	return idxs

def Get_Spearman_Correlation(array1, array2):
	(correlation, pvalue) = stats.spearmanr(array1, array2)
	return correlation

def Get_F1_Score(array1, array2):
	f1 = metrics.f1_score(array1, array2, average="micro")
	return f1

def Print_Line(string=""):
	num = int((120-len(string))/2.0)
	for i in range(num):
		print("-", end="")
	print(string, end="")
	for i in range(num):
		print("-", end="")
	print()

def Get_Local_Time():
	localtime = time.localtime(time.time())
	string = ""
	string += str(localtime.tm_year)+"-"
	string += str(localtime.tm_mon)+"-"
	string += str(localtime.tm_mday)+":"
	string += str(localtime.tm_hour)+"-"
	string += str(localtime.tm_min)+"-"
	string += str(localtime.tm_sec)
	return string

def Thresholdfy(array, threshold, lower, upper):
	new = []
	for obj in array:
		if(obj<threshold):
			new.append(lower)
		else:
			new.append(upper)
	return new

def Extend(array1 ,array2):
	new = []
	for obj in array1:
		new.append(obj)
	for obj in array2:
		new.append(obj)
	return new

def Get_WordPiece(word, collections):
	while(len(word)>0):
		if(word in collections):
			return word
		word = word[:-1]
	return "the"
