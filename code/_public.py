# import matplotlib.pyplot as plt
# import math
import numpy as np
# import time as time
import sklearn.metrics as metrics
import pickle
# import nltk

INF = 999999999

LABELS = []
seed_minlength, seed_maxlength = 1, 5
hard_filtering = 1- 0.20
NAME = ""

TIME = 1

# Sources, source = ["English", "Germany", "Thai", "Arabic", "Japanese", "Chinese"], ""
Sources, source = ["English"], ""

# Targets, target = ["None", "English", "Germany", "Thai","Arabic", "Japanese", "Chinese", "All"], ""
Targets, target = ["Japanese"], ""

Seed_upperbounds, seed_upperbound = [5], 0
Aux_Rep, aux_rep = ["gram", "bert"], ""

c2i_En, i2c_En, mus_En, lvs_En = {}, [], [], []
c2i_Ge, i2c_Ge, mus_Ge, lvs_Ge = {}, [], [], []
c2i_Th, i2c_Th, mus_Th, lvs_Th = {}, [], [], []
c2i_Ar, i2c_Ar, mus_Ar, lvs_Ar = {}, [], [], []
c2i_Ja, i2c_Ja, mus_Ja, lvs_Ja = {}, [], [], []
c2i_Ch, i2c_Ch, mus_Ch, lvs_Ch = {}, [], [], []

def Print_Dotted_Line(title=""):
	print("--------------------------------------------------"+title+"--------------------------------------------------")

def Max_Index(array):
	max_index = 0
	for i in range(len(array)):
		if(array[i]>array[max_index]):
			max_index = i
	return max_index

def Map_To_Sorted_List(map):
	x, y = [], []
	for item in sorted(map.items(), key=lambda item: item[1], reverse=True):
		x.append(item[0])
		y.append(item[1])
	return x, y

def Get_Report(true_labels, pred_labels, labels=None, digits=4):
	recall = metrics.recall_score(true_labels, pred_labels, average='macro')
	precision = metrics.precision_score(true_labels, pred_labels, average='macro')
	macrof1 = metrics.f1_score(true_labels, pred_labels, average='macro')
	microf1 = metrics.f1_score(true_labels, pred_labels, average='micro')
	acc = metrics.accuracy_score(true_labels, pred_labels)
	return recall, precision, macrof1, microf1, acc

def Pickle_Save(variable, path):
	with open(path, 'wb') as file:
		pickle.dump(variable, file)
	print("Pickle Saved {}".format(path))

def Pickle_Read(filepath):
	with open(filepath, 'rb') as file:
		obj = pickle.load(file)
	print("Pickle Read")
	return obj
