import _public as pb
import os
import sys
import dataset
import representation
import model
# import MCTS
import torch
import time
import numpy as np
import pickle
# import crf
# import plot as fig
# import re
# import model
# import feature as ft
# import crf
# import threading
# import time
# import model_baseline
# import sklearn.metrics as metrics
import copy



def Init():
	[pb.c2i_En, pb.i2c_En, pb.mus_En, pb.lvs_En] = pb.Pickle_Read("./sources/English")
	[pb.c2i_Ge, pb.i2c_Ge, pb.mus_Ge, pb.lvs_Ge] = pb.Pickle_Read("./sources/Germany")
	[pb.c2i_Th, pb.i2c_Th, pb.mus_Th, pb.lvs_Th] = pb.Pickle_Read("./sources/Thai")
	[pb.c2i_Ar, pb.i2c_Ar, pb.mus_Ar, pb.lvs_Ar] = pb.Pickle_Read("./sources/Arabic")
	[pb.c2i_Ja, pb.i2c_Ja, pb.mus_Ja, pb.lvs_Ja] = pb.Pickle_Read("./sources/Japanese")
	[pb.c2i_Ch, pb.i2c_Ch, pb.mus_Ch, pb.lvs_Ch] = pb.Pickle_Read("./sources/Chinese")

def main():
	print()

	examples_train, examples_test = dataset.Read_Data("./data/" + pb.source + ".csv")

	pb.LABELS = sorted(set([example.label for example in examples_train]))
	print(pb.source, pb.LABELS, len(examples_train), len(examples_test))

	textEmd = representation.Representator(examples_train)

	data_filepath = "./data/" + pb.source+"_bert"
	if (os.path.exists(data_filepath)==True):
		[examples_train, examples_test] = pb.Pickle_Read(data_filepath)

	textEmd.Get_Representations(examples_train)
	textEmd.Get_Representations(examples_test)
	# pb.Pickle_Save([examples_train, examples_test], data_filepath)

	textEmd.Get_nGram_Representations(examples_train)
	textEmd.Get_nGram_Representations(examples_test)
	# pb.Pickle_Save([examples_train, examples_test], data_filepath)

	examples_train = dataset.Get_Balanced_Data(examples_train)

	xs_train, gs_train, ys_train = dataset.Get_Encoded_Data(examples_train)
	xs_test, gs_test,  ys_test  = dataset.Get_Encoded_Data(examples_test)

	# from sklearn import svm
	# clf = svm.SVC()
	# clf.fit(xs_train, ys_train)
	# pre_labels = clf.predict(xs_test)
	# recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.LABELS, 2)
	# print("recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}".format(recall, precision, macrof1, microf1))

	print()

	mlp_model = model.MLP(xs_train, gs_train)
	rep_width, add_width, best_ma, best_mi = mlp_model.train(xs_train, gs_train, ys_train, xs_test, gs_test, ys_test)

	return rep_width, add_width, best_ma, best_mi

if __name__ == "__main__":
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	Init()

	final_ans = []

	try:
		for times in range(pb.TIME):
			print("times:{}".format(times))
			for pb.source in pb.Sources:
				for pb.target in pb.Targets:
					for pb.seed_upperbound in pb.Seed_upperbounds:
						print('pb.source=',pb.source)
						print('pb.target=',pb.target)
						print('pb.seed_upperbound=',pb.seed_upperbound)
						pb.aux_rep = "gram"
						rep_width, add_width, best_ma, best_mi = main()
						string = "{:15}\t{:15}\t{}\t{}\t{:10}\t{:10}\t{:.2%}\t{:.2%}".format(pb.source, pb.target, pb.aux_rep, pb.seed_upperbound, rep_width, add_width, best_ma, best_mi)
						print(string)
						final_ans.append(string)
			# for pb.source in pb.Sources:
			# 	pb.target = "None"
			# 	pb.aux_rep = "bert"
			# 	pb.seed_upperbound = 0
			# 	rep_width, add_width, best_ma, best_mi = main()
			# 	string = "{:15}\t{:15}\t{}\t{}\t{:10}\t{:10}\t{:.2%}\t{:.2%}".format(pb.source, pb.target, pb.aux_rep, pb.seed_upperbound, rep_width, add_width, best_ma, best_mi)
			# 	print(string)
			# 	final_ans.append(string)
	finally:
		pb.Print_Dotted_Line("Final Ans")
		for line in final_ans:
			print(line)
		pb.Print_Dotted_Line("Final Ans")
