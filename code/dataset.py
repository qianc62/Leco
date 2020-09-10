import _public as pb
import csv
import numpy as np
import random



class Example:
	def __init__(self):
		self.label = ""
		self.En, self.Ch, self.Ge, self.Ja, self.Th, self.Ar = "", "", "", "", "", ""
		self.En_bert, self.Ch_bert, self.Ge_bert, self.Ja_bert, self.Th_bert, self.Ar_bert = [], [], [], [], [], []
		self.En_gram, self.Ch_gram, self.Ge_gram, self.Ja_gram, self.Th_gram, self.Ar_gram = [], [], [], [], [], []

		self.En_baseline, self.Ch_baseline, self.Ge_baseline, self.Ja_baseline, self.Th_baseline, self.Ar_baseline = [], [], [], [], [], []

def Read_Data(path):
	examples_train, exapmles_test = [], []

	examples = examples_train
	with open(path, encoding='UTF-8-sig') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			string = "\t".join(row)
			string = string.lower()

			if(string.startswith("test")):
				examples = exapmles_test
				continue

			if( len(string)>0 ):
				objs = string.split("\t")

				example = Example()

				example.label = objs[0]
				example.En = objs[1] if len(objs[1]) > 0 else "unknown"
				example.Ge = objs[2] if len(objs[2]) > 0 else "unknown"
				example.Th = objs[3] if len(objs[3]) > 0 else "unknown"
				example.Ar = objs[4] if len(objs[4]) > 0 else "unknown"
				example.Ja = objs[5] if len(objs[5]) > 0 else "unknown"
				example.Ch = objs[6] if len(objs[6]) > 0 else "unknown"
				examples.append(example)

	return examples_train, exapmles_test

def Get_Balanced_Data(examples):

	examples_list = [[] for _ in pb.LABELS]
	sampled_examples_list = [[] for _ in pb.LABELS]

	for example in examples:
		index = pb.LABELS.index(example.label)
		examples_list[index].append(example)
		sampled_examples_list[index].append(example)

	sample_num = int( np.max([len(obj) for obj in examples_list]) * 1.0 )

	balanced_examples = []

	for i in range(len(sampled_examples_list)):
		while (len(sampled_examples_list[i]) < sample_num):
			example = random.choice( examples_list[i] )
			sampled_examples_list[i].append( example )
		balanced_examples.extend(sampled_examples_list[i])

	return balanced_examples

def Get_Encoded_Data(examples):

	xs, gs, ys = [], [], []
	for example in examples:
		x, g, y = [0.0], [0.0], [0.0]

		if(pb.source=="English"):
			x.extend(example.En_bert)
		elif(pb.source=="Germany"):
			x.extend(example.Ge_bert)
		elif (pb.source == "Thai"):
			x.extend(example.Th_bert)
		elif (pb.source == "Arabic"):
			x.extend(example.Ar_bert)
		elif (pb.source == "Japanese"):
			x.extend(example.Ja_bert)
		elif (pb.source == "Chinese"):
			x.extend(example.Ch_bert)

		# if (pb.source == "English"):
		# 	x.extend(example.En_gram)
		# elif (pb.source == "Germany"):
		# 	x.extend(example.Ge_gram)
		# elif (pb.source == "Thai"):
		# 	x.extend(example.Th_gram)
		# elif (pb.source == "Arabic"):
		# 	x.extend(example.Ar_gram)
		# elif (pb.source == "Japanese"):
		# 	x.extend(example.Ja_gram)
		# elif (pb.source == "Chinese"):
		# 	x.extend(example.Ch_gram)

		if(pb.aux_rep=="bert"):
			if (pb.target == "English"):
				g.extend(example.En_bert)
			elif (pb.target == "Germany"):
				g.extend(example.Ge_bert)
			elif (pb.target == "Thai"):
				g.extend(example.Th_bert)
			elif (pb.target == "Arabic"):
				g.extend(example.Ar_bert)
			elif (pb.target == "Japanese"):
				g.extend(example.Ja_bert)
			elif (pb.target == "Chinese"):
				g.extend(example.Ch_bert)
			elif (pb.target == "All"):
				g.extend(example.En_bert)
				g.extend(example.Ge_bert)
				g.extend(example.Th_bert)
				g.extend(example.Ar_bert)
				g.extend(example.Ja_bert)
				g.extend(example.Ch_bert)
		elif (pb.aux_rep == "gram"):
			if (pb.target == "English"):
				g.extend(example.En_gram)
			elif (pb.target == "Germany"):
				g.extend(example.Ge_gram)
			elif (pb.target == "Thai"):
				g.extend(example.Th_gram)
			elif (pb.target == "Arabic"):
				g.extend(example.Ar_gram)
			elif (pb.target == "Japanese"):
				g.extend(example.Ja_gram)
			elif (pb.target == "Chinese"):
				g.extend(example.Ch_gram)
			elif (pb.target == "All"):
				g.extend(example.En_gram)
				g.extend(example.Ge_gram)
				g.extend(example.Th_gram)
				g.extend(example.Ar_gram)
				g.extend(example.Ja_gram)
				g.extend(example.Ch_gram)

		y = pb.LABELS.index(example.label)

		xs.append(x)
		gs.append(g)
		ys.append(y)

	return xs, gs, ys
