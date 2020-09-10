import _public as pb
import numpy as np
import torch
from torch import optim
# import plot as fig
import dataset



class MLP(torch.nn.Module):
	def __init__(self, xs, gs):
		super(MLP, self).__init__()

		self.epochs = 1000 + 1
		self.print_delta = 100

		self.batch_size = 64
		self.learning_rate = 0.0001

		shape = np.array(xs).shape
		self.rep_width = shape[len(shape)-1]
		self.hidden_nn_num = 400

		shape = np.array(gs).shape
		self.add_width = shape[len(shape)-1]

		print("self.rep_width:{}".format(self.rep_width))
		print("self.add_width:{}".format(self.add_width))

		self.gate = torch.autograd.Variable(torch.randn(1, self.add_width))

		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(in_features=self.rep_width+self.add_width, out_features=200),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(in_features=200, out_features=100),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(in_features=100, out_features=50),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(in_features=50, out_features=len(pb.LABELS)),
			torch.nn.Softmax()
		)

	def forward(self, x, g):
		x = x.view(-1, self.rep_width)
		# print(x.shape)

		g = g.view(-1, self.add_width)
		# print(g.shape)

		a = g * self.gate
		# # print(a.shape)
		#
		g = a * g
		# # print(g.shape)

		o = torch.cat([x, g], 1)
		# print(o.shape)

		o = self.mlp(o)

		return o

	def train(self, xs_train, gs_train, ys_train, xs_test, gs_test, ys_test):

		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
		all_gs_test  = torch.autograd.Variable(torch.Tensor(np.array(gs_test)))

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

		best_ma, best_mi = 0.0, 0.0
		best_factual_ma, best_factual_mi = 0.0, 0.0
		best_counterfactual_ma, best_counterfactual_mi = 0.0, 0.0
		for epoch in range(self.epochs):
			optimizer.zero_grad()

			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)

			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
			batch_gs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(gs_train)  if i in rand_index])))
			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))

			train_prediction = self.forward(batch_xs, batch_gs)

			loss = criterion(train_prediction, batch_ys)

			loss.backward()

			optimizer.step()

			if (epoch % self.print_delta == 0):
				# prediction_test = self.forward(all_xs_test, all_gs_test)
				# pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
				# recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.LABELS, 2)
				# if (microf1 > best_mi):
				# 	best_ma, best_mi = macrof1, microf1
				# print( "[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}    madev:{:.2%}    midev:{:.2%}".format(epoch, recall, precision, macrof1, microf1, best_ma, best_mi))

				prediction_test = self.forward(all_xs_test, all_gs_test).data.numpy()

				none_prediction = self.forward(torch.autograd.Variable(torch.Tensor(np.array([np.zeros(768+1)]))),
											   torch.autograd.Variable(torch.Tensor(np.array([np.zeros(4920+1)])))).data.numpy()[0]
				print("none_prediction's output", none_prediction)

				factual_prediction = prediction_test
				counterfactual_prediction = [prediction - none_prediction for prediction in factual_prediction]

				pre_labels = [pb.Max_Index(line) for line in factual_prediction]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.LABELS, 2)
				print("[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}".format(epoch,recall,precision,macrof1,microf1))
				best_factual_ma = max(best_factual_ma, macrof1)
				best_factual_mi = max(best_factual_mi, microf1)

				pre_labels = [pb.Max_Index(line) for line in counterfactual_prediction]
				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.LABELS, 2)
				print("[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}".format(epoch,recall,precision,macrof1,microf1))
				print()
				best_counterfactual_ma = max(best_counterfactual_ma, macrof1)
				best_counterfactual_mi = max(best_counterfactual_mi, microf1)


		print("Factual: {:.2%} {:.2%}".format(best_factual_ma, best_factual_mi))
		print("Counter: {:.2%} {:.2%}".format(best_counterfactual_ma, best_counterfactual_mi))
		print("Delta:   {:.2%} {:.2%}".format(best_counterfactual_ma - best_factual_ma,best_counterfactual_mi - best_factual_mi))
		writer = open("./_.txt", "a+")
		writer.write("\n--------------------\n")
		writer.write("Factual: {:.2%} {:.2%}\n".format(best_factual_ma, best_factual_mi))
		writer.write("Counter: {:.2%} {:.2%}\n".format(best_counterfactual_ma, best_counterfactual_mi))
		writer.write("Delta:   {:.2%} {:.2%}\n".format(best_counterfactual_ma - best_factual_ma,best_counterfactual_mi - best_factual_mi))
		writer.write("--------------------\n")
		writer.close()

		return self.rep_width, self.add_width, best_ma, best_mi


# Baselines
# class RWMD_CC(torch.nn.Module):
#
# 	def distance(self, mat1, mat2):
# 		mat1 = np.array(mat1)
# 		# shape1 = mat1.shape
# 		v1 = mat1.reshape((-1, ))
#
# 		mat2 = np.array(mat2)
# 		# shape2 = mat2.shape
# 		v2 = mat2.reshape((-1, ))
#
# 		dis = np.sum( np.abs( v1 - v2 ) )
#
# 		return dis
#
# 	def test(self, examples_train, examples_test):
# 		for e1 in examples_test:
# 			minDis, bestExample = pb.INF, None
# 			for e2 in examples_train:
# 				dis = self.distance(e1.word2vec_mat, e2.word2vec_mat)
# 				if(dis < minDis):
# 					minDis = dis
# 					bestExample = e2
# 			e1.mssm_label = bestExample.label
#
# 		true_labels = [e.label for e in examples_test]
# 		pred_labels = [e.mssm_label for e in examples_test]
# 		recall, precision, macrof1, microf1, acc = pb.Get_Report(true_labels, pred_labels)
# 		print("{:.4f}\t{:.4f}".format(macrof1, microf1))
# 		return macrof1, microf1
#
# class TextWordCNN(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(TextWordCNN, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
# 		self.in_channels = 1
# 		self.out_channels = 2
# 		self.windows = [2, 3, 4]
# 		self.height = np.array(xs).shape[2]
# 		self.width  = np.array(xs).shape[3]
# 		self.rep_width = self.height * self.width
#
# 		self.convs = torch.nn.ModuleList([
# 				torch.nn.Sequential( torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(h, self.width), stride=(1, self.width), padding=0),
# 				torch.nn.ReLU(),
# 				torch.nn.MaxPool2d(kernel_size=(self.height-h+1, 1), stride=(self.height-h+1, 1))
# 			) for h in [2, 3, 4] ])
#
# 		self.fc = torch.nn.Linear( in_features=len(self.windows)*self.out_channels, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		x = torch.cat([conv(x) for conv in self.convs], dim=1)
# 		x = x.view(-1, x.size(1))
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# 	def Print(self):
# 		for pa in self.parameters():
# 			print(pa)
#
# class TextRNN(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(TextRNN, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 32
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.w2v_size = 100
# 		self.hidden_size = 64
# 		self.layer_num = 1
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		self.rnn = torch.nn.RNN( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True)
# 		self.fc = torch.nn.Linear(self.hidden_size, len(pb.label_histogram_x))
#
# 	def forward(self, x):
# 		out, _ = self.rnn(x, None)
# 		out = self.fc(out)
# 		out = out[:, -1, :]
# 		return out
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		# for i,x in enumerate(xs_test):
# 		# 	print(i, np.array(x))
# 		# x = np.array(xs_test)
#
# 		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		# optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
# 			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
# 			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class TextBiLSTM(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(TextBiLSTM, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.w2v_size = 100
# 		self.hidden_size = 200
# 		self.layer_num = 1
# 		self.train_acc_list = []
# 		self.test_acc_list = []
# 		self.rnn = torch.nn.LSTM( input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)
# 		self.fc = torch.nn.Linear(self.hidden_size*2, len(pb.label_histogram_x))
#
# 	def forward(self, x):
# 		out, _ = self.rnn(x, None)
# 		out = self.fc(out)
# 		out = out[:, -1, :]
# 		return out
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
# 			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
# 			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class TextRCNN(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(TextRCNN, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.w2v_size = 100
# 		self.hidden_size = 200
# 		self.layer_num = 1
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		self.rnn = torch.nn.LSTM(input_size=self.w2v_size, hidden_size=self.hidden_size, num_layers=self.layer_num, batch_first=True, bidirectional=True)
#
# 		self.height = np.array(xs).shape[1]
# 		self.width = self.hidden_size * 2
# 		self.rep_width = self.height*2
#
# 		self.convs = torch.nn.Sequential(
# 			torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, self.width), stride=(1, self.width), padding=0),
# 			torch.nn.ReLU(),
# 			torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
# 		)
#
# 		self.fc = torch.nn.Linear(self.rep_width, len(pb.label_histogram_x))
#
# 	def forward(self, x):
# 		out, _ = self.rnn(x, None)
# 		out = out.unsqueeze(1)
# 		out = self.convs(out)
# 		out = out.view(-1, out.size(1)*out.size(2)*out.size(3))
# 		out = self.fc(out)
# 		return out
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_test = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
# 			batch_xs = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train) if i in rand_index])))
# 			batch_ys = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class Word2Vec(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(Word2Vec, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		shape = np.array(xs).shape
# 		self.rep_width = shape[len(shape)-1]
#
# 		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		x = x.view(-1, self.rep_width)
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class FastText(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(FastText, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		shape = np.array(xs).shape
# 		self.rep_width = shape[len(shape)-1]
#
# 		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		# print(x.size)
# 		# last_dimension = x.size(len(x.size)-1)
# 		x = x.view(-1, self.rep_width)
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print( "[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class ELMo(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(ELMo, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		shape = np.array(xs).shape
# 		self.rep_width = shape[len(shape)-1]
#
# 		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		x = x.view(-1, self.rep_width)
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class GPT2(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(GPT2, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 500
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		shape = np.array(xs).shape
# 		self.rep_width = shape[len(shape)-1]
#
# 		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		x = x.view(-1, self.rep_width)
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
#
# class BERT(torch.nn.Module):
# 	def __init__(self, xs, ys):
# 		super(BERT, self).__init__()
#
# 		self.epochs = 4501
# 		self.batch_size = 64
# 		self.print_delta = 1
# 		self.learning_rate = 0.0001
# 		self.hidden_nn_num = 200
# 		self.train_acc_list = []
# 		self.test_acc_list = []
#
# 		shape = np.array(xs).shape
# 		self.rep_width = shape[len(shape)-1]
#
# 		self.fc = torch.nn.Linear( in_features=self.rep_width, out_features=len(pb.label_histogram_x) )
#
# 	def forward(self, x):
# 		x = x.view(-1, self.rep_width)
# 		o = self.fc(x)
# 		return o
#
# 	def train(self, xs_train, ys_train, xs_test, ys_test):
#
# 		all_xs_train = torch.autograd.Variable(torch.Tensor(np.array(xs_train)))
# 		all_xs_test  = torch.autograd.Variable(torch.Tensor(np.array(xs_test)))
#
# 		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
# 		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# 		max_macrof1, max_microf1 = 0.0, 0.0
# 		for epoch in range(self.epochs):
# 			optimizer.zero_grad()
#
# 			rand_index = np.random.choice(len(xs_train), size=self.batch_size, replace=False)
#
# 			batch_xs  = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(xs_train)  if i in rand_index])))
# 			batch_ys  = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(ys_train) if i in rand_index])))
#
# 			train_prediction = self.forward(batch_xs)
#
# 			loss = criterion(train_prediction, batch_ys)
#
# 			loss.backward()
#
# 			optimizer.step()
#
# 			if (epoch % self.print_delta == 0):
# 				prediction_test = self.forward(all_xs_test)
# 				pre_labels = [pb.Max_Index(line) for line in prediction_test.data.numpy()]
# 				recall, precision, macrof1, microf1, acc = pb.Get_Report(ys_test, pre_labels, pb.label_histogram_x, 2)
# 				if (microf1 > max_microf1):
# 					max_macrof1 = macrof1
# 					max_microf1 = microf1
# 				# print("[{:4d}]\tma:{:.2%}\tmi:{:.2%}\tdev( {:.4f}\t{:.4f} )".format(epoch, macrof1, microf1, max_macrof1, max_microf1))
#
# 				if(pb.metric_name=="ma" and max_macrof1>=pb.metric_value):
# 					return max_macrof1, max_microf1
# 				if (pb.metric_name=="mi" and max_microf1>=pb.metric_value):
# 					return max_macrof1, max_microf1
#
# 		return max_macrof1, max_microf1
#
# 	def test(self, examples):
# 		for example in examples:
# 			xs, v1s, v2s, v3s, v4s, v5s, v6s, ys = dataset.Get_Encoded_Data([example])
# 			all_xs  = torch.autograd.Variable(torch.Tensor(np.array(xs)))
# 			example.mssm_probdis = self.forward(all_xs).data.numpy()[0]
# 			example.mssm_label = pb.label_histogram_x[pb.Max_Index(example.mssm_probdis)]
#
# 	def Get_Accuracy_of_Distributions(self, preddis, ys):
# 		up, down = 0, 0
# 		for i in range(len(ys)):
# 			if (pb.Max_Index(preddis[i]) == ys[i]): up += 1
# 			down += 1
# 		return up * 1.0 / down
