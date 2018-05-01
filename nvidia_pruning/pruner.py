import os
import types


from keras import optimizers
import inspect
import numpy as np
import keras.backend as K
from keras.optimizers import TFOptimizer
from sklearn.metrics import accuracy_score, f1_score
from keras.models import clone_model, Model
import collections
import gc
from keras.layers import  Wrapper
from nn.optimizers import  YFOptimizer, FTML
from nn_test.data import generator
from utils.utils_exceptions import InputError
from utils.utils import mk
from config import ROOT_PRUNED
import time
from functools import wraps
import resource
import gc
import operator

def fn_timer(function):
	@wraps(function)
	def function_timer(*args, **kwargs):
		t0 = time.time()
		result = function(*args, **kwargs)
		gc.collect()
		t1 = time.time()
		print ("Total time running %s: %s seconds" %
			   (function.func_name, str(t1 - t0))
			   )
		#print ('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		return result

	return function_timer

def to_list(x):
	"""Normalizes a list/tensor into a list.
	If a tensor is passed, we return
	a list of size 1 containing the tensor.
	# Arguments
		x: target object to be normalized.
	# Returns
		A list.
	"""
	if isinstance(x, list):
		return x
	return [x]


def object_list_uid(object_list):
	object_list = to_list(object_list)
	return ', '.join([str(abs(id(x))) for x in object_list])

def join_path(list_path):
	path = '~'
	for p in list_path:
		path = os.path.join(path, p)
	return path

def get_class_params(_class):
	return inspect.getargspec (_class.__init__)

def filter_args(_class, args):
	_class_args = get_class_params(_class).args
	print(_class_args)
	d = {}
	for k in _class_args[1:]:
		try:
			d[k] = args[k]
		except:
			pass
	return d

### NVIDIA Scorer

class NVIDIA_Scorer:

	def __init__(self, model, name, model_type, optim, nb_class, batch_size=32,**args):
		self.model = model
		self.name = name
		self.model_type = model_type
		self.path = os.path.join(ROOT_PRUNED, name)
		self.optim = optim
		self.args = args
		self.nb_class = nb_class
		self.batch_size = batch_size
		if not isinstance(self.path, types.StringType):
			raise InputError('[INFO] path has to be defined')
		else:
			if os.path.isdir(self.path):
				pass
			else:
				print(ROOT_PRUNED)
				if os.path.isdir (ROOT_PRUNED):
					print('[INFO] ROOT_PRUNED exists')
					mk (self.path)
				else:
					raise InputError ('[INFO] ROOT_PRUNED has to be a path to a directory')
		self.get_optim()
		self.compile(self.optim, **args)
		self.define_gradients()
		self.create_holder()
		self.cnt_class = np.zeros((self.nb_class))
		#self.X, self.Y = self.load_data(self.train_pipe)

	def reset_scorer(self):
		self.compile(self.optim, **self.args)
		self.create_holder()
		self.cnt_class = np.zeros((self.nb_class))

	def get_names(self):
		from keras.models import Model
		holder_names = []
		def func(model, is_wrapper=False):
			for l in model.layers:
				if isinstance(l, Model):
					func(l)
				else:
					if is_wrapper:
						if isinstance(l, Wrapper):
							holder_names.append(l.name_layer)
						else:
							holder_names.append(l.name)
					else:
						holder_names.append(l.name)
			return list(set(holder_names))

		return func

	def get_layer(self, model, layer_name):
		return get_layer_rec(model, layer_name)

	def check_optim(self):
		if self.optim.lower() not in self.optims:
			raise InputError(str(self.optim.lower())+' not in '+str(self.optims))

	def set_generators(self, train_pipe):
		self.train_pipe = train_pipe
		self.n = len(self.train_pipe.augmentor_images)
		self.indexes_holder = np.zeros((self.n))
		self.gen_func = generator
		self.gen_train = self.gen_func(self.train_pipe, self.model_type, with_indexes=False, batch_size=self.batch_size)

	def create_optim(self, **args):
		if self.optim == 'yellowfin':

			filtered_args = filter_args (YFOptimizer, args)
			print(filtered_args)
			opt = TFOptimizer(YFOptimizer(**filtered_args))

		elif self.optim == 'ftml':
			filtered_args = filter_args (FTML, args)
			opt = FTML(**filtered_args)
		else:
			for name, optim_class in inspect.getmembers (optimizers):
				if inspect.isclass(optim_class) and name[0] == name[0].upper():
					filtered_args = filter_args(optim_class, args)
					return  optim_class(**filtered_args)
		return opt

	def compile(self, optim, **args):
		self.optim = optim
		self.check_optim()
		opt = self.create_optim(**args)
		self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

	def filter_tensor_name(self, w_name):
		return str('_'.join(w_name.name.split('/')[0].split('_')[:-1]))

	def has_wrapper(self):
		for layer in self.model.layers:
			if isinstance(layer, Wrapper):
				return True
		return False

	def create_wrapper_holder(self):
		if self.has_wrapper():
			self.mapping_wrapper = {layer.name_layer:layer.name for layer in self.model.layers if isinstance(layer, Wrapper)}
		else:
			self.mapping_wrapper = {}

	def define_gradients(self):
		self.create_wrapper_holder()
		self.define_properties()
		acts = []
		names_conv = []
		for index, w in enumerate(self.model.trainable_weights):
			name = w.name
			if index % 2 == 0:
				name = '_'.join(name.split('_')[:2])
				if bool(self.mapping_wrapper) and self.mapping_wrapper.has_key(name):
					name = self.mapping_wrapper[name]
					names_conv.append(name)
					layer = self.get_layer(self.model, name)
					acts.append(layer.output)
				else:
					if 'conv' in name:
						layer = self.get_layer(self.model, name)
						names_conv.append(name)
						acts.append(layer.output)
		grads = self.model.optimizer.get_gradients(loss=self.model.total_loss, params=acts)
		self.w_names = names_conv
		self.tensors = [names_conv, acts, grads]
		assert len(names_conv) + len(acts) + len(grads) == 3*(len(names_conv)), 'Wierd'

	def create_holder(self):
		self.h = {}
		for w_name in self.w_names:
			self.h[w_name] = OnlineMean(w_name, self.nb_class)

	def define_properties(self):
		self.inp = self.model.input

	def define_functions(self, tensors_grad, tensors_weights):
		inputs = [self.model.inputs[0],  # input data
						 self.model.sample_weights[0],
						 self.model.targets[0],  # labels
						 K.learning_phase (),  # train or test mode
						 ]
		self.get_gradients = K.function(inputs=inputs, outputs=tensors_grad.tolist())
		self.get_acts = K.function(inputs=inputs, outputs=tensors_weights.tolist())

	def evaluate_path(self, tensor_name, _class, name):
		name = name.replace('.png','.npy')
		path_act_name = join_path([self.path, tensor_name, _class, self.optim, name])
		path_grad_name = join_path([self.path, tensor_name, _class, self.optim, name])
		is_act = False
		is_grad = False
		if os.path.exists(path_act_name):is_act = True
		if os.path.exists(path_grad_name): is_grad = True
		return is_act, is_grad

	def _extract(self, X, Y):
		inputs = [X,  # X
				  [1],
				   Y,  # y
				  1  # learning phase in TRAIN mode
				  ]
		#name = self.get_file_name(z)
		#_class = str(np.argmax(y, axis=-1))
		acts_tensors = []
		grads_tensors = []
		if not hasattr(self, 'is_function_defined'):
			for index_tensor, tensor_name in enumerate(self.tensors[0]):
			#	is_act, is_grad = self.evaluate_path(tensor_name, _class, name)
			#	if not is_act:
				if 'conv' in tensor_name:
					acts_tensors.append([tensor_name, self.tensors[1][index_tensor]])
			#	if not is_grad:
					grads_tensors.append([tensor_name, self.tensors[2][index_tensor]])

			self.acts_tensors = np.array(acts_tensors)
			self.grads_tensors = np.array(grads_tensors)
			self.define_functions(self.grads_tensors[:,1], self.acts_tensors[:,1])
			self.is_function_defined = True

		acts = self.get_acts(inputs) #[(x, y) for x,y in zip(self.get_acts(inputs), self.acts_tensors[:,0])]
		grads = self.get_gradients(inputs) #[(x, y) for x,y in zip(self.get_gradients(inputs), self.grads_tensors[:,0])]
		y_true = np.argmax(Y, axis=-1)
		for index in range(self.batch_size):
			_class = str(y_true[index])
			self.create_scores((self.get_values_by_index(acts, index),
								self.get_values_by_index(grads, index)),
								_class)
			self.cnt_class[int(_class)] += 1

	def get_values_by_index(self, b, index):
		return [x[index] for x in b]

	def create_scores(self, values, _class, with_abs=True, with_act_norm=True, scaling_factor=1e13):
		for index in range(len(values[0])):
			act = values[0][index]
			grad = values[1][index]
			prod = np.multiply(act, grad)
			prod = np.sum(np.sum(prod, axis=0), axis=0)
			self.h[self.w_names[index]].add_arr(prod, _class)

	def get_file_name(self, z):
		return self.train_pipe.augmentor_images[z].image_file_name

	def get_output_path(self, y, z):
		name = self.get_file_name(z)
		_class = str(np.argmax(y))
		return join_path([ROOT_PRUNED, _class, name])

	def extract_tensors(self, limit_per_class=626):
		#print(self.w_names)
		cnt = 0
		self.limit_per_class = limit_per_class
		for X, Y in self.gen_train:
			if cnt == self.n or np.min(self.cnt_class) >= self.limit_per_class:
				break
			self._extract(X, Y)
			cnt += self.batch_size

	def get_optim(self, extras=['yellowFin', 'ftml']):
		self.optims = []
		for name, class_obj in inspect.getmembers(optimizers):
			c_name = str(class_obj)
			if 'class' in c_name:
				if  name.lower() not in ['optimizer', 'tfoptimizer','__builtins__', '__doc__']:
					self.optims.append(name.lower())
		self.optims = list(set(self.optims))
		self.optims+=[op.lower() for op in extras]

	def load_data(self, pipeline):
		self.L = len(pipeline.augmentor_images)
		n = np.zeros((self.L,))
		X_d = []
		Y_d = []
		cnt = 0
		for X, Y, Z in generator(pipeline, self.model_type, with_indexes=True):
			if cnt > self.L:
				break
			for x, y, z in zip(X, Y, Z):
				if n[z] == 0:
					n[z] = 1
					X_d.append(x)
					Y_d.append(y)
				cnt += 1
		return np.array(X_d), np.array(Y_d)

### ONLINE MEAN

class OnlineMean:

	def __init__(self, key, nb_class):
		self.key = key
		self.nb_class = nb_class
		self.cnt = np.zeros((self.nb_class,))
		self.arr = [np.array(0) for _ in range(self.nb_class)]
	@property
	def key(self):
		return self.key

	@property
	def cnt(self):
		return self.cnt

	@property
	def arr(self):
		return self.arr

	def add_arr(self, ob, index):
		index = int(index)
		if self.cnt[index] == 0:
			self.arr[index] = ob
		else:
			self.arr[index] += ob #(1/float(self.cnt[index]+1)) * ((self.cnt[index])*self.arr[index] + ob)

		self.cnt[index]+=1

	def get_arr_by_index(self, index):
		index = int(index)
		return self.arr[index]

	def rescale(self, v):
		v = np.abs(v)
		v /= np.linalg.norm(v)
		return v

	def get_scores(self):
		return np.array([self.rescale(np.mean(self.arr, axis=0))])

def get_layer_rec_old(model, layer_name):
	print(model.layers[-1].name)
	for layer in model.layers:
		print (layer.name)
		if isinstance(layer, Model):
			return get_layer_rec(layer, layer_name)
		else:
			if layer.name == layer_name:
				return layer

def get_layer_rec(model, layer_name):
	out_layer = None
	for layer in model.layers: #order_layers(model.layers):
		if isinstance(layer, Model):
			out_layer = get_layer_rec(layer, layer_name)
		else:
			if layer.name == layer_name:
				out_layer = layer
	return out_layer



### TRAINER CLASS

class NVIDIA_Trainer:

	def __init__(self, model, name, model_type, optim, nb_class, with_debug=True, mode='kernel_only'):
		gc.collect()
		self.model = model
		self.name = name
		self.optim = optim
		self.model_type = model_type
		self.path = os.path.join(ROOT_PRUNED, name)
		if isinstance(nb_class, types.NoneType):
			raise Exception
		self.nb_class = nb_class
		self.mode = mode
		self.w_names = [name for name in self.get_names()(model) if
						isinstance(self.model.get_layer(name), Wrapper)]
		self.with_debug = with_debug
		self.create_wrapper_holder()
		#self.eval_model = clone_model(model)
		#self.w_shapes = {w_name:self.get_layer_shape(w_name).shape for w_name in self.w_names}

	def filter_tensor_name(self, w_name):
		return str('_'.join(w_name.split('/')[0].split('_')[:2]))

	def load_data(self, pipeline):
		self.L = len(pipeline.augmentor_images)
		n = np.zeros((self.L,))
		X_d = []
		Y_d = []
		cnt = 0
		for X, Y, Z in generator(pipeline, self.model_type, with_indexes=True):
			if cnt > self.L:
				break
			for x, y, z in zip(X, Y, Z):
				if n[z] == 0:
					n[z] = 1
					X_d.append(x)
					Y_d.append(y)
				cnt += 1
		return np.array(X_d), np.array(Y_d)

	def set_shapes(self, shapes):
		self.w_shapes = shapes
		self.nb_filters = np.sum([self.w_shapes[k][-1] for k in self.w_shapes.keys() if 'conv' in k])

	def get_layer_shape(self, name):
		return self.model.get_layer(name).get_output_at(0).eval(session=K.get_session(),feed_dict={
			self.model.input: np.random.normal(0, 1, (1, 224, 224, 3))
		})

	def create_wrapper_holder(self):
		self.mapping_wrapper = {}
		for layer in self.model.layers:
			if isinstance(layer, Wrapper):
				self.mapping_wrapper[layer.name] = layer.name_layer
			else:
				self.mapping_wrapper[layer.name] = layer.name

	def get_names(self):
		from keras.models import Model
		holder_names = []
		def func(model, is_wrapper=False):
			for l in model.layers:
				if isinstance(l, Model):
					func(l)
				else:
					if is_wrapper:
						if isinstance(l, Wrapper):
							holder_names.append(l.name_layer)
						else:
							holder_names.append(l.name)
					else:
						holder_names.append(l.name)
			return list(set(holder_names))

		return func

	def clone_keras_model(self):
		eval_model = clone_model(self.model)
		for l, l2 in zip(self.model.layers, eval_model.layers):
			try:
				l2.set_weights(l.get_weights())
			except:
				pass
		return eval_model

	def create_subset(self, percentage, data_type):
		if self.is_train(data_type):
			L = len(self.X_train)
		else:
			L = len(self.X_test)
		mask = np.random.permutation(range(L))
		indexes = np.where(mask < int(percentage * L))
		print(len(indexes[0]))
		self.indexes = indexes

	def is_train(self, data_type):
		if data_type == 'train':return True
		return False

	def evaluate(self, model, frozen, is_returned=True, percentage=0.02, data_type='test', is_last=False):
		if not frozen:
			self.create_subset(percentage, data_type)
		if self.is_train(data_type):
				X = self.X_train[self.indexes]
				Y = self.Y_train[self.indexes]
		else:
			X = self.X_test[self.indexes]
			Y = self.Y_test[self.indexes]
		if is_last:
			X = self.X_test
			Y = self.Y_test
		preds = model.predict(X)
		y_true, y_pred = np.argmax(Y, axis=1), np.argmax(preds, axis=1)
		score = accuracy_score(y_true, y_pred)
		print(f1_score(y_true, y_pred, average='macro'))
		if not frozen:
			self.init_score = score
			print('[INFO] INIT SCORE : '+str(self.init_score))
		if is_returned:
			return score

	def evaluate_all(self, model, is_fine=False):
		preds = model.predict(self.X_test)
		y_true, y_pred = np.argmax(self.Y_test, axis=1), np.argmax(preds, axis=1)
		if not is_fine:
			self.pruned_score = accuracy_score(y_true, y_pred)
			print('pruned_score f1_score : '+str(f1_score(y_true, y_pred, average='macro')))
			print('pruned_score acc_score : '+str(self.pruned_score))
		else:
			self.fine_score = accuracy_score(y_true, y_pred)
			print('fine f1_score : ' + str(f1_score(y_true, y_pred, average='macro')))
			print('fine acc_score : ' + str(self.fine_score))



	def set_pipelines(self, train_pipe, test_pipe):
		self.train_pipe = train_pipe
		self.test_pipe = test_pipe

	def init_pruning(self):
		self.X_test, self.Y_test = self.load_data(self.test_pipe)
		self.evaluate(self.model, False, is_last=True)

	def load_weights(self, name):
		self.model.load_weights(name)

	@fn_timer
	def score_model(self, model, limit_per_class, **args):
		if not hasattr(self, 'nvidia_scorer'):
			self.nvidia_scorer = NVIDIA_Scorer(model,
												   self.name,
												   self.model_type,
												   self.optim,
												   self.nb_class,
												   **args)
		else:
			self.nvidia_scorer.model = model
			self.nvidia_scorer.reset_scorer()
		self.nvidia_scorer.set_generators(self.train_pipe)
		self.nvidia_scorer.extract_tensors(limit_per_class=limit_per_class)
		self.scores = self.nvidia_scorer.h
		key = self.nvidia_scorer.h.keys()[0]
		print(self.scores[key].cnt)



	def ratio(self):
		print('[INFO] SCORES : '+str(self.init_score)+'/'+str(self.fine_score)+'/'+str(self.pruned_score))
		return self.init_score - self.fine_score

	def start_pruning(self, nb_iterations=10, nb_epochs=2, incremental_pruned=7.5, batch_size=32, limit_per_class=2, with_debug=False, acc_drop=5, **args):
		self.cnt_loop = 0
		self.incremental_pruned = incremental_pruned
		self.nb_iterations = nb_iterations
		self.nb_epochs = nb_epochs
		self.batch_size = batch_size
		self.limit_per_class = limit_per_class
		self.tobreak = False
		self.count_zeros(self.model)
		if not with_debug:
			for i in range(nb_iterations - 1):
				print('[INFO] ITERATION :'+str(self.cnt_loop))
				self.loop(self.model)
				self.cnt_loop+=1
				ratio = self.ratio()
				print('[INFO] ratio : '+str(ratio))
				if ratio > acc_drop:
					self.tobreak = True
					break
			print('[INFO] ITERATION :' + str(self.cnt_loop))
			if not self.tobreak:
				self.loop(self.model)

	def get_pos(self, index, pos):
		cnt = 0
		for i in pos:
			if i < index:
				cnt += 1
		return cnt

	def linear_epochs(self, step=10):
		if (self.cnt_loop + 1) % step == 0:
			self.nb_epochs+=1
			self.limit_per_class += 1
		print('[INFO] TRAINING FOR EPOCHS : '+str(self.nb_epochs))

	@fn_timer
	def get_indexes(self, with_print=False):
		self.w_names = sorted([str(f) for f in self.scores.keys() if 'conv' in f])
		scores = np.concatenate([self.scores[w_name].get_scores()[0] for w_name in self.w_names])
		pos = np.cumsum([self.scores[w_name].get_scores().shape[-1] for w_name in self.w_names])
		self.th = (self.cnt_loop + 1) * self.incremental_pruned
		print('[INFO] th : '+str(self.th))
		limit = int(self.nb_filters*self.th*0.01)
		print('[INFO] limit : '+str(limit))
		indices = np.argsort(scores)[:limit]
		d = collections.defaultdict(list)
		for i in indices.tolist():
			index = self.get_pos(i, pos)
			rest = 0
			if index == 0:
				pass
			else:
				rest = pos[index - 1] + 1
			if i - rest < self.w_shapes[self.mapping_wrapper[self.w_names[index]]][-1]:
				d[self.w_names[index]].append(i - rest)
		if with_print:
			h = []
			for k in sorted(d.keys()):
				h.append([self.mapping_wrapper[k], len(d[k])])
			h.sort(key=operator.itemgetter(0))
			for e in h:
				print(e)
			#print(h)
			#print(self.mapping_wrapper[k], len(d[k]))
		return d

	@fn_timer
	def prune(self, model):
		self.d = self.get_indexes(with_print=True)
		for w_name in sorted(self.w_names)[1:]:
			wrapper = model.get_layer(w_name)
			if isinstance(wrapper, Wrapper):
				layer = wrapper.layer
			else:
				layer = wrapper
			w, b = layer.get_weights()
			#if hasattr(self, 'd'):
			#	for l in self.d[w_name]:
			#			w[:, :, :, l] = 0
			if isinstance(wrapper, Wrapper):
				wrapper.set_mask(self.d[w_name])
				#wrapper.layer.set_weights([w, b])
			else:
				layer.set_weights([w, b])
	@fn_timer
	def count_zeros(self, model):
		sum_zeros = 0
		sum_all = 0
		#print(self.w_names)
		for w_name in sorted(self.w_names):
			layer = model.get_layer(w_name)
			if isinstance(layer, Wrapper):
				mask = layer.mask.eval(session=K.get_session())
				mask = mask[0, 0, :]
				s = mask.shape[0]
				zeros = len(np.where(mask == 0)[0])
				sum_zeros+=zeros
				sum_all+=s
		print(sum_zeros, sum_all)
		self.pruned_zeros = sum_zeros/float(sum_all)
		print('percentage of zeros : '+str(100 * self.pruned_zeros))

	@fn_timer
	def train(self, model):
		self.linear_epochs()
		history = model.fit_generator(generator(self.train_pipe, self.model_type),
								 validation_data=(self.X_test, self.Y_test),
								 epochs=self.nb_epochs,
								 steps_per_epoch=len(self.train_pipe.augmentor_images)//self.batch_size,
								 shuffle=True)
		try:
			self.fine_score = history.history['val_acc'][-1]
		except:
			self.fine_score = history.history['val_acc'][0]
	@fn_timer
	def save_model(self, model):
		name = 'weights/'+str(np.round(self.fine_score,2))\
			   +'_'+str(self.th)\
			   +'_'+str(np.round(self.pruned_score,2))\
			   +'.h5'

		model.save(name)

	@fn_timer
	def loop(self, model):
		#self.count_zeros(model)
		self.score_model(model, self.limit_per_class)
		self.prune(model)
		self.evaluate_all(model)
		self.count_zeros(model)
		if (self.init_score - self.pruned_score) > 0.25:
			self.train(model)
		else:
			self.fine_score = self.pruned_score
		#self.evaluate_all(model, is_fine=True)
		self.save_model(model)
		#self.count_zeros(model)






