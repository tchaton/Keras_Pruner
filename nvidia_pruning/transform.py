import keras
from keras.models import *
from keras.layers import *
import inspect
import types

class Transformer(object):

    def __init__(self, model):
        self.clone_keras_model(model)
        self.w_names = self.get_names()(self.model)
        self.get_shapes_output()
        self.get_shapes()

    def get_layer_shape(self, name):
        return self.model.get_layer(name).input.eval(session=K.get_session(), feed_dict={
            self.model.input: np.random.normal(0, 1, (1, 224, 224, 3))
        })

    def get_layer_output_shape(self, name):
        return self.model.get_layer(name).output.eval(session=K.get_session(), feed_dict={
            self.model.input: np.random.normal(0, 1, (1, 224, 224, 3))
        })

    def get_shapes(self):
        self.w_shapes = {w_name:self.get_layer_shape(w_name).shape[1:] for w_name in self.w_names}

    def get_shapes_output(self):
        self.w_out_shapes = {w_name:self.get_layer_output_shape(w_name).shape[1:] for w_name in self.w_names[1:]}

    def clone_keras_model(self, model):
        self.model = clone_model(model)
        for l, l2 in zip(model.layers, self.model.layers):
            try:
                l2.set_weights(l.get_weights())
            except:
                pass

    def get_names(self):
        from keras.models import Model
        holder_names = []

        def func(model):
            for l in model.layers:
                if isinstance(l, Model):
                    func(l)
                else:
                    holder_names.append(l.name)
            return list(set(holder_names))
        return func

    def convert(self, wrapper=None):
        if isinstance(wrapper, types.NoneType): return self.model
        seq = Sequential()
        for index, layer in enumerate(self.model.layers):
            if 'conv' in layer.name.lower():
                input_shape = self.w_shapes[layer.name]
                output_shape = self.w_out_shapes[layer.name]
                if index == 1:
                    seq.add(wrapper(layer,
                                      input_shape=input_shape, output_shape=output_shape))
                else:
                    seq.add(wrapper(layer, output_shape=output_shape))
            elif 'input' in layer.name.lower():
                pass
            else:
                seq.add(layer)
        return seq