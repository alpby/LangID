import numpy as np
import theano
import lasagne

import PIL.Image as Image

class Network:

    def __init__(self, trndata, tstdata, batchsize, dropout, mode, **kwargs):
        def convpool(model,filters,filterSize,stride,pooling,poolingStride,padding):
            model = lasagne.layers.Conv2DLayer(model,filters,filterSize,stride)
            model = lasagne.layers.MaxPool2DLayer(model,pooling,poolingStride,padding)
            model = lasagne.layers.BatchNormLayer(model)
            return model

        self.trndata = trndata
        self.tstdata = tstdata
        self.batchsize = batchsize
        self.dropout = dropout
        self.mode = mode

        self.input_var = theano.tensor.tensor4('input_var')
        self.answer_var = theano.tensor.ivector('answer_var')

        model = lasagne.layers.InputLayer((None, 1, 256, 856), self.input_var)

        model = convpool(model,16,(7,7),1,(3,3),2,2)
        model = convpool(model,32,(5,5),1,(3,3),2,2)
        model = convpool(model,64,(3,3),1,(3,3),2,2)
        model = convpool(model,128,(3,3),1,(3,3),2,2)
        model = convpool(model,128,(3,3),1,(3,3),2,2)
        model = convpool(model,256,(3,3),1,(3,3),(3,2),2)

        model = lasagne.layers.DenseLayer(model, 1024)
        model = lasagne.layers.BatchNormLayer(model)
        if (self.dropout > 0):
            model = lasagne.layers.dropout(model, self.dropout)

        model = lasagne.layers.DenseLayer(model, 5, nonlinearity=lasagne.nonlinearities.softmax)

        self.params = lasagne.layers.get_all_params(model, trainable=True)
        self.prediction = lasagne.layers.get_output(model)

        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.answer_var).mean()
        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=0.003)
        # TODO: L2 regularization may be applied here
        # Try with different optimizers

        if self.mode == 'train':
            self.train_fn = theano.function([self.input_var,self.answer_var],[self.prediction,self.loss],updates=updates)
        self.test_fn = theano.function([self.input_var,self.answer_var],[self.prediction,self.loss])

    def step(self, batch, mode):
		def read_batch(self, data, batch):

            start = batch * self.batchsize
            data = np.zeros((self.batchsize, 1, 256, 856), dtype=np.float32)
            answers = list()

            for i in range(start, start + self.batchsize):
                answers.append(int(data[i].split(',')[1]))
                name = data[i].split(',')[0]
                img = Image.open("../../Data/Topcoder1/trainingData/png/" + name + ".png")
                data[i - start, 0, :, :] = np.array(img).astype(np.float32) / 256.0

            answers = np.array(answers, dtype=np.int32)
            return data, answers

		if (mode == "train"):
			data, answers = read_batch(self.trndata, batch)
			theano_fn = self.train_fn
		elif (mode == "test" or mode == "predict"):
			data, answers = read_batch(self.tstdata, batch)
			theano_fn = self.test_fn
		elif (mode == "predict_on_train"):
			data, answers = read_batch(self.trndata, batch)
			theano_fn = self.test_fn
		else:
			raise Exception("unrecognized mode")

		ret = theano_fn(data, answers)
		return {"prediction": ret[0],
				"answers": answers,
				"current_loss": ret[1],
				}
