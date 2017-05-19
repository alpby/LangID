import numpy as np
import theano
import lasagne
import PIL.Image as Image

class Model:
    def __init__(self, datapath, dataclass, trndata, tstdata, batchsize, rnnunits, **kwargs):
        # TODO: L2 regularization may be applied here
        # Try with different optimizers
        # Apply dropout
        # Try with different RNN models
        self.trndata = trndata
        self.tstdata = tstdata
        self.batchsize = batchsize
        self.units = rnnunits
        self.dataclass = dataclass
        self.datapath = datapath

        self.spectrograms = theano.tensor.tensor3('spectrograms')
        self.langs = theano.tensor.ivector('langs')

        model = lasagne.layers.InputLayer((None, 800, 256), 2 * self.spectrograms - 1)  # InputLayer
        model = lasagne.layers.GRULayer(model, self.units)                              # GRULayer
        model = lasagne.layers.BatchNormLayer(model)                                    # BatchNormalization Layer
        model = lasagne.layers.GRULayer(model, self.units, only_return_final=True)      # GRULayer
        model = lasagne.layers.DenseLayer(model, self.dataclass, nonlinearity=lasagne.nonlinearities.softmax)    # Softmax

        self.params = lasagne.layers.get_all_params(model, trainable=True)
        self.prediction = lasagne.layers.get_output(model)

        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.langs).mean()
        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=0.003)

        self.train_fn = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss],updates=updates)
        self.test_fn = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss])

    def step(self, batch, mode):
		def batchthrough(specs, batch):

			start = batch * self.batchsize
			data = np.zeros((self.batchsize, 800, 256), dtype=np.float32)
			langs = list()

			for i in range(start, start + self.batchsize):
				name, answer = specs[i].split(',')
				langs.append(int(answer))
				img = Image.open(self.datapath + "trainingData/spectrograms/" + name + ".png")
				data[i - start, :, :] = np.transpose(np.array(img).astype(np.float32) / 256.0)

			langs = np.array(langs, dtype=np.int32)
			return data, langs

		if (mode == "train"):
			data, langs = batchthrough(self.trndata, batch)
			theano_fn = self.train_fn
		elif (mode == "test"):
			data, langs = batchthrough(self.tstdata, batch)
			theano_fn = self.test_fn
		batchprediction, batchloss = theano_fn(data, langs)
		return {"batchprediction": batchprediction,"langs": langs,"batchloss": batchloss,}
