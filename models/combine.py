import numpy as np
import theano
import lasagne
from PIL.Image import open

class Model:
    def __init__(self, datapath, dataclass, trndata, tstdata, batchsize, rnnunits, **kwargs):
        # TODO: L2 regularization may be applied here
        # Try with different optimizers
        # Apply dropout
        # Try with different sized filters
        # Try with different and multiple RNN models

        # NOTE: I created function below to apply convolution, max-pooling and batch normalization altogether.
        def convpool(model,filters,filterSize,stride,pooling,poolingStride,padding):
            model = lasagne.layers.Conv2DLayer(model,filters,filterSize,stride)
            model = lasagne.layers.MaxPool2DLayer(model,pooling,poolingStride,padding)
            model = lasagne.layers.BatchNormLayer(model)
            return model

        self.trndata = trndata          # Training data
        self.tstdata = tstdata          # Test data
        self.batchsize = batchsize      # Batchsize
        self.units = rnnunits           # RNN hidden size
        self.dataclass = dataclass      # Number of languages to classify
        self.datapath = datapath        # Path to small or large dataset

        self.spectrograms = theano.tensor.tensor4("spectrograms") # Input tensor is three dimensional in RNN case
        self.langs = theano.tensor.ivector("langs")               # Output is a vector with "dataclass" dimension

        # NOTE: Inputs are different sized so I had to crop plots. They have size of (200,800), since high frequencies are not that important and training takes shorter.
        # You can add or remove convpool layer below.
        model = lasagne.layers.InputLayer((None, 1, 256, 800), self.spectrograms)

        model = convpool(model,16,(7,7),1,3,2,2)
        model = convpool(model,32,(5,5),1,3,2,2)
        model = convpool(model,32,(3,3),1,3,2,1)
        model = convpool(model,32,(3,3),1,3,2,1)

        self.weights = lasagne.layers.get_all_params(model, trainable=True)

        # NOTE: The output of CNN layers will be feeded into RNN here.
        # TODO: There is no transpose and flatten functionality in theano. "dimshuffle" only works for transpose and expand_dims. Create this functionality.
        output = lasagne.layers.get_output(model)
        output = output.transpose((0, 3, 1, 2))
        output = output.flatten(ndim=3)

        model = lasagne.layers.InputLayer((None, 32, 448), output)
        model = lasagne.layers.GRULayer(model, self.units, only_return_final=True)
        model = lasagne.layers.BatchNormLayer(model)

        # NOTE: Below where the classification happens. Softmax classifier used to output the language with highest probability.
        model = lasagne.layers.DenseLayer(model, self.dataclass, nonlinearity=lasagne.nonlinearities.softmax)

        self.weights += lasagne.layers.get_all_params(model, trainable=True)
        self.prediction = lasagne.layers.get_output(model)

        # NOTE: Cost function is the result of cross entropy function.
        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.langs).mean()
        opts = lasagne.updates.adam(self.loss, self.weights, learning_rate=0.003)

        # NOTE: The train and test sets are feeded into same network, no backpropagation for test set though.
        self.runTrain = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss],updates=opts)
        self.runTest = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss])

    def forward(self, batch, mode):
        def batchthrough(specs, batch):

            start = batch * self.batchsize
            data = np.zeros((self.batchsize, 1, 256, 800), dtype=np.float32)
            langs = list()

            for i in range(start, start + self.batchsize):
                name, lang = specs[i].split(',')
                langs.append(int(lang))
                img = open(self.datapath + "/data/spectrograms/" + name + ".png")
                data[i - start, 0, :, :] = np.array(img).astype(np.float32) / 256.0

            langs = np.array(langs, dtype=np.int32)
            return data, langs

        if (mode == "train"):
            data, langs = batchthrough(self.trndata, batch)
            f = self.runTrain
        elif (mode == "test"):
            data, langs = batchthrough(self.tstdata, batch)
            f = self.runTest
        batchprediction, batchloss = f(data, langs)
        return {"batchprediction": batchprediction,"langs": langs,"batchloss": batchloss,}
