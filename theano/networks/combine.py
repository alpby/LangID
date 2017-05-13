import numpy as np
import theano
import lasagne
import PIL.Image as Image

class Model:

    def __init__(self, dataclass, trndata, tstdata, batchsize, dropout, rnnunits, **kwargs):
        def convpool(model,filters,filterSize,stride,pooling,poolingStride,padding):
            model = lasagne.layers.Conv2DLayer(model,filters,filterSize,stride)
            model = lasagne.layers.MaxPool2DLayer(model,pooling,poolingStride,padding)
            model = lasagne.layers.BatchNormLayer(model)
            return model

        self.trndata = trndata
        self.tstdata = tstdata
        self.batchsize = batchsize
        self.dropout = dropout
        self.units = rnnunits
        self.dataclass = dataclass

        self.spectrograms = theano.tensor.tensor4('spectrograms')
        self.langs = theano.tensor.ivector('langs')

        model = lasagne.layers.InputLayer((None, 1, 256, 856), self.spectrograms)

        model = convpool(model,16,(7,7),1,(3,3),2,2)
        model = convpool(model,32,(5,5),1,(3,3),2,2)
        model = convpool(model,32,(3,3),1,(3,3),2,2)
        model = convpool(model,32,(3,3),1,(3,3),2,2)

        self.params = lasagne.layers.get_all_params(model, trainable=True)

        output = lasagne.layers.get_output(model)
        output = output.transpose((0, 3, 1, 2))
        output = output.flatten(ndim=3)

        # NOTE: these constants are shapes of last pool layer, it can be symbolic
        # explicit values are better for optimizations
        num_channels = 64
        filter_W = 54
        filter_H = 8

        # InputLayer
        model = lasagne.layers.InputLayer((None, filter_W, num_channels * filter_H), output)
        # GRULayer
        model = lasagne.layers.GRULayer(model, self.units, only_return_final=True)
        model = lasagne.layers.BatchNormLayer(model)
        if (self.dropout > 0):
            model = lasagne.layers.dropout(model, self.dropout)

        # Last layer: classification
        model = lasagne.layers.DenseLayer(model, self.dataclass, nonlinearity=lasagne.nonlinearities.softmax)

        self.params += lasagne.layers.get_all_params(model, trainable=True)
        self.prediction = lasagne.layers.get_output(model)

        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.langs).mean()

        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=0.003)

        self.train_fn = theano.function([self.spectrograms, self.langs],[self.prediction, self.loss],updates=updates)
        self.test_fn = theano.function([self.spectrograms,self.langs],[self.prediction, self.loss])

    def step(self, batch, mode):
        def read_batch(specs, batch):

            start = batch * self.batchsize
            data = np.zeros((self.batchsize, 1, 256, 856), dtype=np.float32)
            answers = list()

            for i in range(start, start + self.batchsize):
                name, answer = specs[i].split(',')
                answers.append(int(answer))
                img = Image.open("../../Data/small/trainingData/spectrograms/" + name + ".png")
                data[i - start, 0, :, :] = np.array(img).astype(np.float32) / 256.0

            answers = np.array(answers, dtype=np.int32)
            return data, answers

        if (mode == "train"):
            data, answers = read_batch(self.trndata, batch)
            theano_fn = self.train_fn
        elif (mode == "test"):
            data, answers = read_batch(self.tstdata, batch)
            theano_fn = self.test_fn
        ret = theano_fn(data, answers)
        return {"prediction": ret[0],"answers": answers,"current_loss": ret[1],}
