import numpy as np
import theano
import lasagne
import PIL.Image as Image

class Model:

    def __init__(self, datapath, dataclass, trndata, tstdata, batchsize, **kwargs):
        # TODO: L2 regularization may be applied here
        # Try with different optimizers
        # Apply dropout
        # Try with different sized filters
        def convpool(model,filters,filterSize,stride,pooling,poolingStride,padding):
            model = lasagne.layers.Conv2DLayer(model,filters,filterSize,stride)
            model = lasagne.layers.MaxPool2DLayer(model,pooling,poolingStride,padding)
            model = lasagne.layers.BatchNormLayer(model)
            return model

        self.trndata = trndata
        self.tstdata = tstdata
        self.batchsize = batchsize
        self.dataclass = dataclass
        self.datapath = datapath

        self.spectrograms = theano.tensor.tensor4('spectrograms')
        self.langs = theano.tensor.ivector('langs')

        model = lasagne.layers.InputLayer((None, 1, 256, 800), self.spectrograms)

        model = convpool(model,16,(7,7),1,(3,3),2,2)
        model = convpool(model,32,(5,5),1,(3,3),2,2)
        model = convpool(model,64,(3,3),1,(3,3),2,2)
        model = convpool(model,128,(3,3),1,(3,3),2,2)
        model = convpool(model,128,(3,3),1,(3,3),2,2)
        model = convpool(model,256,(3,3),1,(3,3),(3,2),2)

        model = lasagne.layers.DenseLayer(model, 1024)
        model = lasagne.layers.BatchNormLayer(model)

        model = lasagne.layers.DenseLayer(model, self.dataclass, nonlinearity=lasagne.nonlinearities.softmax)

        self.params = lasagne.layers.get_all_params(model, trainable=True)
        self.prediction = lasagne.layers.get_output(model)

        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.langs).mean()
        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=0.003)

        self.train_fn = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss],updates=updates)
        self.test_fn = theano.function([self.spectrograms,self.langs],[self.prediction,self.loss])

    def step(self, batch, mode):
        def read_batch(specs, batch):

            start = batch * self.batchsize
            data = np.zeros((self.batchsize, 1, 256, 800), dtype=np.float32)
            langs = list()

            for i in range(start, start + self.batchsize):
                name, lang = specs[i].split(',')
                langs.append(int(lang))
                img = Image.open(self.datapath + "/trainingData/spectrograms/" + name + ".png")
                data[i - start, 0, :, :] = np.array(img).astype(np.float32) / 256.0

            langs = np.array(langs, dtype=np.int32)
            return data, langs

        if (mode == "train"):
            data, langs = read_batch(self.trndata, batch)
            f = self.train_fn
        elif (mode == "test"):
            data, langs = read_batch(self.tstdata, batch)
            f = self.test_fn
        batchprediction, batchloss = f(data, langs)
        return {"batchprediction": batchprediction,"langs": langs,"batchloss": batchloss,}
