# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
import json
import importlib
import os

# NOTE: Datasets are explained in "preprocess.py"
# There are three different models I applied for language identification problem.
# Model "cnn" consists of only CNN layers with various number of filters.
# Model "rnn" has two GRU/LSTM layers. Number of RNN units can be set manually.
# Model "combine" has CNN layers and at the end there is GRU/LSTM layer.
# You can use minibatching by setting batchsize.

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="small", help='Choose small or large dataset')
parser.add_argument('--model', type=str, default="rnn", help='Choose cnn,rnn or combine')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=30, help='Number of images in one batch')
parser.add_argument('--rnnunits', type=int, default=100, help='RNN hidden units')
parser.add_argument('--langs', type=int, default=5, help='Number of classes to be used')

args = parser.parse_args()

# NOTE: Dataclass variable represents the number of different languages each dataset includes.
if args.dataset == "small":
    datapath = "../Data/small/"
elif args.dataset == "large":
    datapath = "../Data/large/"
else:
    raise Exception('Dataset can only be small or large!')

trndata = open(datapath + "trainset.csv", "r").readlines()
tstdata = open(datapath + "testset.csv", "r").readlines()

args_dict = dict(args._get_kwargs())
args_dict['trndata'] = trndata
args_dict['tstdata'] = tstdata
args_dict['dataclass'] = args.langs
args_dict['datapath'] = datapath

models = importlib.import_module("models." + args.model)
model = models.Model(**args_dict)

print "Model specifications and training parameters are the following:"
print "Model structure is %s." % args.model
print "Using %s dataset..." %args.dataset
print "%d languages are to be classfied..." %args.langs
print "There are %d training examples and %d validation examples." %(len(trndata), len(tstdata))
print "Model will be trained for %d epochs." %args.epochs
print "Batchsize is %d." % args.batchsize
if args.model == "rnn" or args.model == "combine":
    print "Number of hidden RNN units is %d." % args.rnnunits

def training(mode, trndatasize, tstdatasize, epoch):
    cost = 0
    accuracy = 0
    if (mode == 'train'):
        batches = trndatasize / args.batchsize
    elif (mode == 'test'):
        batches = tstdatasize / args.batchsize

    for i in range(batches):
        # Fancy progress bar
        sys.stdout.write('\r%s %d (%s)\t |%s| %s%% %s' % ('Epoch No.', epoch+1, mode, '█' * int(50 * (i+1) // batches) + '-' * (50 - int(50 * (i+1) // batches)), ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(batches))), 'Complete'))

        forwardRun = model.forward(i, mode)
        cost += forwardRun["batchloss"]
        accuracy += sum(1 for x,y in zip(forwardRun["batchprediction"].argmax(axis=1),forwardRun["langs"]) if x == y)

        sys.stdout.flush()
    print ":----->\t%s accuracy: %f%%" % (mode, accuracy * 100.0 / batches / args.batchsize)

    return forwardRun["batchprediction"].argmax(axis=1), forwardRun["langs"], cost / batches

def recall(truepos,falseneg):
    return truepos/float(truepos + falseneg)

def precision(truepos,falsepos):
    return truepos/float(truepos + falsepos)

for epoch in range(args.epochs):
    training('train', len(trndata), len(tstdata), epoch)
    pred, ygold, _ = training('test', len(trndata), len(tstdata), epoch)

# NOTE: It is only working when whole test set is fed into model at once.
if len(tstdata) == args.batchsize:
    truepos = args.langs*[0]
    falsepos = args.langs*[0]
    falseneg = args.langs*[0]

    for i in range(len(pred)):
        if pred[i] == ygold[i]:
            truepos[pred[i]] += 1
        else:
            falsepos[pred[i]] += 1
            falseneg[ygold[i]] += 1

    print ""
    for i in range(args.langs):
        print "For language number %d recall is %.2f and precision is %.2f" %(i + 1, 100*recall(truepos[i],falseneg[i]), 100*precision(truepos[i],falsepos[i]))
