# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
import json
import importlib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="small", help='Choose small or large dataset')
parser.add_argument('--model', type=str, default="rnn", help='Choose cnn,rnn or combine')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='Number of images in one batch')
parser.add_argument('--rnnunits', type=int, default=100, help='RNN hidden units')

args = parser.parse_args()

if args.dataset == "small":
    datapath = "../../Data/small/"
    dataclass = 5
elif args.dataset == "large":
    datapath = "../../Data/large/"
    dataclass = 176
else:
    raise Exception('Dataset can only be small or large!')

trndata = open(datapath + "trainset.csv", "r").readlines()
tstdata = open(datapath + "testset.csv", "r").readlines()

args_dict = dict(args._get_kwargs())
args_dict['trndata'] = trndata
args_dict['tstdata'] = tstdata
args_dict['dataclass'] = dataclass
args_dict['datapath'] = datapath

models = importlib.import_module("networks." + args.model)
model = models.Model(**args_dict)

print "Model specifications and training parameters are the following:"
print "Model structure is %s" % args.model
print "There are %d training examples and %d validation examples" %(len(trndata), len(tstdata))
print "Model will be trained for %d" %args.epochs
print "Batchsize is %d" % args.batchsize
if args.model == "rnn" or args.model == "combine":
    print "Number of hidden RNN units is %d" % args.rnnunits
print "Using %s dataset..." %args.dataset

def training(mode, trndatasize, tstdatasize, epoch):

    loss = 0
    accuracy = 0
    if (mode == 'train'):
        batches = trndatasize / args.batchsize
    elif (mode == 'test'):
        batches = tstdatasize / args.batchsize

    for i in range(batches):
        sys.stdout.write('\r%s %d (%s)\t |%s| %s%% %s' % ('Epoch No.', epoch+1, mode, '█' * int(50 * (i+1) // batches) + '-' * (50 - int(50 * (i+1) // batches)), ("{0:." + str(1) + "f}").format(100 * ((i+1) / float(batches))), 'Complete'))

        step_data = model.step(i, mode)
        loss += step_data["batchloss"]
        accuracy += sum(1 for x,y in zip(step_data["batchprediction"].argmax(axis=1),step_data["langs"]) if x == y)

        sys.stdout.flush()
    print ":----->\t%s accuracy: %f%%" % (mode, accuracy * 100.0 / batches / args.batchsize)

    return loss / batches

for epoch in range(args.epochs):
    training('train', len(trndata), len(tstdata), epoch)
    training('test', len(trndata), len(tstdata), epoch)
